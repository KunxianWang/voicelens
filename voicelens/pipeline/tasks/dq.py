from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from voicelens.config import (
    ALLOWED_LANGUAGES,
    DQ_MIN_LANG_CONFIDENCE,
    DQ_MIN_TEXT_LEN,
)

CHECK_DUPLICATE = "duplicate_review_id"
CHECK_MISSING_ASIN = "missing_asin"
CHECK_INVALID_RATING = "invalid_rating"
CHECK_NON_ENGLISH = "non_english_language"
CHECK_LOW_LANG_CONFIDENCE = "low_lang_confidence"
CHECK_TEXT_TOO_SHORT = "text_too_short"
CHECK_EMPTY_OR_SPAM = "empty_or_spam"

ALL_CHECKS = [
    CHECK_DUPLICATE,
    CHECK_MISSING_ASIN,
    CHECK_INVALID_RATING,
    CHECK_NON_ENGLISH,
    CHECK_LOW_LANG_CONFIDENCE,
    CHECK_TEXT_TOO_SHORT,
    CHECK_EMPTY_OR_SPAM,
]

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9\s]")


@dataclass
class DQOutcome:
    valid_rows: list[dict[str, Any]] = field(default_factory=list)
    rejected: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    per_check_failed: Counter = field(default_factory=Counter)
    total_rows: int = 0

    @property
    def passed_rows(self) -> int:
        return len(self.valid_rows)

    @property
    def failed_rows(self) -> int:
        return self.total_rows - self.passed_rows

    @property
    def pass_rate(self) -> float:
        return self.passed_rows / self.total_rows if self.total_rows else 0.0


def _row_dedup_key(row: dict[str, Any]) -> str:
    base = "|".join(
        [
            str(row.get("source", "")),
            str(row.get("asin", "")),
            str(row.get("source_id", "")),
            str(row.get("posted_at", "")),
            hashlib.sha1(str(row.get("text_raw", "")).encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _looks_like_spam(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _URL_RE.sub("", stripped).strip() == "":
        return True
    letters = _NON_ALNUM_RE.sub("", stripped)
    if len(letters) >= 10 and letters.isupper():
        return True
    return False


def _check_row(row: dict[str, Any]) -> str | None:
    """Return the first failing check name, or None if all pass.

    Order matters: dedup is handled at the batch level outside this function.
    """
    asin = row.get("asin")
    if not asin or not str(asin).strip():
        return CHECK_MISSING_ASIN

    rating = row.get("rating")
    try:
        rating_int = int(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating_int = None
    if rating_int is None or rating_int < 1 or rating_int > 5:
        return CHECK_INVALID_RATING

    language = (row.get("language") or "").lower()
    if language not in ALLOWED_LANGUAGES:
        return CHECK_NON_ENGLISH

    try:
        lang_conf = float(row.get("lang_confidence", 0.0))
    except (TypeError, ValueError):
        lang_conf = 0.0
    if lang_conf < DQ_MIN_LANG_CONFIDENCE:
        return CHECK_LOW_LANG_CONFIDENCE

    text = row.get("text_raw", "") or ""
    if len(text.strip()) < DQ_MIN_TEXT_LEN:
        return CHECK_TEXT_TOO_SHORT

    if _looks_like_spam(text):
        return CHECK_EMPTY_OR_SPAM

    return None


def run_dq(rows: list[dict[str, Any]]) -> DQOutcome:
    """Apply the MVP DQ checks and return a structured outcome.

    Dedup is computed at batch level (across the input rows). Other checks are
    row-local. Rows that fail any check are routed to ``rejected`` keyed by the
    check name; surviving rows are returned in ``valid_rows`` order-preserved.
    """
    outcome = DQOutcome(total_rows=len(rows))
    seen: set[str] = set()

    for row in rows:
        normalized = dict(row)
        normalized["posted_at"] = _coerce_dt_str(normalized.get("posted_at"))

        key = _row_dedup_key(normalized)
        if key in seen:
            outcome.rejected.setdefault(CHECK_DUPLICATE, []).append(normalized)
            outcome.per_check_failed[CHECK_DUPLICATE] += 1
            continue
        seen.add(key)

        failed = _check_row(normalized)
        if failed is not None:
            outcome.rejected.setdefault(failed, []).append(normalized)
            outcome.per_check_failed[failed] += 1
            continue

        outcome.valid_rows.append(normalized)

    return outcome


def _coerce_dt_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
