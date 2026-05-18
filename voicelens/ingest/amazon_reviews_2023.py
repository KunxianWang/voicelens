"""Adapter for Amazon Reviews 2023 (McAuley Lab, UCSD).

The dataset ships as JSONL (or JSONL.GZ) of two flavors:

- reviews file: per-row {rating, title, text, asin, parent_asin, user_id,
  timestamp, helpful_vote, verified_purchase, ...}
- metadata file: per-row {parent_asin, title, store, details, categories, ...}

The reviews file *does not* contain a brand field. To filter by brand we
either look up brand from the metadata file (preferred) or fall back to
any inline ``brand``/``store`` field that may be on the row.
"""
from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON on line %d of %s: %s", i + 1, path, e)


def canonicalize_brand(raw_brand: str | None, allowlist: tuple[str, ...]) -> str | None:
    """Map a raw brand string to a canonical allowlist entry via case-insensitive
    substring match. Returns the canonical brand or None if no match.

    Handles real-world dirty data like "Anker Innovations", "ANKER", "anker direct".
    """
    if not raw_brand:
        return None
    needle = raw_brand.strip().lower()
    if not needle:
        return None
    for allowed in allowlist:
        if allowed.lower() in needle:
            return allowed
    return None


def load_asin_brand_map(
    metadata_path: Path,
    allowlist: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Stream the Amazon metadata file and return ``{asin: canonical_brand}``.

    Both ``parent_asin`` and ``asin`` keys (if present) are populated so that a
    reviews row can resolve via either. If ``allowlist`` is provided the brand
    is canonicalized; otherwise the raw brand string is kept.
    """
    mapping: dict[str, str] = {}
    for row in _stream_jsonl(metadata_path):
        raw_brand = _extract_brand_from_metadata(row)
        if not raw_brand:
            continue
        brand = canonicalize_brand(raw_brand, allowlist) if allowlist else raw_brand
        if not brand:
            continue
        for key in ("parent_asin", "asin"):
            asin = row.get(key)
            if asin:
                mapping[str(asin)] = brand
    logger.info("Loaded %d asin→brand entries from %s", len(mapping), metadata_path)
    return mapping


def _extract_brand_from_metadata(row: dict[str, Any]) -> str | None:
    details = row.get("details") or {}
    if isinstance(details, dict):
        for k in ("Brand", "brand", "Manufacturer", "manufacturer"):
            if details.get(k):
                return str(details[k])
    for k in ("store", "brand", "manufacturer"):
        if row.get(k):
            return str(row[k])
    return None


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, bool):
            return int(v)
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return None


def _coerce_posted_at(v: Any) -> str | None:
    """Amazon 2023 timestamps are unix epoch in milliseconds. Some forks use
    ISO strings. Normalize to ISO 8601."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        ts = float(v)
        if ts > 10_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=UTC).replace(tzinfo=None).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(v, str):
        s = v.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s).replace(tzinfo=None).isoformat()
        except ValueError:
            return None
    return None


def _normalize_row(
    raw: dict[str, Any],
    asin_to_brand: dict[str, str] | None,
    allowlist: tuple[str, ...] | None,
) -> dict[str, Any] | None:
    asin = raw.get("asin") or raw.get("parent_asin")
    asin = str(asin).strip() if asin else None

    raw_brand: str | None = None
    if asin_to_brand and asin and asin in asin_to_brand:
        raw_brand = asin_to_brand[asin]
    elif raw.get("brand"):
        raw_brand = str(raw["brand"])
    elif raw.get("store"):
        raw_brand = str(raw["store"])

    brand = canonicalize_brand(raw_brand, allowlist) if allowlist else raw_brand

    source_id = raw.get("review_id")
    if not source_id:
        source_id = "_".join(
            str(x) for x in (raw.get("user_id"), asin, raw.get("timestamp")) if x is not None
        )
    if not source_id:
        return None

    rating_raw = raw.get("rating")
    rating: int | None
    if rating_raw is None:
        rating = None
    else:
        try:
            rating = int(round(_coerce_float(rating_raw, default=-1)))
        except (TypeError, ValueError):
            rating = None

    title = (raw.get("title") or "").strip()
    body = (raw.get("text") or raw.get("review_body") or "").strip()
    if title and body:
        text_raw = f"{title}. {body}"
    else:
        text_raw = body or title

    return {
        "source": "amazon_reviews_2023",
        "source_id": str(source_id),
        "asin": asin,
        "brand": brand,
        "model_number": None,
        "category": raw.get("category") or raw.get("main_category"),
        "rating": rating,
        "verified": _coerce_bool(raw.get("verified_purchase")),
        "posted_at": _coerce_posted_at(raw.get("timestamp") or raw.get("posted_at")),
        "helpful_count": _coerce_int(raw.get("helpful_vote") or raw.get("helpful_count"), 0),
        "text_raw": text_raw,
        "language": "en",
        "lang_confidence": 0.95,
        "locale": "en-US",
    }


def iter_amazon_reviews(
    path: Path,
    *,
    asin_to_brand: dict[str, str] | None = None,
    brand_allowlist: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream normalized review rows from an Amazon Reviews 2023 file.

    Filters:
    - rows without a resolvable ASIN are dropped (cannot map to SKU)
    - if ``brand_allowlist`` is set, rows whose brand does not match are dropped
    - ``limit`` caps the number of *matched* rows yielded

    Defensive parsing is applied throughout: malformed JSON lines are skipped,
    missing optional fields fall back to defaults, rating is coerced to int,
    timestamps to ISO 8601.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Amazon reviews file not found: {path}\n"
            f"Place the dataset at this path or set AMAZON_REVIEWS_PATH in .env."
        )

    yielded = 0
    for raw in _stream_jsonl(path):
        normalized = _normalize_row(raw, asin_to_brand, brand_allowlist)
        if normalized is None:
            continue
        if brand_allowlist is not None and not normalized.get("brand"):
            continue
        yield normalized
        yielded += 1
        if limit is not None and yielded >= limit:
            break
