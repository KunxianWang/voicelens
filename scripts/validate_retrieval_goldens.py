"""Schema-validate a retrieval golden set (weak or hand-refined).

Catches the mistakes a manual refinement pass tends to introduce before
they silently skew an eval run: duplicate ``query_id``s, malformed
``gold_review_ids``, an ``expected_aspect`` that is not in the v2
ontology, and so on.

Validation is schema-only — it does **not** need Postgres or Qdrant, so
``make validate-retrieval-goldens-refined`` is fast. Index coverage of
the gold ids is a separate concern handled by
``scripts/analyze_retrieval_goldens.py``.

Exit code is 0 when the file is valid (warnings allowed), 1 on any
error, 2 when the file is missing or unreadable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from voicelens.nlp.absa.schema import SENTIMENTS, ontology_codes

REQUIRED_KEYS = ("query_id", "query", "gold_review_ids")
OPTIONAL_KEYS = ("expected_aspect", "expected_sentiment", "expected_brand", "notes")
DEFAULT_REFINED = Path("data/eval/retrieval_goldens_refined.jsonl")


def validate_goldens(
    rows: list[dict[str, Any]],
    *,
    aspect_version: str = "v2",
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a parsed golden set.

    Errors fail the run; warnings (e.g. an empty gold list) are surfaced
    but tolerated — a query with no gold is legitimately "no signal".
    """
    errors: list[str] = []
    warnings: list[str] = []
    valid_aspects = set(ontology_codes(aspect_version))
    valid_sentiments = set(SENTIMENTS)

    seen_ids: dict[str, int] = {}
    for i, row in enumerate(rows, start=1):
        tag = f"row {i}"
        if not isinstance(row, dict):
            errors.append(f"{tag}: not a JSON object")
            continue

        for key in REQUIRED_KEYS:
            if key not in row:
                errors.append(f"{tag}: missing required key {key!r}")

        qid = row.get("query_id")
        if isinstance(qid, str) and qid.strip():
            tag = f"row {i} ({qid})"
            if qid in seen_ids:
                errors.append(f"{tag}: duplicate query_id (also row {seen_ids[qid]})")
            seen_ids[qid] = i
        elif "query_id" in row:
            errors.append(f"{tag}: query_id must be a non-empty string")

        query = row.get("query")
        if "query" in row and (not isinstance(query, str) or not query.strip()):
            errors.append(f"{tag}: query must be a non-empty string")

        gold = row.get("gold_review_ids")
        if "gold_review_ids" in row:
            if not isinstance(gold, list):
                errors.append(f"{tag}: gold_review_ids must be a list")
            else:
                if not all(isinstance(g, int) and not isinstance(g, bool) for g in gold):
                    errors.append(f"{tag}: gold_review_ids must all be integers")
                elif any(g <= 0 for g in gold):
                    errors.append(f"{tag}: gold_review_ids must be positive")
                elif len(gold) != len(set(gold)):
                    errors.append(f"{tag}: gold_review_ids has duplicate ids")
                if not gold:
                    warnings.append(f"{tag}: empty gold_review_ids (no retrieval signal)")

        aspect = row.get("expected_aspect")
        if aspect is not None and aspect not in valid_aspects:
            errors.append(
                f"{tag}: expected_aspect {aspect!r} not in {aspect_version} ontology "
                f"{sorted(valid_aspects)}"
            )

        sentiment = row.get("expected_sentiment")
        if sentiment is not None and sentiment not in valid_sentiments:
            errors.append(
                f"{tag}: expected_sentiment {sentiment!r} not in {sorted(valid_sentiments)}"
            )

        brand = row.get("expected_brand")
        if brand is not None and (not isinstance(brand, str) or not brand.strip()):
            errors.append(f"{tag}: expected_brand must be null or a non-empty string")

        unknown = set(row) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
        if unknown:
            warnings.append(f"{tag}: unknown keys {sorted(unknown)}")

    return errors, warnings


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_errors.append(f"line {i}: invalid JSON ({exc})")
    return rows, parse_errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schema-validate retrieval goldens.")
    parser.add_argument("--goldens", default=str(DEFAULT_REFINED))
    parser.add_argument("--aspect-version", default="v2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.goldens)
    if not path.exists():
        print(f"ERROR: goldens file {path} does not exist.", file=sys.stderr)
        return 2

    rows, parse_errors = _read_jsonl(path)
    errors, warnings = validate_goldens(rows, aspect_version=args.aspect_version)
    errors = parse_errors + errors

    n_gold = sum(len(r.get("gold_review_ids") or []) for r in rows if isinstance(r, dict))
    print(f"Validated               : {path}")
    print(f"Rows                    : {len(rows)}")
    print(f"Total gold review ids   : {n_gold}")
    for warning in warnings:
        print(f"  WARN  {warning}")
    if errors:
        for error in errors:
            print(f"  ERROR {error}", file=sys.stderr)
        print(f"FAILED: {len(errors)} error(s).", file=sys.stderr)
        return 1
    print(f"OK: schema valid ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
