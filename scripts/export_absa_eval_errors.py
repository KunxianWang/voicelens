"""Export a human-readable CSV of ABSA prediction errors against the
labeled holdout.

Granularity: one CSV row per ``(review, aspect_code)`` discrepancy. A
review whose gold and predicted aspect sets match perfectly (same codes,
same sentiment, same severity, all evidence_quotes verbatim) writes a
single ``error_type=correct`` row so the export always has at least one
row per labeled review.

Error types, evaluated in this priority order:

- ``missed_aspect``        aspect_code is in gold but not in predictions
- ``extra_aspect``         aspect_code is in predictions but not in gold
- ``non_verbatim_evidence``predicted evidence_quote not in review text
- ``wrong_sentiment``      gold + pred agree on aspect_code, sentiment differs
- ``wrong_severity``       gold + pred agree on aspect + sentiment, severity differs
- ``correct``              every aspect matches; one summary row per review
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Defaults mirror scripts/evaluate_absa.py. They are inlined here so this
# script runs cleanly both as ``python scripts/export_absa_eval_errors.py``
# (where ``scripts`` is not importable as a package) and as
# ``python -m scripts.export_absa_eval_errors``.
DEFAULT_LABELED = Path("data/labeling/absa_holdout_labeled.jsonl")
DEFAULT_PREDICTIONS = Path("data/labeling/absa_holdout_predictions.jsonl")
DEFAULT_OUTPUT = Path("data/labeling/absa_eval_errors.csv")


def _is_labeled(row: dict[str, Any]) -> bool:
    if row.get("labeled") is True or row.get("reviewed") is True:
        return True
    gold = row.get("gold_aspects")
    return isinstance(gold, list) and len(gold) > 0


def filter_labeled_rows(
    rows: list[dict[str, Any]], *, max_rows: int | None = None
) -> list[dict[str, Any]]:
    labeled = [row for row in rows if _is_labeled(row)]
    if max_rows is not None and max_rows > 0:
        labeled = labeled[:max_rows]
    return labeled

ERROR_MISSED = "missed_aspect"
ERROR_EXTRA = "extra_aspect"
ERROR_NON_VERBATIM = "non_verbatim_evidence"
ERROR_WRONG_SENTIMENT = "wrong_sentiment"
ERROR_WRONG_SEVERITY = "wrong_severity"
CORRECT = "correct"

ERROR_TYPES: tuple[str, ...] = (
    ERROR_MISSED,
    ERROR_EXTRA,
    ERROR_NON_VERBATIM,
    ERROR_WRONG_SENTIMENT,
    ERROR_WRONG_SEVERITY,
    CORRECT,
)

CSV_COLUMNS: tuple[str, ...] = (
    "review_id",
    "brand",
    "rating",
    "text_raw",
    "aspect_code",
    "gold_aspects",
    "predicted_aspects",
    "error_type",
    "notes",
)


@dataclass
class ErrorRow:
    review_id: Any
    brand: Any
    rating: Any
    text_raw: str
    aspect_code: str | None
    gold_aspects: list[dict[str, Any]]
    predicted_aspects: list[dict[str, Any]]
    error_type: str
    notes: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "review_id": "" if self.review_id is None else str(self.review_id),
            "brand": "" if self.brand is None else str(self.brand),
            "rating": "" if self.rating is None else str(self.rating),
            "text_raw": self.text_raw,
            "aspect_code": self.aspect_code or "",
            "gold_aspects": json.dumps(self.gold_aspects, ensure_ascii=False),
            "predicted_aspects": json.dumps(self.predicted_aspects, ensure_ascii=False),
            "error_type": self.error_type,
            "notes": self.notes,
        }


def _by_code(aspects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for aspect in aspects:
        if isinstance(aspect, dict):
            code = aspect.get("aspect_code")
            if isinstance(code, str) and code not in out:
                out[code] = aspect
    return out


def _aspects_field(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = row.get(key) or []
    return [a for a in value if isinstance(a, dict)] if isinstance(value, list) else []


def classify_review_errors(row: dict[str, Any]) -> list[ErrorRow]:
    review_id = row.get("review_id")
    brand = row.get("brand")
    rating = row.get("rating")
    text_raw = str(row.get("text_raw") or "")
    gold_aspects = _aspects_field(row, "gold_aspects")
    pred_aspects = _aspects_field(row, "predicted_aspects")
    gold = _by_code(gold_aspects)
    pred = _by_code(pred_aspects)
    codes = sorted(set(gold) | set(pred))

    errors: list[ErrorRow] = []
    for code in codes:
        in_gold = code in gold
        in_pred = code in pred
        if in_gold and not in_pred:
            errors.append(
                ErrorRow(
                    review_id=review_id,
                    brand=brand,
                    rating=rating,
                    text_raw=text_raw,
                    aspect_code=code,
                    gold_aspects=[gold[code]],
                    predicted_aspects=[],
                    error_type=ERROR_MISSED,
                    notes=f"gold has {code!r} but predictions do not",
                )
            )
            continue
        if in_pred and not in_gold:
            errors.append(
                ErrorRow(
                    review_id=review_id,
                    brand=brand,
                    rating=rating,
                    text_raw=text_raw,
                    aspect_code=code,
                    gold_aspects=[],
                    predicted_aspects=[pred[code]],
                    error_type=ERROR_EXTRA,
                    notes=f"prediction has {code!r} but gold does not",
                )
            )
            continue
        # Aspect present in both — check evidence, then sentiment, then severity.
        pred_aspect = pred[code]
        gold_aspect = gold[code]
        pred_quote = pred_aspect.get("evidence_quote")
        if isinstance(pred_quote, str) and pred_quote and pred_quote not in text_raw:
            errors.append(
                ErrorRow(
                    review_id=review_id,
                    brand=brand,
                    rating=rating,
                    text_raw=text_raw,
                    aspect_code=code,
                    gold_aspects=[gold_aspect],
                    predicted_aspects=[pred_aspect],
                    error_type=ERROR_NON_VERBATIM,
                    notes="predicted evidence_quote not a substring of review text",
                )
            )
            continue
        gold_sent = gold_aspect.get("sentiment")
        pred_sent = pred_aspect.get("sentiment")
        if gold_sent != pred_sent:
            errors.append(
                ErrorRow(
                    review_id=review_id,
                    brand=brand,
                    rating=rating,
                    text_raw=text_raw,
                    aspect_code=code,
                    gold_aspects=[gold_aspect],
                    predicted_aspects=[pred_aspect],
                    error_type=ERROR_WRONG_SENTIMENT,
                    notes=f"gold sentiment={gold_sent!r}, pred sentiment={pred_sent!r}",
                )
            )
            continue
        if gold_sent == "negative":
            gold_sev = gold_aspect.get("severity")
            pred_sev = pred_aspect.get("severity")
            if gold_sev != pred_sev:
                errors.append(
                    ErrorRow(
                        review_id=review_id,
                        brand=brand,
                        rating=rating,
                        text_raw=text_raw,
                        aspect_code=code,
                        gold_aspects=[gold_aspect],
                        predicted_aspects=[pred_aspect],
                        error_type=ERROR_WRONG_SEVERITY,
                        notes=f"gold severity={gold_sev!r}, pred severity={pred_sev!r}",
                    )
                )

    if not errors:
        errors.append(
            ErrorRow(
                review_id=review_id,
                brand=brand,
                rating=rating,
                text_raw=text_raw,
                aspect_code=None,
                gold_aspects=gold_aspects,
                predicted_aspects=pred_aspects,
                error_type=CORRECT,
                notes="all aspects match",
            )
        )
    return errors


def write_eval_errors_csv(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    breakdown: dict[str, int] = {key: 0 for key in ERROR_TYPES}
    all_rows: list[ErrorRow] = []
    for row in rows:
        for err in classify_review_errors(row):
            all_rows.append(err)
            breakdown[err.error_type] = breakdown.get(err.error_type, 0) + 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for err in all_rows:
            writer.writerow(err.to_csv_row())
    return {"rows_written": len(all_rows), "by_error_type": breakdown}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge_predictions(
    labeled: list[dict[str, Any]], predictions_path: Path
) -> list[dict[str, Any]]:
    if all("predicted_aspects" in row for row in labeled):
        return labeled
    if not predictions_path.exists():
        return labeled
    preds = {str(row.get("review_id")): row for row in _read_jsonl(predictions_path)}
    merged: list[dict[str, Any]] = []
    for row in labeled:
        payload = dict(row)
        pred = preds.get(str(row.get("review_id")))
        if pred:
            payload.setdefault("predicted_aspects", pred.get("predicted_aspects", []))
            payload.setdefault("validation_errors", pred.get("validation_errors"))
        else:
            payload.setdefault("predicted_aspects", [])
        merged.append(payload)
    return merged


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ABSA eval errors to CSV for human review.")
    parser.add_argument("--labeled", default=str(DEFAULT_LABELED))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap export at the first N labeled rows (post-filter).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    labeled_path = Path(args.labeled)
    if not labeled_path.exists():
        print(
            f"ERROR: labeled file missing: {labeled_path}\n"
            "Run scripts/sample_absa_holdout.py and label some rows first.",
            file=sys.stderr,
        )
        return 2

    all_rows = _merge_predictions(_read_jsonl(labeled_path), Path(args.predictions))
    labeled_rows = filter_labeled_rows(all_rows, max_rows=args.max_rows)
    if not labeled_rows:
        print(
            "ERROR: no labeled rows found in input.\n"
            "  Fill some gold_aspects in the labeled file first.",
            file=sys.stderr,
        )
        return 3

    output_path = Path(args.output)
    summary = write_eval_errors_csv(labeled_rows, output_path)
    print(f"Wrote {summary['rows_written']} rows to {output_path}")
    print("-- breakdown --")
    for error_type, n in summary["by_error_type"].items():
        print(f"  {error_type:<24} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
