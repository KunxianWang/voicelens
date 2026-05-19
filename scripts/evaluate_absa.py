"""Score the LLM ABSA predictions against the hand-labeled holdout.

By default reads ``data/labeling/absa_holdout_labeled.jsonl``. The full
1000-row holdout takes hours to label, so the script supports a
*partial-eval* mode that scores whichever rows the labeler has finished:

- ``--max-rows N``      cap the labeled subset at N rows (post-filter).
- ``--require-min-labeled K`` exit non-zero if fewer than K rows are
  labeled (default ``20``). Set to 0 to disable.

A row counts as "labeled" when ``gold_aspects`` is non-empty OR when the
labeler set ``"labeled": true``. Rows that are still placeholders
(``gold_aspects: []`` with no flag) are skipped, not failed — that way
the workflow is "label 50, eval, fix, label more, eval again".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from voicelens.eval.absa_eval import evaluate_absa_rows

DEFAULT_LABELED = Path("data/labeling/absa_holdout_labeled.jsonl")
DEFAULT_SEED = Path("data/labeling/absa_holdout_seed.jsonl")
DEFAULT_PREDICTIONS = Path("data/labeling/absa_holdout_predictions.jsonl")
DEFAULT_REQUIRE_MIN_LABELED = 20


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge_predictions(
    labeled: list[dict[str, Any]],
    predictions_path: Path,
) -> list[dict[str, Any]]:
    if all("predicted_aspects" in row for row in labeled):
        return labeled
    if not predictions_path.exists():
        return labeled
    predictions = {str(row.get("review_id")): row for row in _read_jsonl(predictions_path)}
    merged: list[dict[str, Any]] = []
    for row in labeled:
        payload = dict(row)
        pred = predictions.get(str(row.get("review_id")))
        if pred:
            payload.setdefault("predicted_aspects", pred.get("predicted_aspects", []))
            payload.setdefault("validation_errors", pred.get("validation_errors"))
        else:
            payload.setdefault("predicted_aspects", [])
        merged.append(payload)
    return merged


def is_labeled(row: dict[str, Any]) -> bool:
    """A row is labeled iff gold_aspects has content OR the labeler set a flag."""
    if row.get("labeled") is True or row.get("reviewed") is True:
        return True
    gold = row.get("gold_aspects")
    return isinstance(gold, list) and len(gold) > 0


def filter_labeled_rows(
    rows: list[dict[str, Any]], *, max_rows: int | None = None
) -> list[dict[str, Any]]:
    labeled = [row for row in rows if is_labeled(row)]
    if max_rows is not None and max_rows > 0:
        labeled = labeled[:max_rows]
    return labeled


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ABSA predictions against labeled holdout.")
    parser.add_argument("--labeled", default=str(DEFAULT_LABELED))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap evaluation at the first N labeled rows (post-filter). Default: no cap.",
    )
    parser.add_argument(
        "--require-min-labeled",
        type=int,
        default=DEFAULT_REQUIRE_MIN_LABELED,
        help=(
            "Exit non-zero with a clear message if fewer than this many rows are labeled. "
            "Default: 20. Set to 0 to disable."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    labeled_path = Path(args.labeled)
    if not labeled_path.exists():
        print(
            "ERROR: labeled ABSA holdout file is missing.\n"
            f"Copy {DEFAULT_SEED} to {DEFAULT_LABELED} and fill gold_aspects.",
            file=sys.stderr,
        )
        return 2

    all_rows = _merge_predictions(_read_jsonl(labeled_path), Path(args.predictions))
    labeled_rows = filter_labeled_rows(all_rows, max_rows=args.max_rows)

    if args.require_min_labeled > 0 and len(labeled_rows) < args.require_min_labeled:
        print(
            "ERROR: not enough labeled rows for evaluation.\n"
            f"  labeled = {len(labeled_rows)}, required >= {args.require_min_labeled}\n"
            f"  total rows in {labeled_path} = {len(all_rows)}\n"
            "  Fill more `gold_aspects` (or set `\"labeled\": true` on no-aspect rows), "
            "or rerun with --require-min-labeled 0 to bypass.",
            file=sys.stderr,
        )
        return 3

    metrics = evaluate_absa_rows(labeled_rows)
    metrics["evaluated_rows"] = len(labeled_rows)
    metrics["total_rows_in_file"] = len(all_rows)
    metrics["unlabeled_rows_skipped"] = len(all_rows) - len(labeled_rows)
    if args.max_rows is not None:
        metrics["max_rows_cap"] = args.max_rows
    print(json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
