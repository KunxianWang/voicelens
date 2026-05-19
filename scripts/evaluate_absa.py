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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ABSA predictions against labeled holdout.")
    parser.add_argument("--labeled", default=str(DEFAULT_LABELED))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
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
    rows = _merge_predictions(_read_jsonl(labeled_path), Path(args.predictions))
    metrics = evaluate_absa_rows(rows)
    print(json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
