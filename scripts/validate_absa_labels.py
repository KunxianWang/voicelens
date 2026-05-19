"""Validate the hand-labeled ABSA holdout file.

Reads ``data/labeling/absa_holdout_labeled.jsonl`` (one row per labeled
review) and enforces the same schema the LLM is held to:

- each row has ``gold_aspects`` (list, possibly empty)
- each gold aspect has a valid ``aspect_code`` from the v1 ontology
- ``sentiment`` is in ``{positive, neutral, negative}``
- ``severity`` is required iff ``sentiment == "negative"`` and otherwise
  must be null
- ``evidence_quote`` is a verbatim substring of ``text_raw``
- ``aspect_code`` is unique within one review

A row is treated as **labeled** when ``gold_aspects`` is non-empty or
when the labeler set ``"labeled": true`` (the convention used by humans
who reviewed a row and confirmed it has no in-ontology aspects). An
empty ``gold_aspects`` with no ``labeled`` flag is left for future
annotation and reported as ``unlabeled_rows`` instead of failing
validation.

Exits non-zero when any row has a schema violation, so this script
plugs straight into CI before any evaluation runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicelens.nlp.absa.schema import ONTOLOGY_CODES_V1, SENTIMENTS, SEVERITIES

DEFAULT_LABELED = Path("data/labeling/absa_holdout_labeled.jsonl")
SEED_PATH = Path("data/labeling/absa_holdout_seed.jsonl")


@dataclass
class LabelValidationReport:
    total_rows: int = 0
    labeled_rows: int = 0
    unlabeled_rows: int = 0
    invalid_rows: int = 0
    aspect_distribution: Counter[str] = field(default_factory=Counter)
    sentiment_distribution: Counter[str] = field(default_factory=Counter)
    severity_distribution: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.invalid_rows == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "labeled_rows": self.labeled_rows,
            "unlabeled_rows": self.unlabeled_rows,
            "invalid_rows": self.invalid_rows,
            "aspect_distribution": dict(self.aspect_distribution.most_common()),
            "sentiment_distribution": dict(self.sentiment_distribution.most_common()),
            "severity_distribution": dict(self.severity_distribution.most_common()),
            "errors": list(self.errors),
        }


def _row_is_labeled(row: dict[str, Any]) -> bool:
    if row.get("labeled") is True or row.get("reviewed") is True:
        return True
    aspects = row.get("gold_aspects")
    return isinstance(aspects, list) and len(aspects) > 0


def _validate_aspect(
    aspect: Any, text_raw: str, row_index: int, aspect_index: int
) -> list[str]:
    errors: list[str] = []
    prefix = f"row {row_index} aspect {aspect_index}"
    if not isinstance(aspect, dict):
        return [f"{prefix}: must be an object, got {type(aspect).__name__}"]

    code = aspect.get("aspect_code")
    if code not in ONTOLOGY_CODES_V1:
        errors.append(
            f"{prefix}: aspect_code {code!r} is not in ontology {ONTOLOGY_CODES_V1}"
        )

    sentiment = aspect.get("sentiment")
    if sentiment not in SENTIMENTS:
        errors.append(
            f"{prefix}: sentiment {sentiment!r} must be one of {SENTIMENTS}"
        )

    severity = aspect.get("severity")
    if sentiment == "negative":
        if severity not in SEVERITIES:
            errors.append(
                f"{prefix}: severity {severity!r} is required and must be one of "
                f"{SEVERITIES} when sentiment == 'negative'"
            )
    else:
        if severity not in (None, ""):
            errors.append(
                f"{prefix}: severity must be null when sentiment != 'negative' "
                f"(got {severity!r})"
            )

    evidence = aspect.get("evidence_quote")
    if not isinstance(evidence, str) or not evidence.strip():
        errors.append(f"{prefix}: evidence_quote must be a non-empty string")
    elif evidence not in (text_raw or ""):
        errors.append(
            f"{prefix}: evidence_quote {evidence!r} is not a verbatim substring of text_raw"
        )
    return errors


def validate_labeled_rows(rows: list[dict[str, Any]]) -> LabelValidationReport:
    report = LabelValidationReport(total_rows=len(rows))
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            report.invalid_rows += 1
            report.errors.append(f"row {row_index}: not an object")
            continue
        if "gold_aspects" not in row:
            report.invalid_rows += 1
            report.errors.append(f"row {row_index}: missing 'gold_aspects' field")
            continue
        gold = row.get("gold_aspects")
        if not isinstance(gold, list):
            report.invalid_rows += 1
            report.errors.append(
                f"row {row_index}: 'gold_aspects' must be a list, got {type(gold).__name__}"
            )
            continue

        text_raw = row.get("text_raw") or ""
        row_errors: list[str] = []
        seen_codes: set[str] = set()
        for aspect_index, aspect in enumerate(gold, start=1):
            aspect_errors = _validate_aspect(aspect, text_raw, row_index, aspect_index)
            row_errors.extend(aspect_errors)
            if isinstance(aspect, dict):
                code = aspect.get("aspect_code")
                if isinstance(code, str):
                    if code in seen_codes:
                        row_errors.append(
                            f"row {row_index} aspect {aspect_index}: duplicate "
                            f"aspect_code {code!r} in same review"
                        )
                    seen_codes.add(code)
                    if not aspect_errors:
                        report.aspect_distribution[code] += 1
                if not aspect_errors:
                    sentiment = aspect.get("sentiment")
                    severity = aspect.get("severity")
                    if isinstance(sentiment, str):
                        report.sentiment_distribution[sentiment] += 1
                    if isinstance(severity, str):
                        report.severity_distribution[severity] += 1

        if row_errors:
            report.invalid_rows += 1
            report.errors.extend(row_errors)
            continue

        if _row_is_labeled(row):
            report.labeled_rows += 1
        else:
            report.unlabeled_rows += 1
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _print_report(report: LabelValidationReport, path: Path) -> None:
    print(f"== validate_absa_labels.py: {path} ==")
    print(f"  total_rows         : {report.total_rows}")
    print(f"  labeled_rows       : {report.labeled_rows}")
    print(f"  unlabeled_rows     : {report.unlabeled_rows}")
    print(f"  invalid_rows       : {report.invalid_rows}")
    if report.aspect_distribution:
        print("-- aspect distribution (labeled) --")
        for code, n in report.aspect_distribution.most_common():
            print(f"  {code:<16} {n}")
    if report.sentiment_distribution:
        print("-- sentiment distribution (labeled) --")
        for sent, n in report.sentiment_distribution.most_common():
            print(f"  {sent:<10} {n}")
    if report.severity_distribution:
        print("-- severity distribution (labeled, negative-only) --")
        for sev, n in report.severity_distribution.most_common():
            print(f"  {sev:<10} {n}")
    if report.errors:
        print("-- errors --")
        for err in report.errors[:50]:
            print(f"  {err}")
        if len(report.errors) > 50:
            print(f"  ... and {len(report.errors) - 50} more")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate hand-labeled ABSA holdout file.")
    parser.add_argument("--labeled", default=str(DEFAULT_LABELED))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.labeled)
    if not path.exists():
        print(
            "ERROR: labeled file is missing.\n"
            f"Copy {SEED_PATH} to {path} and fill in gold_aspects.",
            file=sys.stderr,
        )
        return 2
    try:
        rows = _read_jsonl(path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: {path} is not valid JSONL: {exc}", file=sys.stderr)
        return 2
    report = validate_labeled_rows(rows)
    _print_report(report, path)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
