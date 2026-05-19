from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_absa_eval_errors import (
    CORRECT,
    CSV_COLUMNS,
    ERROR_EXTRA,
    ERROR_MISSED,
    ERROR_NON_VERBATIM,
    ERROR_WRONG_SENTIMENT,
    ERROR_WRONG_SEVERITY,
    classify_review_errors,
    write_eval_errors_csv,
)
from scripts.export_absa_eval_errors import (
    main as export_main,
)

TEXT = "The battery dies after two hours and the bluetooth keeps dropping out."


def _row(*, gold: list[dict], pred: list[dict]) -> dict:
    return {
        "review_id": 1,
        "brand": "Anker",
        "rating": 1,
        "text_raw": TEXT,
        "gold_aspects": gold,
        "predicted_aspects": pred,
    }


def test_classify_correct_when_everything_matches():
    aspect = {
        "aspect_code": "battery",
        "sentiment": "negative",
        "severity": "medium",
        "evidence_quote": "battery dies after two hours",
    }
    rows = classify_review_errors(_row(gold=[aspect], pred=[aspect]))
    assert len(rows) == 1
    assert rows[0].error_type == CORRECT


def test_classify_missed_aspect():
    gold = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery dies after two hours",
        }
    ]
    rows = classify_review_errors(_row(gold=gold, pred=[]))
    assert {r.error_type for r in rows} == {ERROR_MISSED}
    assert rows[0].aspect_code == "battery"


def test_classify_extra_aspect():
    pred = [
        {
            "aspect_code": "charging",
            "sentiment": "neutral",
            "severity": None,
            "evidence_quote": "battery dies",
        }
    ]
    rows = classify_review_errors(_row(gold=[], pred=pred))
    assert {r.error_type for r in rows} == {ERROR_EXTRA}
    assert rows[0].aspect_code == "charging"


def test_classify_non_verbatim_evidence():
    gold = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery dies after two hours",
        }
    ]
    pred = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery is super bad",
        }
    ]
    rows = classify_review_errors(_row(gold=gold, pred=pred))
    assert {r.error_type for r in rows} == {ERROR_NON_VERBATIM}


def test_classify_wrong_sentiment():
    gold = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery dies after two hours",
        }
    ]
    pred = [
        {
            "aspect_code": "battery",
            "sentiment": "positive",
            "severity": None,
            "evidence_quote": "battery dies after two hours",
        }
    ]
    rows = classify_review_errors(_row(gold=gold, pred=pred))
    assert {r.error_type for r in rows} == {ERROR_WRONG_SENTIMENT}


def test_classify_wrong_severity_on_negative_match():
    gold = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery dies after two hours",
        }
    ]
    pred = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "high",
            "evidence_quote": "battery dies after two hours",
        }
    ]
    rows = classify_review_errors(_row(gold=gold, pred=pred))
    assert {r.error_type for r in rows} == {ERROR_WRONG_SEVERITY}


def test_classify_emits_one_row_per_aspect_issue():
    gold = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery dies after two hours",
        },
        {
            "aspect_code": "bluetooth",
            "sentiment": "negative",
            "severity": "low",
            "evidence_quote": "bluetooth keeps dropping out",
        },
    ]
    pred = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery dies after two hours",
        },
    ]
    rows = classify_review_errors(_row(gold=gold, pred=pred))
    assert {r.error_type for r in rows} == {ERROR_MISSED}
    assert {r.aspect_code for r in rows} == {"bluetooth"}


def test_write_eval_errors_csv_writes_expected_columns(tmp_path: Path):
    rows = [
        _row(
            gold=[
                {
                    "aspect_code": "battery",
                    "sentiment": "negative",
                    "severity": "medium",
                    "evidence_quote": "battery dies after two hours",
                }
            ],
            pred=[],
        )
    ]
    out = tmp_path / "errors.csv"
    summary = write_eval_errors_csv(rows, out)

    assert summary["rows_written"] == 1
    assert summary["by_error_type"][ERROR_MISSED] == 1

    with open(out, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == list(CSV_COLUMNS)
        row = next(reader)
    assert row["error_type"] == ERROR_MISSED
    assert row["aspect_code"] == "battery"
    assert json.loads(row["gold_aspects"])[0]["aspect_code"] == "battery"
    assert json.loads(row["predicted_aspects"]) == []


def test_export_main_runs_end_to_end(tmp_path: Path):
    labeled = tmp_path / "labeled.jsonl"
    output = tmp_path / "errors.csv"
    with open(labeled, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                _row(
                    gold=[
                        {
                            "aspect_code": "battery",
                            "sentiment": "negative",
                            "severity": "medium",
                            "evidence_quote": "battery dies after two hours",
                        }
                    ],
                    pred=[
                        {
                            "aspect_code": "charging",
                            "sentiment": "neutral",
                            "severity": None,
                            "evidence_quote": "battery dies",
                        }
                    ],
                )
            )
            + "\n"
        )

    rc = export_main(
        [
            "--labeled",
            str(labeled),
            "--predictions",
            str(tmp_path / "none.jsonl"),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()

    with open(output, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    error_types = {row["error_type"] for row in rows}
    assert ERROR_MISSED in error_types
    assert ERROR_EXTRA in error_types


def test_export_main_missing_file_returns_clear_error(tmp_path: Path, capsys):
    rc = export_main(["--labeled", str(tmp_path / "missing.jsonl")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "labeled file missing" in err


def test_export_main_no_labeled_rows_returns_clear_error(tmp_path: Path, capsys):
    path = tmp_path / "labeled.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "review_id": 1,
                    "brand": "Anker",
                    "rating": 5,
                    "text_raw": "Bought this last week.",
                    "gold_aspects": [],
                    "predicted_aspects": [],
                }
            )
            + "\n"
        )
    rc = export_main(
        [
            "--labeled",
            str(path),
            "--predictions",
            str(tmp_path / "none.jsonl"),
            "--output",
            str(tmp_path / "errors.csv"),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 3
    assert "no labeled rows" in err
