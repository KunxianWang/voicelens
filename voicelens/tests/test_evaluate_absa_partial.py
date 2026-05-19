from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_absa import filter_labeled_rows, is_labeled
from scripts.evaluate_absa import main as eval_main


def _labeled_row(rid: int) -> dict:
    return {
        "review_id": rid,
        "text_raw": "Battery is great.",
        "gold_aspects": [
            {
                "aspect_code": "battery",
                "sentiment": "positive",
                "severity": None,
                "evidence_quote": "Battery is great",
            }
        ],
        "predicted_aspects": [
            {
                "aspect_code": "battery",
                "sentiment": "positive",
                "severity": None,
                "evidence_quote": "Battery is great",
            }
        ],
    }


def _unlabeled_row(rid: int) -> dict:
    return {
        "review_id": rid,
        "text_raw": "Bought this last week.",
        "gold_aspects": [],
        "predicted_aspects": [],
    }


def _no_aspect_but_labeled_row(rid: int) -> dict:
    return {
        "review_id": rid,
        "text_raw": "Bought this last week.",
        "gold_aspects": [],
        "labeled": True,
        "predicted_aspects": [],
    }


def test_is_labeled_handles_three_states():
    assert is_labeled(_labeled_row(1)) is True
    assert is_labeled(_unlabeled_row(2)) is False
    assert is_labeled(_no_aspect_but_labeled_row(3)) is True


def test_filter_labeled_rows_ignores_unlabeled():
    rows = [_labeled_row(1), _unlabeled_row(2), _labeled_row(3)]
    out = filter_labeled_rows(rows)
    assert [r["review_id"] for r in out] == [1, 3]


def test_filter_labeled_rows_max_rows_caps_post_filter():
    rows = [_labeled_row(i) for i in range(10)] + [_unlabeled_row(99)]
    out = filter_labeled_rows(rows, max_rows=3)
    assert [r["review_id"] for r in out] == [0, 1, 2]


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "labeled.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_partial_eval_skips_unlabeled_and_caps_at_max_rows(tmp_path: Path, capsys):
    rows = [_labeled_row(i) for i in range(25)] + [_unlabeled_row(99)]
    path = _write(tmp_path, rows)

    rc = eval_main(
        [
            "--labeled",
            str(path),
            "--predictions",
            str(tmp_path / "no_such_predictions.jsonl"),
            "--max-rows",
            "5",
            "--require-min-labeled",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    metrics = json.loads(out)
    assert metrics["evaluated_rows"] == 5
    assert metrics["unlabeled_rows_skipped"] == 21
    assert metrics["total_rows_in_file"] == 26
    assert metrics["max_rows_cap"] == 5


def test_partial_eval_fails_with_too_few_labeled_rows(tmp_path: Path, capsys):
    rows = [_labeled_row(i) for i in range(3)] + [_unlabeled_row(99)]
    path = _write(tmp_path, rows)

    rc = eval_main(
        [
            "--labeled",
            str(path),
            "--predictions",
            str(tmp_path / "no_such_predictions.jsonl"),
            "--require-min-labeled",
            "20",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 3
    assert "not enough labeled rows" in err
    assert "labeled = 3" in err
    assert "required >= 20" in err


def test_partial_eval_missing_file_returns_clear_error(tmp_path: Path, capsys):
    rc = eval_main(["--labeled", str(tmp_path / "missing.jsonl")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "labeled ABSA holdout file is missing" in err


def test_partial_eval_disables_min_with_zero(tmp_path: Path):
    path = _write(tmp_path, [_labeled_row(1)])
    rc = eval_main(
        [
            "--labeled",
            str(path),
            "--predictions",
            str(tmp_path / "none.jsonl"),
            "--require-min-labeled",
            "0",
        ]
    )
    assert rc == 0
