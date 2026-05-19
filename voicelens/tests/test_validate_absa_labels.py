from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_absa_labels import validate_labeled_rows


def _row(
    *,
    review_id: int = 1,
    text_raw: str = "The battery dies after two hours.",
    gold_aspects: list[dict] | None = None,
    labeled: bool | None = None,
) -> dict:
    payload: dict = {
        "review_id": review_id,
        "source_id": f"R{review_id}",
        "brand": "Anker",
        "asin": "B0ANK",
        "rating": 1,
        "text_raw": text_raw,
        "gold_aspects": gold_aspects if gold_aspects is not None else [],
    }
    if labeled is not None:
        payload["labeled"] = labeled
    return payload


def test_valid_labeled_file_passes():
    rows = [
        _row(
            gold_aspects=[
                {
                    "aspect_code": "battery",
                    "sentiment": "negative",
                    "severity": "medium",
                    "evidence_quote": "battery dies after two hours",
                }
            ]
        ),
        _row(
            review_id=2,
            text_raw="Great sound and works fine.",
            gold_aspects=[
                {
                    "aspect_code": "sound_quality",
                    "sentiment": "positive",
                    "severity": None,
                    "evidence_quote": "Great sound",
                }
            ],
        ),
    ]
    report = validate_labeled_rows(rows)

    assert report.is_valid
    assert report.invalid_rows == 0
    assert report.labeled_rows == 2
    assert report.unlabeled_rows == 0
    assert report.aspect_distribution["battery"] == 1
    assert report.aspect_distribution["sound_quality"] == 1
    assert report.sentiment_distribution["negative"] == 1
    assert report.severity_distribution["medium"] == 1


def test_invalid_aspect_code_fails():
    rows = [
        _row(
            gold_aspects=[
                {
                    "aspect_code": "made_up_aspect",
                    "sentiment": "positive",
                    "severity": None,
                    "evidence_quote": "battery",
                }
            ]
        )
    ]
    report = validate_labeled_rows(rows)

    assert not report.is_valid
    assert report.invalid_rows == 1
    assert any("not in ontology" in err for err in report.errors)


def test_non_verbatim_evidence_quote_fails():
    rows = [
        _row(
            text_raw="The battery dies after two hours.",
            gold_aspects=[
                {
                    "aspect_code": "battery",
                    "sentiment": "negative",
                    "severity": "low",
                    "evidence_quote": "battery is bad",
                }
            ],
        )
    ]
    report = validate_labeled_rows(rows)

    assert not report.is_valid
    assert any("verbatim substring" in err for err in report.errors)


def test_severity_required_on_negative_sentiment():
    rows = [
        _row(
            gold_aspects=[
                {
                    "aspect_code": "battery",
                    "sentiment": "negative",
                    "severity": None,
                    "evidence_quote": "battery dies after two hours",
                }
            ]
        )
    ]
    report = validate_labeled_rows(rows)

    assert not report.is_valid
    assert any("severity" in err.lower() and "required" in err for err in report.errors)


def test_severity_forbidden_on_non_negative_sentiment():
    rows = [
        _row(
            text_raw="Great sound and works fine.",
            gold_aspects=[
                {
                    "aspect_code": "sound_quality",
                    "sentiment": "positive",
                    "severity": "low",
                    "evidence_quote": "Great sound",
                }
            ],
        )
    ]
    report = validate_labeled_rows(rows)

    assert not report.is_valid
    assert any("severity must be null" in err for err in report.errors)


def test_duplicate_aspect_code_within_review_fails():
    rows = [
        _row(
            gold_aspects=[
                {
                    "aspect_code": "battery",
                    "sentiment": "negative",
                    "severity": "low",
                    "evidence_quote": "battery dies after two hours",
                },
                {
                    "aspect_code": "battery",
                    "sentiment": "negative",
                    "severity": "high",
                    "evidence_quote": "battery dies",
                },
            ]
        )
    ]
    report = validate_labeled_rows(rows)

    assert not report.is_valid
    assert any("duplicate" in err for err in report.errors)


def test_empty_gold_aspects_treated_as_unlabeled_by_default():
    rows = [_row(gold_aspects=[])]
    report = validate_labeled_rows(rows)

    assert report.is_valid
    assert report.labeled_rows == 0
    assert report.unlabeled_rows == 1


def test_labeled_true_flag_marks_no_aspect_row_as_labeled():
    rows = [_row(gold_aspects=[], labeled=True)]
    report = validate_labeled_rows(rows)

    assert report.is_valid
    assert report.labeled_rows == 1
    assert report.unlabeled_rows == 0


def test_missing_gold_aspects_field_fails():
    bad = {
        "review_id": 1,
        "text_raw": "anything",
    }
    report = validate_labeled_rows([bad])

    assert not report.is_valid
    assert any("missing 'gold_aspects'" in err for err in report.errors)


def test_invalid_sentiment_fails():
    rows = [
        _row(
            gold_aspects=[
                {
                    "aspect_code": "battery",
                    "sentiment": "ambivalent",
                    "severity": None,
                    "evidence_quote": "battery",
                }
            ]
        )
    ]
    report = validate_labeled_rows(rows)

    assert not report.is_valid
    assert any("sentiment" in err and "must be one of" in err for err in report.errors)


def test_validator_main_exit_codes(tmp_path: Path):
    from scripts.validate_absa_labels import main

    missing = tmp_path / "missing.jsonl"
    assert main(["--labeled", str(missing)]) == 2

    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps(
            _row(
                gold_aspects=[
                    {
                        "aspect_code": "battery",
                        "sentiment": "negative",
                        "severity": "low",
                        "evidence_quote": "battery dies after two hours",
                    }
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--labeled", str(good)]) == 0

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            _row(
                gold_aspects=[
                    {
                        "aspect_code": "not_real",
                        "sentiment": "negative",
                        "severity": "low",
                        "evidence_quote": "battery",
                    }
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--labeled", str(bad)]) == 1
