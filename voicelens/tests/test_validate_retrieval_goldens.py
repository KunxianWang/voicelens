"""Tests for the retrieval-golden schema validator."""
from __future__ import annotations

from scripts.validate_retrieval_goldens import validate_goldens


def _row(**overrides):
    base = {
        "query_id": "q1",
        "query": "stopped working",
        "expected_aspect": "reliability",
        "expected_sentiment": "negative",
        "expected_brand": None,
        "gold_review_ids": [1, 2, 3],
        "notes": "ok",
    }
    base.update(overrides)
    return base


def test_valid_goldens_have_no_errors():
    errors, warnings = validate_goldens([_row(), _row(query_id="q2")])
    assert errors == []
    assert warnings == []


def test_duplicate_query_id_is_an_error():
    errors, _ = validate_goldens([_row(), _row()])
    assert any("duplicate query_id" in e for e in errors)


def test_missing_required_key_is_an_error():
    bad = _row()
    del bad["gold_review_ids"]
    errors, _ = validate_goldens([bad])
    assert any("missing required key 'gold_review_ids'" in e for e in errors)


def test_non_integer_gold_ids_rejected():
    errors, _ = validate_goldens([_row(gold_review_ids=[1, "2", 3])])
    assert any("must all be integers" in e for e in errors)


def test_duplicate_gold_ids_rejected():
    errors, _ = validate_goldens([_row(gold_review_ids=[1, 1, 2])])
    assert any("duplicate ids" in e for e in errors)


def test_unknown_aspect_rejected():
    errors, _ = validate_goldens([_row(expected_aspect="warranty")])
    assert any("expected_aspect" in e for e in errors)


def test_unknown_sentiment_rejected():
    errors, _ = validate_goldens([_row(expected_sentiment="angry")])
    assert any("expected_sentiment" in e for e in errors)


def test_empty_gold_is_a_warning_not_an_error():
    errors, warnings = validate_goldens([_row(gold_review_ids=[])])
    assert errors == []
    assert any("empty gold_review_ids" in w for w in warnings)


def test_null_aspect_and_brand_are_allowed():
    errors, _ = validate_goldens(
        [_row(expected_aspect=None, expected_sentiment=None, expected_brand=None)]
    )
    assert errors == []
