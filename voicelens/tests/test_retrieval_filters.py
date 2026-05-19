from __future__ import annotations

from qdrant_client.http import models as rest

from voicelens.retrieval.filters import build_search_filter


def _conditions(filt: rest.Filter) -> list[rest.FieldCondition]:
    return list(filt.must or [])


def test_no_predicates_returns_none():
    assert build_search_filter() is None


def test_brand_predicate_emits_match_value():
    filt = build_search_filter(brand="Anker")
    assert filt is not None
    [cond] = _conditions(filt)
    assert cond.key == "brand"
    assert cond.match.value == "Anker"


def test_aspect_predicate_targets_aspect_codes_list():
    """A filter on aspect_codes should match if ANY element equals value."""
    filt = build_search_filter(aspect="bluetooth")
    [cond] = _conditions(filt)
    assert cond.key == "aspect_codes"
    assert cond.match.value == "bluetooth"


def test_sentiment_predicate_targets_sentiments_list():
    filt = build_search_filter(sentiment="negative")
    [cond] = _conditions(filt)
    assert cond.key == "sentiments"
    assert cond.match.value == "negative"


def test_rating_range_uses_gte_lte():
    filt = build_search_filter(rating_min=2, rating_max=4)
    [cond] = _conditions(filt)
    assert cond.key == "rating"
    assert cond.range.gte == 2.0
    assert cond.range.lte == 4.0


def test_rating_min_only_emits_one_sided_range():
    filt = build_search_filter(rating_min=4)
    [cond] = _conditions(filt)
    assert cond.range.gte == 4.0
    assert cond.range.lte is None


def test_combined_predicates_all_in_must():
    filt = build_search_filter(
        brand="Anker",
        asin="B0X",
        aspect="reliability",
        sentiment="negative",
        rating_max=2,
        aspect_version="v2",
        provider="anthropic",
        model_name="claude-opus-4.6",
        absa_status="success",
    )
    keys = sorted(cond.key for cond in _conditions(filt))
    assert keys == sorted(
        [
            "brand",
            "asin",
            "aspect_codes",
            "sentiments",
            "rating",
            "aspect_version",
            "provider",
            "model_name",
            "absa_status",
        ]
    )
