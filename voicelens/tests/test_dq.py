from __future__ import annotations

from voicelens.pipeline.tasks.dq import (
    CHECK_DUPLICATE,
    CHECK_EMPTY_OR_SPAM,
    CHECK_INVALID_RATING,
    CHECK_LOW_LANG_CONFIDENCE,
    CHECK_MISSING_ASIN,
    CHECK_NON_ENGLISH,
    CHECK_TEXT_TOO_SHORT,
    run_dq,
)


def test_valid_row_passes(sample_row_factory):
    rows = [sample_row_factory()]
    outcome = run_dq(rows)
    assert outcome.passed_rows == 1
    assert outcome.failed_rows == 0
    assert outcome.pass_rate == 1.0


def test_missing_asin_rejected(sample_row_factory):
    rows = [sample_row_factory(asin=None, source_id="R-no-asin")]
    outcome = run_dq(rows)
    assert outcome.passed_rows == 0
    assert outcome.per_check_failed[CHECK_MISSING_ASIN] == 1


def test_invalid_rating_rejected(sample_row_factory):
    bad = [
        sample_row_factory(rating=0, source_id="R-bad-rating-0"),
        sample_row_factory(rating=6, source_id="R-bad-rating-6"),
        sample_row_factory(rating=-1, source_id="R-bad-rating-neg"),
    ]
    outcome = run_dq(bad)
    assert outcome.passed_rows == 0
    assert outcome.per_check_failed[CHECK_INVALID_RATING] == 3


def test_low_language_confidence_rejected(sample_row_factory):
    rows = [sample_row_factory(lang_confidence=0.42, source_id="R-low-conf")]
    outcome = run_dq(rows)
    assert outcome.passed_rows == 0
    assert outcome.per_check_failed[CHECK_LOW_LANG_CONFIDENCE] == 1


def test_non_english_rejected(sample_row_factory):
    rows = [
        sample_row_factory(language="de", source_id="R-de"),
        sample_row_factory(language="ja", source_id="R-ja"),
    ]
    outcome = run_dq(rows)
    assert outcome.passed_rows == 0
    assert outcome.per_check_failed[CHECK_NON_ENGLISH] == 2


def test_short_text_rejected(sample_row_factory):
    rows = [sample_row_factory(text_raw="too short", source_id="R-short")]
    outcome = run_dq(rows)
    assert outcome.passed_rows == 0
    assert outcome.per_check_failed[CHECK_TEXT_TOO_SHORT] == 1


def test_spam_rejected(sample_row_factory):
    rows = [
        sample_row_factory(
            text_raw="https://buycheap.example.com https://promo.example.com",
            source_id="R-urls",
        ),
        sample_row_factory(
            text_raw="BUY NOW DISCOUNT CODE PROMO LIMITED TIME OFFER",
            source_id="R-allcaps",
        ),
    ]
    outcome = run_dq(rows)
    assert outcome.passed_rows == 0
    assert outcome.per_check_failed[CHECK_EMPTY_OR_SPAM] == 2


def test_duplicate_rejected(sample_row_factory):
    base = sample_row_factory(source_id="R-dup-001")
    rows = [base, dict(base)]
    outcome = run_dq(rows)
    assert outcome.passed_rows == 1
    assert outcome.per_check_failed[CHECK_DUPLICATE] == 1


def test_mixed_batch_counts(sample_row_factory):
    rows = [
        sample_row_factory(source_id="R-1"),
        sample_row_factory(source_id="R-2", rating=0),
        sample_row_factory(source_id="R-3", language="de"),
        sample_row_factory(source_id="R-4", text_raw="hi"),
        sample_row_factory(source_id="R-5"),
    ]
    outcome = run_dq(rows)
    assert outcome.total_rows == 5
    assert outcome.passed_rows == 2
    assert outcome.failed_rows == 3
    assert round(outcome.pass_rate, 2) == 0.40
