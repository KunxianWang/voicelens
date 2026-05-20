"""Tests for the manual-review candidate exporter."""
from __future__ import annotations

from scripts.export_retrieval_review_candidates import (
    CANDIDATE_FIELDS,
    build_candidate_rows,
)
from voicelens.retrieval.search import SearchHit


def _hit(review_id: int, score: float) -> SearchHit:
    return SearchHit(
        score=score,
        review_id=review_id,
        brand="Anker",
        asin="A1",
        rating=1,
        aspect_codes=["reliability"],
        sentiments=["negative"],
        evidence_quotes=["STOPPED working"],
        text_raw="Battery died\nand STOPPED working after a week.",
    )


def test_build_candidate_rows_has_expected_columns():
    rows = build_candidate_rows(
        query_id="rel-stopped",
        query="stopped working",
        mode="dense",
        hits=[_hit(1, 0.9), _hit(2, 0.8)],
        gold_ids=[1],
        top_n=15,
    )
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == set(CANDIDATE_FIELDS)


def test_build_candidate_rows_marks_currently_gold():
    rows = build_candidate_rows(
        query_id="q", query="x", mode="lexical",
        hits=[_hit(7, 1.0), _hit(8, 0.5)],
        gold_ids=[7],
        top_n=15,
    )
    assert rows[0]["currently_gold"] is True
    assert rows[1]["currently_gold"] is False


def test_build_candidate_rows_ranks_and_trims_to_top_n():
    rows = build_candidate_rows(
        query_id="q", query="x", mode="hybrid",
        hits=[_hit(i, 1.0 - i / 10) for i in range(1, 11)],
        gold_ids=[],
        top_n=3,
    )
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert len(rows) == 3


def test_build_candidate_rows_snippet_is_single_line():
    rows = build_candidate_rows(
        query_id="q", query="x", mode="dense",
        hits=[_hit(1, 0.9)], gold_ids=[], top_n=15,
    )
    assert "\n" not in rows[0]["text_snippet"]
    assert rows[0]["suggested_action"] == ""
