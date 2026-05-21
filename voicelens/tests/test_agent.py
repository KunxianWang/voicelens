"""Tests for the M6B LangGraph agent workflow (voicelens.agent).

Router tests are pure. Route-tool tests use the seeded dashboard
corpus (Postgres via SQLite) and an injected fake retriever — no live
Qdrant and no real LLM.
"""
from __future__ import annotations

import pytest
from sqlalchemy.orm import sessionmaker

from scripts.ask_agent import main as ask_agent_main
from voicelens.agent import (
    ROUTE_ANALYTICS,
    ROUTE_INCIDENT,
    ROUTE_INSUFFICIENT,
    ROUTE_RETRIEVAL,
    classify_route,
    extract_filters,
    result_summary,
    run_agent,
)
from voicelens.agent.prompts import INSUFFICIENT_SCOPE_MESSAGE
from voicelens.retrieval.search import SearchHit
from voicelens.tests.test_ui_db import _seed_dashboard_corpus

# ---- router -------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What are the main reliability complaints?", ROUTE_RETRIEVAL),
        ("What are customers saying about bluetooth?", ROUTE_RETRIEVAL),
        ("Show me example reviews about charging", ROUTE_RETRIEVAL),
        ("Tell me about battery problems", ROUTE_RETRIEVAL),  # aspect fallback
        ("Which issues spiked recently?", ROUTE_INCIDENT),
        ("Are there any emerging anomalies?", ROUTE_INCIDENT),
        ("Which aspect has the most negative mentions?", ROUTE_ANALYTICS),
        ("What is the sentiment distribution?", ROUTE_ANALYTICS),
        ("How many reviews are there?", ROUTE_ANALYTICS),
        ("What is the capital of France?", ROUTE_INSUFFICIENT),
        ("", ROUTE_INSUFFICIENT),
    ],
)
def test_classify_route(question, expected):
    assert classify_route(question) == expected


def test_extract_filters_detects_aspect_and_sentiment():
    filters = extract_filters("What are the negative reliability complaints?")
    assert filters["aspect"] == "reliability"
    assert filters["sentiment"] == "negative"


def test_extract_filters_detects_brand_and_positive():
    filters = extract_filters("What do customers praise about Anker bluetooth?")
    assert filters["aspect"] == "bluetooth"
    assert filters["sentiment"] == "positive"
    assert filters["brand"] == "Anker"


def test_extract_filters_empty_when_no_hints():
    assert extract_filters("show me some reviews") == {}


# ---- retrieval route ----------------------------------------------------


def _fake_hits() -> list[SearchHit]:
    return [
        SearchHit(
            score=0.9, review_id=11, brand="Anker", asin="A1", rating=1,
            aspect_codes=["reliability"], sentiments=["negative"],
            evidence_quotes=["stopped working after a week"],
            text_raw="The unit stopped working after a week.",
            raw_payload={"source_id": "S11", "review_id": 11},
        ),
        SearchHit(
            score=0.7, review_id=12, brand="Anker", asin="A2", rating=1,
            aspect_codes=["reliability"], sentiments=["negative"],
            evidence_quotes=["broke on the second use"],
            text_raw="It broke on the second use.",
            raw_payload={"source_id": "S12", "review_id": 12},
        ),
    ]


def test_retrieval_route_returns_citations():
    def fake_retrieve(question, filters, top_k):  # noqa: ARG001
        return _fake_hits()

    state = run_agent(
        "What are the main reliability complaints?",
        provider="mock", retrieve_fn=fake_retrieve,
    )
    summary = result_summary(state)
    assert summary["route"] == ROUTE_RETRIEVAL
    assert summary["filters"]["aspect"] == "reliability"
    assert summary["citations"]
    assert summary["retrieved_review_ids"] == [11, 12]
    assert summary["answer"]


def test_retrieval_route_handles_unavailable_retrieval():
    def broken_retrieve(question, filters, top_k):  # noqa: ARG001
        raise RuntimeError("Qdrant collection 'reviews_v2' does not exist.")

    state = run_agent(
        "reliability complaints", provider="mock", retrieve_fn=broken_retrieve
    )
    summary = result_summary(state)
    assert summary["route"] == ROUTE_RETRIEVAL
    assert "unavailable" in summary["answer"].lower()
    assert summary["warnings"]
    assert summary["citations"] == []


# ---- DB-backed routes ---------------------------------------------------


@pytest.fixture
def agent_db(fresh_engine):
    """A test DB seeded with the dashboard corpus (incidents + mentions)."""
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        _seed_dashboard_corpus(s)
    yield fresh_engine


def test_incident_route_with_incidents(agent_db):
    state = run_agent("Which issues spiked recently?", provider="mock")
    summary = result_summary(state)
    assert summary["route"] == ROUTE_INCIDENT
    assert summary["incidents_result"]          # 2 incidents seeded
    assert "severity" in summary["answer"].lower()
    assert not summary["warnings"]


def test_incident_route_with_empty_table(fresh_engine):  # noqa: ARG001
    state = run_agent("Are there emerging anomalies?", provider="mock")
    summary = result_summary(state)
    assert summary["route"] == ROUTE_INCIDENT
    assert summary["incidents_result"] == []
    assert "no emerging" in summary["answer"].lower()
    assert summary["warnings"]                  # flags the empty table


def test_analytics_route_aspect_stats(agent_db):
    state = run_agent("Which aspect has the most negative mentions?", provider="mock")
    summary = result_summary(state)
    assert summary["route"] == ROUTE_ANALYTICS
    result = summary["analytics_result"]
    assert result["kind"] == "aspect_distribution"
    assert result["aspect_distribution"]
    assert any(ch.isdigit() for ch in summary["answer"])


def test_analytics_route_sentiment_stats(agent_db):
    state = run_agent("What is the sentiment distribution?", provider="mock")
    summary = result_summary(state)
    assert summary["route"] == ROUTE_ANALYTICS
    assert summary["analytics_result"]["kind"] == "sentiment_distribution"
    assert "negative" in summary["answer"].lower()


# ---- insufficient-scope route ------------------------------------------


def test_insufficient_route_returns_safe_message():
    state = run_agent("What is the capital of France?", provider="mock")
    summary = result_summary(state)
    assert summary["route"] == ROUTE_INSUFFICIENT
    assert summary["answer"] == INSUFFICIENT_SCOPE_MESSAGE
    assert summary["citations"] == []


# ---- ask_agent CLI ------------------------------------------------------


def test_ask_agent_script_with_mock_provider(agent_db, capsys):
    """The CLI runs end-to-end on a non-retrieval route with mock provider."""
    exit_code = ask_agent_main(
        ["--question", "Which aspect has the most negative mentions?",
         "--provider", "mock", "--json"]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"route": "analytics_summary"' in out
