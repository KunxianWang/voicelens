"""Tests for the ask_voicelens CLI helpers (retrieve + answer, mock provider).

Uses an in-memory Qdrant + mock embeddings, so no Qdrant server, no
embedding-model download, and no LLM are involved.
"""
from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from scripts.ask_voicelens import retrieve_hits
from voicelens.rag.answer import generate_answer
from voicelens.retrieval.embeddings import get_embedding_provider
from voicelens.retrieval.qdrant_index import (
    build_embedding_text,
    build_payload,
    build_point_id,
    ensure_collection,
    upsert_points,
)

_COLLECTION = "ask_voicelens_fixture"

_CORPUS = [
    {
        "review_id": 101, "brand": "Anker", "asin": "A1", "rating": 1,
        "text_raw": "The charger STOPPED working completely after a week.",
        "mentions": [{"aspect_code": "reliability", "sentiment": "negative",
                      "severity": "high", "evidence_quote": "STOPPED working"}],
    },
    {
        "review_id": 102, "brand": "Anker", "asin": "A2", "rating": 1,
        "text_raw": "Dead on arrival; it STOPPED working out of the box.",
        "mentions": [{"aspect_code": "reliability", "sentiment": "negative",
                      "severity": "high", "evidence_quote": "STOPPED working"}],
    },
    {
        "review_id": 103, "brand": "Bose", "asin": "B1", "rating": 5,
        "text_raw": "Bluetooth pairs INSTANTLY and never drops.",
        "mentions": [{"aspect_code": "bluetooth", "sentiment": "positive",
                      "severity": None, "evidence_quote": "pairs INSTANTLY"}],
    },
]


@pytest.fixture
def indexed_client() -> QdrantClient:
    """An in-memory Qdrant collection indexed with mock embeddings."""
    client = QdrantClient(":memory:")
    embedder = get_embedding_provider("mock")
    points: list[rest.PointStruct] = []
    for review in _CORPUS:
        emb_text = build_embedding_text(review["text_raw"], review["mentions"])
        vector = embedder.embed_batch([emb_text])[0]
        payload = build_payload(
            review={**review, "source": "test", "source_id": f"S{review['review_id']}",
                    "sku_id": 1, "verified": True, "posted_at": "2024-01-01"},
            mentions=review["mentions"],
            aspect_version="v2", provider="anthropic", model_name="claude-opus-4.6",
            absa_status="success",
        )
        point_id = build_point_id(review["review_id"], "v2", "anthropic",
                                  "claude-opus-4.6")
        points.append(rest.PointStruct(id=point_id, vector=vector, payload=payload))
    ensure_collection(client, _COLLECTION, vector_size=embedder.dimension)
    upsert_points(client, _COLLECTION, points)
    return client


def test_retrieve_hits_returns_ranked_reviews(indexed_client):
    hits = retrieve_hits(
        "products that STOPPED working", mode="hybrid", collection=_COLLECTION,
        embedding_provider="mock", client=indexed_client, top_k=5,
    )
    assert hits
    assert all(h.review_id is not None for h in hits)
    # the STOPPED-working reliability reviews should be retrieved
    assert {101, 102} & {h.review_id for h in hits}


def test_ask_voicelens_pipeline_with_mock_provider(indexed_client):
    """retrieve_hits -> generate_answer is the whole script path."""
    question = "What are customers saying about products that stopped working?"
    hits = retrieve_hits(
        question, mode="hybrid", collection=_COLLECTION,
        embedding_provider="mock", client=indexed_client, top_k=5,
    )
    result = generate_answer(question, hits, provider="mock")

    assert result.citations
    retrieved_ids = {h.review_id for h in hits}
    assert all(c.review_id in retrieved_ids for c in result.citations)
    assert result.guardrail_flags["unsupported_citations"] == []
    assert result.provider == "mock"


def test_retrieve_hits_rejects_bad_mode(indexed_client):
    with pytest.raises(ValueError, match="mode must be one of"):
        retrieve_hits(
            "anything", mode="bogus", collection=_COLLECTION,
            embedding_provider="mock", client=indexed_client,
        )


def test_retrieve_hits_missing_collection_raises(indexed_client):
    with pytest.raises(RuntimeError, match="does not exist"):
        retrieve_hits(
            "anything", mode="hybrid", collection="no_such_collection",
            embedding_provider="mock", client=indexed_client,
        )
