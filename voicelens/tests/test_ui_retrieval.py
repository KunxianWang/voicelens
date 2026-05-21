"""Tests for the dashboard retrieval search helper (run_search).

Uses an in-memory Qdrant client + deterministic mock embeddings, so no
real Qdrant server, no embedding model download, and no LLM are needed.
"""
from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from voicelens.retrieval.embeddings import MockEmbeddingProvider
from voicelens.retrieval.lexical import BM25LexicalRetriever
from voicelens.retrieval.qdrant_index import (
    build_embedding_text,
    build_payload,
    build_point_id,
    ensure_collection,
    upsert_points,
)
from voicelens.ui.pages.retrieval import RetrievalBackends, run_search

_DIM = 32

_CORPUS = [
    {
        "review_id": 1, "brand": "Anker", "asin": "A1", "rating": 1,
        "text_raw": "Battery died after a week and STOPPED working completely.",
        "mentions": [{"aspect_code": "reliability", "sentiment": "negative",
                      "severity": "high", "evidence_quote": "STOPPED working"}],
    },
    {
        "review_id": 2, "brand": "Anker", "asin": "A2", "rating": 1,
        "text_raw": "Device is dead on arrival; STOPPED working out of the box.",
        "mentions": [{"aspect_code": "reliability", "sentiment": "negative",
                      "severity": "high", "evidence_quote": "STOPPED working"}],
    },
    {
        "review_id": 3, "brand": "Bose", "asin": "B1", "rating": 5,
        "text_raw": "Bluetooth pairs INSTANTLY and stays connected all day.",
        "mentions": [{"aspect_code": "bluetooth", "sentiment": "positive",
                      "severity": None, "evidence_quote": "pairs INSTANTLY"}],
    },
]


@pytest.fixture
def backends() -> RetrievalBackends:
    """An in-memory Qdrant + mock embedder + BM25 over the fake corpus."""
    client = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=_DIM)
    collection = "ui_retrieval_fixture"
    points: list[rest.PointStruct] = []
    payloads: list[dict] = []
    for review in _CORPUS:
        emb_text = build_embedding_text(review["text_raw"], review["mentions"])
        vector = embedder.embed_batch([emb_text])[0]
        payload = build_payload(
            review={**review, "source": "test", "source_id": f"S{review['review_id']}",
                    "sku_id": 1, "verified": True, "posted_at": "2024-01-01"},
            mentions=review["mentions"],
            aspect_version="v2",
            provider="anthropic",
            model_name="claude-opus-4.6",
            absa_status="success",
        )
        point_id = build_point_id(review["review_id"], "v2", "anthropic",
                                  "claude-opus-4.6")
        points.append(rest.PointStruct(id=point_id, vector=vector, payload=payload))
        payloads.append(payload)
    ensure_collection(client, collection, vector_size=_DIM)
    upsert_points(client, collection, points)
    bm25 = BM25LexicalRetriever(payloads)
    return RetrievalBackends(
        client=client, embedder=embedder, bm25=bm25, collection=collection
    )


def test_empty_query_returns_no_hits(backends):
    assert run_search("", backends=backends) == []
    assert run_search("   ", backends=backends, mode="lexical") == []


def test_invalid_mode_rejected(backends):
    with pytest.raises(ValueError, match="mode must be one of"):
        run_search("STOPPED working", backends=backends, mode="bogus")


def test_lexical_search_finds_reliability_docs(backends):
    hits = run_search("STOPPED working", backends=backends, mode="lexical", top_k=5)
    assert hits
    assert {h.review_id for h in hits} <= {1, 2, 3}
    # the STOPPED-working reviews (1, 2) must surface on this query
    assert {1, 2} & {h.review_id for h in hits}


def test_dense_search_returns_hits(backends):
    hits = run_search("STOPPED working", backends=backends, mode="dense", top_k=3)
    assert hits
    assert all(h.review_id is not None for h in hits)


def test_hybrid_search_is_the_default_mode(backends):
    hits = run_search("STOPPED working", backends=backends, top_k=3)
    assert hits  # default mode is hybrid / rrf_equal
    assert len(hits) <= 3


def test_hybrid_search_respects_brand_filter(backends):
    hits = run_search(
        "STOPPED working", backends=backends, mode="hybrid", brand="Anker", top_k=5
    )
    assert all(h.brand == "Anker" for h in hits)


def test_search_empty_result_is_graceful(backends):
    """A query with no lexical/dense overlap must return [] cleanly."""
    hits = run_search(
        "STOPPED working", backends=backends, mode="lexical", brand="NoSuchBrand"
    )
    assert hits == []
