"""End-to-end retrieval eval over a tiny in-memory corpus."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from scripts.evaluate_retrieval import evaluate_retrieval
from voicelens.retrieval.embeddings import MockEmbeddingProvider
from voicelens.retrieval.lexical import BM25LexicalRetriever
from voicelens.retrieval.qdrant_index import (
    build_embedding_text,
    build_payload,
    build_point_id,
    ensure_collection,
    upsert_points,
)


@pytest.fixture
def fake_corpus():
    """Six reviews covering reliability / bluetooth / charging / price.

    Mock embeddings are deterministic SHA-256-based; we control the
    "query <-> doc" similarity by sharing tokens between the query text
    and each doc's embedding_text. The token "STOPPED" is intentionally
    unique to reliability docs so the query "STOPPED working" puts them
    on top, etc.
    """
    return [
        {
            "review_id": 1, "brand": "Anker", "asin": "A1", "rating": 1,
            "text_raw": "Battery died after a week and STOPPED working.",
            "mentions": [{"aspect_code": "reliability", "sentiment": "negative",
                          "severity": "medium", "evidence_quote": "STOPPED working"}],
            "absa_status": "success",
        },
        {
            "review_id": 2, "brand": "Anker", "asin": "A2", "rating": 1,
            "text_raw": "Device is DEAD on arrival; STOPPED working out of the box.",
            "mentions": [{"aspect_code": "reliability", "sentiment": "negative",
                          "severity": "medium", "evidence_quote": "STOPPED working"}],
            "absa_status": "success",
        },
        {
            "review_id": 3, "brand": "Anker", "asin": "A3", "rating": 1,
            "text_raw": "Bluetooth DISCONNECTS every two minutes; pairing also fails.",
            "mentions": [{"aspect_code": "bluetooth", "sentiment": "negative",
                          "severity": "medium",
                          "evidence_quote": "Bluetooth DISCONNECTS every two minutes"}],
            "absa_status": "success",
        },
        {
            "review_id": 4, "brand": "Bose", "asin": "B1", "rating": 5,
            "text_raw": "Bluetooth pairs INSTANTLY and stays connected.",
            "mentions": [{"aspect_code": "bluetooth", "sentiment": "positive",
                          "severity": None, "evidence_quote": "pairs INSTANTLY"}],
            "absa_status": "success",
        },
        {
            "review_id": 5, "brand": "JBL", "asin": "J1", "rating": 2,
            "text_raw": "Charger STOPPED working after a week.",
            "mentions": [{"aspect_code": "charging", "sentiment": "negative",
                          "severity": "medium", "evidence_quote": "Charger STOPPED working"}],
            "absa_status": "success",
        },
        {
            "review_id": 6, "brand": "Anker", "asin": "A4", "rating": 4,
            "text_raw": "Great PRICE for the value.",
            "mentions": [{"aspect_code": "price", "sentiment": "positive",
                          "severity": None, "evidence_quote": "Great PRICE for the value"}],
            "absa_status": "success",
        },
    ]


@pytest.fixture
def indexed_client(fake_corpus):
    client = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=32)
    collection = "retrieval_eval_fixture"
    points: list[rest.PointStruct] = []
    payloads: list[dict] = []
    for review in fake_corpus:
        emb_text = build_embedding_text(review["text_raw"], review["mentions"])
        vector = embedder.embed_batch([emb_text])[0]
        payload = build_payload(
            review={**review, "source": "test", "source_id": f"S{review['review_id']}",
                    "sku_id": 1, "verified": True, "posted_at": "2024-01-01"},
            mentions=review["mentions"],
            aspect_version="v2",
            provider="anthropic",
            model_name="claude-opus-4.6",
            absa_status=review["absa_status"],
        )
        point_id = build_point_id(
            review["review_id"], "v2", "anthropic", "claude-opus-4.6"
        )
        points.append(rest.PointStruct(id=point_id, vector=vector, payload=payload))
        payloads.append(payload)
    ensure_collection(client, collection, vector_size=32)
    upsert_points(client, collection, points)
    return client, collection, embedder, payloads


def test_dense_eval_produces_per_query_and_summary(indexed_client):
    client, collection, embedder, _payloads = indexed_client
    goldens = [
        {
            "query_id": "rel-stopped",
            "query": "STOPPED working",
            "expected_aspect": "reliability",
            "expected_sentiment": "negative",
            "expected_brand": None,
            "gold_review_ids": [1, 2],
        },
        {
            "query_id": "bt-disconnect",
            "query": "Bluetooth DISCONNECTS",
            "expected_aspect": "bluetooth",
            "expected_sentiment": "negative",
            "expected_brand": None,
            "gold_review_ids": [3],
        },
    ]
    result = evaluate_retrieval(
        goldens=goldens,
        mode="dense",
        client=client,
        collection=collection,
        embedder=embedder,
        bm25=None,
        limit=5,
    )
    summary = result["summary"]
    assert summary["n_queries"] == 2
    assert summary["mode"] == "dense"
    assert summary["queries_with_gold"] == 2
    assert summary["queries_without_gold"] == 0
    assert summary["error_breakdown"]
    # Per-query rows always carry recall_at_5 / mrr_at_10.
    assert all("recall_at_5" in row for row in result["per_query"])
    assert all("mrr_at_10" in row for row in result["per_query"])


def test_lexical_eval_works_on_fake_corpus(indexed_client):
    client, collection, embedder, payloads = indexed_client
    bm25 = BM25LexicalRetriever(payloads)
    goldens = [
        {
            "query_id": "rel-stopped",
            "query": "STOPPED working week",
            "expected_aspect": "reliability",
            "expected_sentiment": "negative",
            "expected_brand": None,
            "gold_review_ids": [1, 2],
        },
    ]
    result = evaluate_retrieval(
        goldens=goldens, mode="lexical", client=client, collection=collection,
        embedder=embedder, bm25=bm25, limit=5,
    )
    # BM25 should fire on "STOPPED working" -> reviews 1 and 2.
    assert result["summary"]["hits_at_5"] >= 1
    assert result["per_query"][0]["recall_at_5"] > 0.0


def test_hybrid_eval_runs_and_emits_error_rows(indexed_client):
    client, collection, embedder, payloads = indexed_client
    bm25 = BM25LexicalRetriever(payloads)
    goldens = [
        {
            "query_id": "rel-stopped",
            "query": "STOPPED working",
            "expected_aspect": "reliability",
            "expected_sentiment": "negative",
            "expected_brand": None,
            "gold_review_ids": [1, 2],
        },
        {
            "query_id": "off-topic",
            "query": "unrelated banana pizza review",
            "expected_aspect": None,
            "expected_sentiment": None,
            "expected_brand": None,
            "gold_review_ids": [],
        },
    ]
    result = evaluate_retrieval(
        goldens=goldens, mode="hybrid", client=client, collection=collection,
        embedder=embedder, bm25=bm25, limit=5,
    )
    summary = result["summary"]
    assert summary["mode"] == "hybrid"
    assert summary["queries_with_gold"] == 1
    assert summary["queries_without_gold"] == 1
    assert {row["error_type"] for row in result["errors"]} <= {
        "good", "no_gold_hit", "filter_too_strict", "wrong_aspect",
        "wrong_sentiment", "lexical_miss",
    }


def test_lexical_first_preserves_lexical_top_result(indexed_client):
    """lexical_first must keep the lexical ranking's #1 hit at the top."""
    client, collection, embedder, payloads = indexed_client
    bm25 = BM25LexicalRetriever(payloads)
    goldens = [
        {
            "query_id": "rel-stopped", "query": "STOPPED working week",
            "expected_aspect": "reliability", "expected_sentiment": "negative",
            "expected_brand": None, "gold_review_ids": [1, 2],
        },
    ]
    lex = evaluate_retrieval(
        goldens=goldens, mode="lexical", client=client, collection=collection,
        embedder=embedder, bm25=bm25, limit=5,
    )
    lex_first = evaluate_retrieval(
        goldens=goldens, mode="lexical_first", client=client, collection=collection,
        embedder=embedder, bm25=bm25, limit=5,
    )
    lex_top = json.loads(lex["errors"][0]["retrieved_review_ids"])
    lf_top = json.loads(lex_first["errors"][0]["retrieved_review_ids"])
    # The lexical primary ranking leads the lexical_first fused order.
    assert lf_top[0] == lex_top[0]
    assert lex_first["summary"]["mode"] == "lexical_first"


def test_hybrid_weighted_runs_and_records_weights(indexed_client):
    client, collection, embedder, payloads = indexed_client
    bm25 = BM25LexicalRetriever(payloads)
    goldens = [
        {
            "query_id": "rel-stopped", "query": "STOPPED working",
            "expected_aspect": "reliability", "expected_sentiment": "negative",
            "expected_brand": None, "gold_review_ids": [1, 2],
        },
    ]
    result = evaluate_retrieval(
        goldens=goldens, mode="hybrid_weighted", client=client, collection=collection,
        embedder=embedder, bm25=bm25, limit=5,
        lexical_weight=0.8, dense_weight=0.2,
    )
    summary = result["summary"]
    assert summary["mode"] == "hybrid_weighted"
    assert summary["lexical_weight"] == 0.8
    assert summary["dense_weight"] == 0.2
    # New M3C metrics surface in the per-query rows + summary.
    assert "hit_at_5" in result["per_query"][0]
    assert "capped_recall_at_5" in result["per_query"][0]
    assert "r_precision" in result["per_query"][0]


def test_lexical_mode_requires_bm25(indexed_client):
    client, collection, embedder, _payloads = indexed_client
    with pytest.raises(ValueError, match="requires a BM25 retriever"):
        evaluate_retrieval(
            goldens=[], mode="lexical", client=client, collection=collection,
            embedder=embedder, bm25=None,
        )


def test_lexical_first_mode_requires_bm25(indexed_client):
    client, collection, embedder, _payloads = indexed_client
    with pytest.raises(ValueError, match="requires a BM25 retriever"):
        evaluate_retrieval(
            goldens=[], mode="lexical_first", client=client, collection=collection,
            embedder=embedder, bm25=None,
        )


def test_invalid_mode_rejected(indexed_client):
    client, collection, embedder, _payloads = indexed_client
    with pytest.raises(ValueError, match="mode must be one of"):
        evaluate_retrieval(
            goldens=[], mode="nonsense", client=client, collection=collection,
            embedder=embedder, bm25=None,
        )


def test_evaluator_writes_csv_and_summary(tmp_path: Path, indexed_client):
    """Smoke the CSV + JSON writer paths via the runner-equivalent flow."""
    from scripts.evaluate_retrieval import _write_csv

    client, collection, embedder, _payloads = indexed_client
    goldens = [
        {
            "query_id": "rel-stopped", "query": "STOPPED working",
            "expected_aspect": "reliability", "expected_sentiment": "negative",
            "expected_brand": None, "gold_review_ids": [1, 2],
        }
    ]
    result = evaluate_retrieval(
        goldens=goldens, mode="dense", client=client, collection=collection,
        embedder=embedder, bm25=None, limit=5,
    )
    results_path = tmp_path / "per_query.csv"
    errors_path = tmp_path / "errors.csv"
    summary_path = tmp_path / "summary.json"

    _write_csv(result["per_query"], results_path)
    _write_csv(result["errors"], errors_path)
    summary_path.write_text(json.dumps(result["summary"]), encoding="utf-8")

    with open(results_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "recall_at_5" in rows[0]

    with open(errors_path, encoding="utf-8") as f:
        err_rows = list(csv.DictReader(f))
    assert err_rows
    assert "error_type" in err_rows[0]
