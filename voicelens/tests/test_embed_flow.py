from __future__ import annotations

from datetime import datetime

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from scripts.seed_aspect_ontology import seed_aspect_ontology
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    IngestRun,
    Review,
    Sku,
)
from voicelens.pipeline.flows.embed_flow import embed_flow
from voicelens.retrieval.embeddings import MockEmbeddingProvider
from voicelens.retrieval.filters import build_search_filter
from voicelens.retrieval.qdrant_index import build_point_id
from voicelens.retrieval.search import retrieve

_FIXED_POSTED_AT = datetime(2024, 1, 1, 12, 0, 0)

REVIEWS = (
    # source_id,           brand,       asin,         rating, text,                                                            status,           aspects
    ("ANK-success-1",      "Anker",     "B0ANK10001", 1,
     "The battery died after a week and the device stopped working.",
     "success",
     [("battery", "negative", "medium", "The battery died after a week")]),
    ("ANK-no-mentions",    "Anker",     "B0ANK10002", 4,
     "Bought this last weekend; nothing special to report.",
     "no_mentions",
     []),
    ("BOSE-failed",        "Bose",      "B0BOS30001", 2,
     "Sound quality is bad but the LLM gateway timed out.",
     "failed",
     []),
    ("JBL-invalid",        "JBL",       "B0JBL40001", 3,
     "Charger arrived broken; provider returned non-verbatim quote.",
     "invalid",
     []),
    ("UGREEN-success-2",   "UGREEN",    "B0UGR50001", 5,
     "USB-C charging works perfectly and the price is great.",
     "success",
     [
         ("charging", "positive", None, "USB-C charging works perfectly"),
         ("price", "positive", None, "the price is great"),
     ]),
)


def _seed_corpus(session, *, aspect_version: str, provider: str, model_name: str) -> dict[str, int]:
    """Insert reviews + their ABSA status rows + aspect mentions; return source_id -> review_id."""
    ingest = IngestRun(source="amazon_reviews_2023_mvp_subset", status="completed")
    session.add(ingest)
    session.flush()

    ids: dict[str, int] = {}
    aspect_by_code = {
        row.code: row.id
        for row in session.execute(
            select(AspectOntology).where(AspectOntology.version == aspect_version)
        ).scalars()
    }

    for source_id, brand_name, asin, rating, text, status, aspects in REVIEWS:
        brand = session.scalar(select(Brand).where(Brand.name == brand_name))
        if brand is None:
            brand = Brand(name=brand_name)
            session.add(brand)
            session.flush()
        sku = session.scalar(select(Sku).where(Sku.brand_id == brand.id, Sku.asin == asin))
        if sku is None:
            sku = Sku(brand_id=brand.id, asin=asin)
            session.add(sku)
            session.flush()
        review = Review(
            source="amazon_reviews_2023",
            source_id=source_id,
            sku_id=sku.id,
            language="en",
            lang_confidence=0.95,
            rating=rating,
            posted_at=_FIXED_POSTED_AT,
            text_raw=text,
            char_len=len(text),
            ingest_run_id=ingest.id,
        )
        session.add(review)
        session.flush()
        ids[source_id] = review.id

        session.add(
            ABSAReviewStatus(
                review_id=review.id,
                aspect_version=aspect_version,
                provider=provider,
                model_name=model_name,
                status=status,
                n_mentions=len(aspects),
                n_errors=0,
                error_codes_json=None,
            )
        )
        for code, sentiment, severity, quote in aspects:
            session.add(
                AspectMention(
                    review_id=review.id,
                    aspect_id=aspect_by_code[code],
                    sentiment=sentiment,
                    severity=severity,
                    evidence_quote=quote,
                    model_name=model_name,
                    aspect_version=aspect_version,
                )
            )
    session.flush()
    return ids


@pytest.fixture
def absa_processed_corpus(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s, version="v2")
        ids = _seed_corpus(
            s, aspect_version="v2", provider="anthropic", model_name="claude-opus-4.6"
        )
        s.commit()
    yield fresh_engine, ids


def test_embed_flow_excludes_failed_and_invalid_by_default(absa_processed_corpus):
    engine, ids = absa_processed_corpus
    qdrant = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=16)

    summary = embed_flow(
        aspect_version="v2",
        provider="anthropic",
        model="claude-opus-4.6",
        collection="reviews_v2_test",
        client=qdrant,
        embedder=embedder,
    )

    # 3 success/no_mentions, 2 skipped (failed + invalid).
    assert summary["selected_reviews"] == 3
    assert summary["embedded_reviews"] == 3
    assert summary["upserted_points"] == 3
    assert summary["skipped_failed_or_invalid"] == 2
    assert summary["vector_size"] == 16
    assert summary["embedding_provider"] == "mock"

    point_count = qdrant.count(collection_name="reviews_v2_test", exact=True).count
    assert point_count == 3


def test_embed_flow_includes_explicit_statuses(absa_processed_corpus):
    """Caller can opt back into failed/invalid for forensic re-indexing."""
    engine, ids = absa_processed_corpus
    qdrant = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=16)

    summary = embed_flow(
        aspect_version="v2",
        provider="anthropic",
        model="claude-opus-4.6",
        collection="reviews_v2_all",
        client=qdrant,
        embedder=embedder,
        statuses=("success", "no_mentions", "failed", "invalid"),
    )
    assert summary["selected_reviews"] == 5
    assert summary["upserted_points"] == 5
    assert summary["skipped_failed_or_invalid"] == 0


def test_embed_flow_is_idempotent_on_rerun(absa_processed_corpus):
    """Same point_id key → re-running upserts in place rather than duplicating."""
    engine, _ids = absa_processed_corpus
    qdrant = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=16)

    embed_flow(
        aspect_version="v2", provider="anthropic", model="claude-opus-4.6",
        collection="reviews_v2_idem", client=qdrant, embedder=embedder,
    )
    embed_flow(
        aspect_version="v2", provider="anthropic", model="claude-opus-4.6",
        collection="reviews_v2_idem", client=qdrant, embedder=embedder,
    )
    point_count = qdrant.count(collection_name="reviews_v2_idem", exact=True).count
    assert point_count == 3


def test_embed_flow_payload_round_trips_aspect_codes(absa_processed_corpus):
    engine, ids = absa_processed_corpus
    qdrant = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=16)

    embed_flow(
        aspect_version="v2", provider="anthropic", model="claude-opus-4.6",
        collection="reviews_v2_payload", client=qdrant, embedder=embedder,
    )

    pid = build_point_id(
        ids["UGREEN-success-2"], "v2", "anthropic", "claude-opus-4.6"
    )
    [point] = qdrant.retrieve(collection_name="reviews_v2_payload", ids=[pid])
    payload = dict(point.payload or {})
    assert payload["brand"] == "UGREEN"
    assert payload["aspect_codes"] == ["charging", "price"]
    assert payload["sentiments"] == ["positive", "positive"]
    assert payload["mention_count"] == 2
    assert payload["absa_status"] == "success"


def test_embed_flow_no_matches_returns_empty_summary(absa_processed_corpus):
    """Filtering on an unused (provider, model) should produce a clean no-op."""
    engine, _ids = absa_processed_corpus
    qdrant = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=16)

    summary = embed_flow(
        aspect_version="v2", provider="anthropic", model="never-trained",
        collection="reviews_v2_empty", client=qdrant, embedder=embedder,
    )
    assert summary["selected_reviews"] == 0
    assert summary["upserted_points"] == 0
    assert summary["skipped_failed_or_invalid"] == 0
    assert not qdrant.collection_exists("reviews_v2_empty")


def test_retrieve_with_filter_only_returns_matching_brand(absa_processed_corpus):
    engine, _ids = absa_processed_corpus
    qdrant = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=32)

    embed_flow(
        aspect_version="v2", provider="anthropic", model="claude-opus-4.6",
        collection="reviews_v2_search", client=qdrant, embedder=embedder,
    )

    hits = retrieve(
        client=qdrant,
        collection="reviews_v2_search",
        embedder=embedder,
        query="anything",
        limit=5,
        filter_=build_search_filter(brand="UGREEN"),
    )
    assert hits, "filter should still return at least one hit"
    assert all(h.brand == "UGREEN" for h in hits)


def test_retrieve_with_aspect_filter_uses_aspect_codes_field(absa_processed_corpus):
    engine, _ids = absa_processed_corpus
    qdrant = QdrantClient(":memory:")
    embedder = MockEmbeddingProvider(dimension=32)

    embed_flow(
        aspect_version="v2", provider="anthropic", model="claude-opus-4.6",
        collection="reviews_v2_aspect", client=qdrant, embedder=embedder,
    )

    hits = retrieve(
        client=qdrant,
        collection="reviews_v2_aspect",
        embedder=embedder,
        query="anything",
        limit=5,
        filter_=build_search_filter(aspect="charging"),
    )
    assert hits
    assert all("charging" in h.aspect_codes for h in hits)
