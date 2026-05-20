"""DB-backed tests: clustering input builder, cluster_flow, cluster_stats."""
from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from scripts.cluster_stats import collect_cluster_stats
from scripts.seed_aspect_ontology import seed_aspect_ontology
from voicelens.analytics.clustering import build_clustering_records
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Cluster,
    IngestRun,
    Review,
    ReviewCluster,
    Sku,
)
from voicelens.pipeline.flows.cluster_flow import cluster_flow

_ASPECT_VERSION = "v2"
_PROVIDER = "anthropic"
_MODEL = "claude-opus-4.6"


def _seed_review(
    session,
    *,
    source_id: str,
    text: str,
    status: str,
    mentions: list[tuple[str, str, str | None, str]],
    brand: str = "Anker",
    asin: str = "B0X",
    rating: int = 1,
) -> int:
    """Insert one review + its ABSA status + aspect mentions; return review_id."""
    run = session.scalar(select(IngestRun).where(IngestRun.source == "test_src"))
    if run is None:
        run = IngestRun(source="test_src", status="completed")
        session.add(run)
        session.flush()
    brand_row = session.scalar(select(Brand).where(Brand.name == brand))
    if brand_row is None:
        brand_row = Brand(name=brand)
        session.add(brand_row)
        session.flush()
    sku = session.scalar(select(Sku).where(Sku.brand_id == brand_row.id, Sku.asin == asin))
    if sku is None:
        sku = Sku(brand_id=brand_row.id, asin=asin)
        session.add(sku)
        session.flush()
    review = Review(
        source="amazon_reviews_2023",
        source_id=source_id,
        sku_id=sku.id,
        language="en",
        lang_confidence=0.95,
        rating=rating,
        posted_at=datetime(2024, 1, 1),
        text_raw=text,
        char_len=len(text),
        ingest_run_id=run.id,
    )
    session.add(review)
    session.flush()

    session.add(
        ABSAReviewStatus(
            review_id=review.id,
            aspect_version=_ASPECT_VERSION,
            provider=_PROVIDER,
            model_name=_MODEL,
            status=status,
            n_mentions=len(mentions),
            n_errors=0,
        )
    )
    aspect_by_code = {
        row.code: row.id
        for row in session.execute(
            select(AspectOntology).where(AspectOntology.version == _ASPECT_VERSION)
        ).scalars()
    }
    for code, sentiment, severity, quote in mentions:
        session.add(
            AspectMention(
                review_id=review.id,
                aspect_id=aspect_by_code[code],
                sentiment=sentiment,
                severity=severity,
                evidence_quote=quote,
                model_name=_MODEL,
                aspect_version=_ASPECT_VERSION,
            )
        )
    session.flush()
    return review.id


@pytest.fixture
def clustering_corpus(fresh_engine):
    """Seed a corpus: reliability(8 neg) + charging(4 neg) + battery(1 neg) +
    a positive mention and a failed-status review that must be excluded."""
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s, version=_ASPECT_VERSION)
        for i in range(8):
            _seed_review(
                s,
                source_id=f"rel-{i}",
                text=f"The unit stopped working completely after a week number {i}.",
                status="success",
                mentions=[(
                    "reliability", "negative", "high",
                    f"stopped working completely after a week {i}",
                )],
                asin=f"B0R{i}",
            )
        for i in range(4):
            _seed_review(
                s,
                source_id=f"chg-{i}",
                text=f"Charging port failed and would not charge at all {i}.",
                status="success",
                mentions=[(
                    "charging", "negative", "medium",
                    f"charging port failed would not charge {i}",
                )],
                asin=f"B0C{i}",
            )
        # battery: a single negative mention -> below min_cluster_size.
        _seed_review(
            s,
            source_id="bat-0",
            text="Battery drains overnight even when switched off.",
            status="success",
            mentions=[("battery", "negative", "low", "battery drains overnight")],
            asin="B0B0",
        )
        # positive mention -> must be excluded by the input builder.
        _seed_review(
            s,
            source_id="pos-0",
            text="Great price and excellent value for money.",
            status="success",
            mentions=[("price", "positive", None, "great price excellent value")],
            asin="B0P0",
        )
        # failed-status review with a negative mention -> must be excluded.
        _seed_review(
            s,
            source_id="failed-0",
            text="Device stopped working but the LLM gateway timed out.",
            status="failed",
            mentions=[("reliability", "negative", "high", "device stopped working")],
            asin="B0F0",
        )
        s.commit()
    yield fresh_engine


# ---- input builder ------------------------------------------------------


def test_build_clustering_records_selects_only_negative_success(clustering_corpus):
    SessionLocal = sessionmaker(bind=clustering_corpus, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        records = build_clustering_records(
            s, aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model_name=_MODEL
        )
    # 8 reliability + 4 charging + 1 battery negatives = 13; positive and
    # the failed-status review are excluded.
    assert len(records) == 13
    assert {r.aspect_code for r in records} == {"reliability", "charging", "battery"}
    assert all(r.evidence_quote for r in records)
    # the positive "price" mention never made it in
    assert "price" not in {r.aspect_code for r in records}


# ---- cluster_flow -------------------------------------------------------


def test_cluster_flow_writes_cluster_and_review_cluster_rows(clustering_corpus):
    summary = cluster_flow(
        aspect_version=_ASPECT_VERSION,
        provider=_PROVIDER,
        model=_MODEL,
        min_cluster_size=3,
    )
    assert summary["negative_mentions"] == 13
    assert summary["clusters"] >= 1
    # battery has 1 mention -> below the floor -> skipped.
    assert "battery" in summary["skipped_aspects"]
    assert summary["clusters_by_aspect"]  # reliability and/or charging

    SessionLocal = sessionmaker(bind=clustering_corpus, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_clusters = s.scalar(select(func.count()).select_from(Cluster))
        n_members = s.scalar(select(func.count()).select_from(ReviewCluster))
        assert n_clusters == summary["clusters"]
        assert n_members == summary["clustered_mentions"]
        clusters = list(s.execute(select(Cluster)).scalars())
        for c in clusters:
            assert c.size >= 3
            assert c.label
            assert c.aspect_code in {"reliability", "charging"}


def test_cluster_flow_rerun_is_idempotent(clustering_corpus):
    first = cluster_flow(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        min_cluster_size=3,
    )
    second = cluster_flow(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        min_cluster_size=3,
    )
    assert second["removed_prior_clusters"] == first["clusters"]
    SessionLocal = sessionmaker(bind=clustering_corpus, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        # Only the second run's rows survive — no accumulation.
        assert s.scalar(select(func.count()).select_from(Cluster)) == second["clusters"]


def test_cluster_flow_no_negative_mentions_returns_empty(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s, version=_ASPECT_VERSION)
        s.commit()
    summary = cluster_flow(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        min_cluster_size=3,
    )
    assert summary["negative_mentions"] == 0
    assert summary["clusters"] == 0


# ---- cluster_stats ------------------------------------------------------


def test_collect_cluster_stats_after_flow(clustering_corpus):
    cluster_flow(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        min_cluster_size=3,
    )
    stats = collect_cluster_stats(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model_name=_MODEL
    )
    assert stats["total_clusters"] >= 1
    assert stats["total_clustered_mentions"] >= 1
    assert stats["clusters_by_aspect"]
    assert len(stats["top_by_size"]) >= 1
    top = stats["top_by_size"][0]
    assert {"aspect_code", "label", "size", "severity_weighted_size"} <= set(top)


def test_collect_cluster_stats_empty_when_no_clusters(fresh_engine):
    stats = collect_cluster_stats(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model_name=_MODEL
    )
    assert stats["total_clusters"] == 0
    assert stats["clusters_by_aspect"] == {}
    assert stats["top_by_size"] == []
