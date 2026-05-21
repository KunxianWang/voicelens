"""DB-backed tests: mention-event builder, anomaly_flow, anomaly_stats."""
from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from scripts.anomaly_stats import collect_anomaly_stats
from scripts.seed_aspect_ontology import seed_aspect_ontology
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Cluster,
    Incident,
    IngestRun,
    Review,
    ReviewCluster,
    Sku,
)
from voicelens.pipeline.flows.anomaly_flow import (
    anomaly_flow,
    build_mention_events,
    upsert_incident,
)

_ASPECT_VERSION = "v2"
_PROVIDER = "anthropic"
_MODEL = "claude-opus-4.6"
_BASE_MONDAY = date(2024, 1, 1)  # a Monday


def _week(i: int) -> datetime:
    """Midnight Monday of week ``i`` (0-based) from the fixed base."""
    return datetime.combine(
        date.fromordinal(_BASE_MONDAY.toordinal() + 7 * i), datetime.min.time()
    )


def _seed_clustered_review(
    session,
    *,
    source_id: str,
    posted_at: datetime,
    cluster_id: int,
    aspect_code: str = "charging",
    severity: str = "medium",
    rating: int = 1,
) -> int:
    """Insert a negative review + ABSA status + mention + cluster membership."""
    run = session.scalar(select(IngestRun).where(IngestRun.source == "test_src"))
    if run is None:
        run = IngestRun(source="test_src", status="completed")
        session.add(run)
        session.flush()
    brand = session.scalar(select(Brand).where(Brand.name == "Anker"))
    if brand is None:
        brand = Brand(name="Anker")
        session.add(brand)
        session.flush()
    sku = session.scalar(select(Sku).where(Sku.brand_id == brand.id, Sku.asin == "B0X"))
    if sku is None:
        sku = Sku(brand_id=brand.id, asin="B0X")
        session.add(sku)
        session.flush()

    text = f"Charging port failed and would not charge {source_id}."
    review = Review(
        source="amazon_reviews_2023",
        source_id=source_id,
        sku_id=sku.id,
        language="en",
        lang_confidence=0.95,
        rating=rating,
        posted_at=posted_at,
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
            status=ABSAReviewStatus.STATUS_SUCCESS,
            n_mentions=1,
            n_errors=0,
        )
    )
    aspect_id = session.scalar(
        select(AspectOntology.id).where(
            AspectOntology.version == _ASPECT_VERSION,
            AspectOntology.code == aspect_code,
        )
    )
    session.add(
        AspectMention(
            review_id=review.id,
            aspect_id=aspect_id,
            sentiment="negative",
            severity=severity,
            evidence_quote=f"charging port failed {source_id}",
            model_name=_MODEL,
            aspect_version=_ASPECT_VERSION,
        )
    )
    session.add(
        ReviewCluster(
            cluster_id=cluster_id,
            review_id=review.id,
            aspect_code=aspect_code,
            severity=severity,
        )
    )
    session.flush()
    return review.id


def _make_cluster(session, *, aspect_code: str = "charging", label: str | None = None) -> int:
    cluster = Cluster(
        run_id="seed-run",
        aspect_version=_ASPECT_VERSION,
        provider=_PROVIDER,
        model_name=_MODEL,
        aspect_code=aspect_code,
        label=label or "charging port failed would not charge",
        algorithm="tfidf_kmeans",
        size=0,
        severity_weighted_size=0.0,
    )
    session.add(cluster)
    session.flush()
    return cluster.id


@pytest.fixture
def anomaly_corpus(fresh_engine):
    """One charging cluster: five quiet weeks of 2 mentions, then a 12-week spike."""
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s, version=_ASPECT_VERSION)
        cluster_id = _make_cluster(s)
        rid = 0
        for week in range(5):  # quiet weeks 0-4: 2 mentions each
            for _ in range(2):
                rid += 1
                _seed_clustered_review(
                    s, source_id=f"q{rid}", posted_at=_week(week),
                    cluster_id=cluster_id,
                )
        for _ in range(12):  # week 5: a clear spike
            rid += 1
            _seed_clustered_review(
                s, source_id=f"s{rid}", posted_at=_week(5), cluster_id=cluster_id,
            )
        s.commit()
    yield fresh_engine, cluster_id


# ---- mention-event builder ----------------------------------------------


def test_build_mention_events_loads_clustered_negatives(anomaly_corpus):
    engine, cluster_id = anomaly_corpus
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        events, labels = build_mention_events(
            s, aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model_name=_MODEL
        )
    assert len(events) == 22  # 5*2 quiet + 12 spike
    assert {e.cluster_id for e in events} == {cluster_id}
    assert {e.aspect_code for e in events} == {"charging"}
    assert all(e.evidence_quote for e in events)
    assert labels[cluster_id]  # the cluster label was loaded


# ---- anomaly_flow -------------------------------------------------------


def test_anomaly_flow_writes_incident_rows(anomaly_corpus):
    engine, cluster_id = anomaly_corpus
    summary = anomaly_flow(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        granularity="cluster", min_history_weeks=3, z_threshold=2.0, min_volume=3,
    )
    assert summary["total_series"] == 1
    assert summary["total_weeks"] == 6
    assert summary["incidents_detected"] == 1
    assert summary["incidents_by_aspect"] == {"charging": 1}

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        incidents = list(s.execute(select(Incident)).scalars())
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.granularity == "cluster"
    assert inc.cluster_id == cluster_id
    assert inc.aspect_code == "charging"
    assert inc.week_start == date.fromordinal(_BASE_MONDAY.toordinal() + 7 * 5)
    assert inc.observed_volume == 12
    assert inc.z_score >= 2.0
    assert inc.baseline_volume < 5.0  # baseline reflects the quiet weeks, not the spike
    assert inc.summary  # a deterministic summary was written
    assert inc.status == Incident.STATUS_OPEN


def test_anomaly_flow_rerun_is_idempotent(anomaly_corpus):
    engine, _ = anomaly_corpus
    kwargs = dict(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        granularity="cluster", min_history_weeks=3, z_threshold=2.0, min_volume=3,
    )
    anomaly_flow(**kwargs)
    anomaly_flow(**kwargs)  # second run must not accumulate rows
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        assert s.scalar(select(func.count()).select_from(Incident)) == 1


def test_anomaly_flow_no_clusters_returns_empty(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s, version=_ASPECT_VERSION)
        s.commit()
    summary = anomaly_flow(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        granularity="cluster",
    )
    assert summary["total_series"] == 0
    assert summary["incidents_detected"] == 0


# ---- upsert idempotency -------------------------------------------------


def test_upsert_incident_idempotent_on_natural_key(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    key = dict(
        granularity="cluster", aspect_version=_ASPECT_VERSION, provider=_PROVIDER,
        model_name=_MODEL, cluster_id=None, aspect_code="charging",
        week_start=date(2024, 2, 5),
    )
    with SessionLocal() as s:
        first = upsert_incident(
            s, run_id="run-1", cluster_label=None, observed_volume=10,
            baseline_volume=2.0, z_score=3.0, severity_score=20.0,
            unique_review_count=10, avg_rating=1.2, example_quotes=["q"],
            summary="first", **key,
        )
        first_id = first.id
        # same natural key, new values -> updates the row, no insert.
        second = upsert_incident(
            s, run_id="run-2", cluster_label=None, observed_volume=15,
            baseline_volume=2.5, z_score=4.0, severity_score=30.0,
            unique_review_count=15, avg_rating=1.0, example_quotes=["q2"],
            summary="second", **key,
        )
        s.commit()
        assert second.id == first_id
        assert s.scalar(select(func.count()).select_from(Incident)) == 1
        assert second.observed_volume == 15
        assert second.summary == "second"


# ---- anomaly_stats ------------------------------------------------------


def test_collect_anomaly_stats_after_flow(anomaly_corpus):
    anomaly_flow(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model=_MODEL,
        granularity="cluster", min_history_weeks=3, z_threshold=2.0, min_volume=3,
    )
    stats = collect_anomaly_stats(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model_name=_MODEL
    )
    assert stats["total_incidents"] == 1
    assert stats["incidents_by_aspect"] == {"charging": 1}
    assert stats["incidents_by_granularity"] == {"cluster": 1}
    assert len(stats["top_incidents"]) == 1
    top = stats["top_incidents"][0]
    assert top["aspect_code"] == "charging"
    assert top["observed_volume"] == 12
    assert top["summary"]


def test_collect_anomaly_stats_empty_when_no_incidents(fresh_engine):
    stats = collect_anomaly_stats(
        aspect_version=_ASPECT_VERSION, provider=_PROVIDER, model_name=_MODEL
    )
    assert stats["total_incidents"] == 0
    assert stats["incidents_by_aspect"] == {}
    assert stats["top_incidents"] == []
