"""Tests for the dashboard's read-only query helpers (voicelens.ui.db)."""
from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy.orm import sessionmaker

from scripts.seed_aspect_ontology import seed_aspect_ontology
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Cluster,
    DQEvent,
    Incident,
    IngestRun,
    Review,
    ReviewCluster,
    Sku,
)
from voicelens.ui import db

_AV = "v2"
_PROVIDER = "anthropic"
_MODEL = "claude-opus-4.6"


def _seed_dashboard_corpus(session) -> dict:
    """Seed a small but complete corpus touching every dashboard table."""
    seed_aspect_ontology(session, version=_AV)
    aspect_id = {
        row.code: row.id
        for row in session.query(AspectOntology).filter_by(version=_AV).all()
    }

    run = IngestRun(source="amazon_reviews_2023", status="completed", n_rows=6)
    session.add(run)
    session.flush()

    brands = {}
    for name in ("Anker", "Soundcore"):
        b = Brand(name=name)
        session.add(b)
        session.flush()
        brands[name] = b.id

    def _sku(brand: str, asin: str) -> int:
        s = Sku(brand_id=brands[brand], asin=asin)
        session.add(s)
        session.flush()
        return s.id

    # (brand, asin, rating, absa_status, [(aspect, sentiment, severity, quote)])
    spec = [
        ("Anker", "A1", 1, "success",
         [("reliability", "negative", "high", "stopped working after a week")]),
        ("Anker", "A2", 1, "success",
         [("reliability", "negative", "medium", "broke on the second use")]),
        ("Anker", "A3", 2, "success",
         [("charging", "negative", "medium", "would not charge at all")]),
        ("Soundcore", "S1", 5, "success",
         [("price", "positive", None, "great value for the price")]),
        ("Soundcore", "S2", 3, "no_mentions", []),
        ("Soundcore", "S3", 1, "failed",
         [("reliability", "negative", "high", "dead on arrival")]),
    ]
    review_ids: list[int] = []
    for i, (brand, asin, rating, status, mentions) in enumerate(spec):
        sku_id = _sku(brand, asin)
        text = f"Review {i} about {brand} {asin}: the product had issues."
        review = Review(
            source="amazon_reviews_2023", source_id=f"R{i}", sku_id=sku_id,
            language="en", lang_confidence=0.95, rating=rating,
            posted_at=datetime(2024, 1, 1 + i), text_raw=text,
            char_len=len(text), ingest_run_id=run.id,
        )
        session.add(review)
        session.flush()
        review_ids.append(review.id)
        session.add(
            ABSAReviewStatus(
                review_id=review.id, aspect_version=_AV, provider=_PROVIDER,
                model_name=_MODEL, status=status, n_mentions=len(mentions),
                n_errors=1 if status == "failed" else 0,
            )
        )
        # the failed-status review's mentions are still excluded downstream,
        # but a row is written so the status distribution has a "failed".
        if status == "failed":
            continue
        for code, sentiment, severity, quote in mentions:
            session.add(
                AspectMention(
                    review_id=review.id, aspect_id=aspect_id[code],
                    sentiment=sentiment, severity=severity, evidence_quote=quote,
                    model_name=_MODEL, aspect_version=_AV,
                )
            )

    # two clusters over the reliability / charging negatives
    rel_cluster = Cluster(
        run_id="r1", aspect_version=_AV, provider=_PROVIDER, model_name=_MODEL,
        aspect_code="reliability", label="stopped working broke",
        algorithm="tfidf_kmeans", size=2, severity_weighted_size=7.0,
        topic_keywords=["stopped", "working", "broke"],
        representative_quotes=["stopped working after a week"],
    )
    chg_cluster = Cluster(
        run_id="r1", aspect_version=_AV, provider=_PROVIDER, model_name=_MODEL,
        aspect_code="charging", label="would not charge",
        algorithm="tfidf_kmeans", size=1, severity_weighted_size=2.0,
        topic_keywords=["charge", "charging"],
        representative_quotes=["would not charge at all"],
    )
    session.add_all([rel_cluster, chg_cluster])
    session.flush()
    session.add_all([
        ReviewCluster(cluster_id=rel_cluster.id, review_id=review_ids[0],
                      aspect_code="reliability", severity="high"),
        ReviewCluster(cluster_id=rel_cluster.id, review_id=review_ids[1],
                      aspect_code="reliability", severity="medium"),
        ReviewCluster(cluster_id=chg_cluster.id, review_id=review_ids[2],
                      aspect_code="charging", severity="medium"),
    ])

    # one cluster-level + one aspect-level incident
    session.add_all([
        Incident(
            run_id="run-c", granularity="cluster", aspect_version=_AV,
            provider=_PROVIDER, model_name=_MODEL, cluster_id=rel_cluster.id,
            aspect_code="reliability", cluster_label="stopped working broke",
            week_start=date(2024, 1, 1), observed_volume=8, baseline_volume=2.1,
            z_score=3.4, severity_score=20.0, unique_review_count=8,
            avg_rating=1.3, example_quotes=["stopped working after a week"],
            summary="Reliability spike in cluster.", status="open",
        ),
        Incident(
            run_id="run-a", granularity="aspect", aspect_version=_AV,
            provider=_PROVIDER, model_name=_MODEL, cluster_id=None,
            aspect_code="charging", cluster_label=None,
            week_start=date(2024, 2, 5), observed_volume=5, baseline_volume=1.8,
            z_score=2.2, severity_score=10.0, unique_review_count=5,
            avg_rating=2.0, example_quotes=["would not charge at all"],
            summary="Charging spike overall.", status="open",
        ),
    ])

    # data-quality events — each check_name is itself a failure reason.
    session.add_all([
        DQEvent(ingest_run_id=run.id, check_name="text_too_short", n_rows=6,
                n_failed=1, reason_codes={"examples": []}),
        DQEvent(ingest_run_id=run.id, check_name="non_english_language",
                n_rows=6, n_failed=2, reason_codes={"examples": []}),
        DQEvent(ingest_run_id=run.id, check_name="invalid_rating", n_rows=6,
                n_failed=0, reason_codes=None),
    ])
    session.commit()
    return {"review_ids": review_ids}


@pytest.fixture
def dashboard_db(fresh_engine):
    """A test DB seeded with the full dashboard corpus."""
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        _seed_dashboard_corpus(s)
    yield fresh_engine


# ---- overview -----------------------------------------------------------


def test_overview_stats(dashboard_db):
    stats = db.get_overview_stats()
    assert stats["total_reviews"] == 6
    assert stats["total_brands"] == 2
    assert stats["absa_processed_reviews"] == 6  # every review has a status row
    assert stats["total_aspect_mentions"] == 4   # failed review's mentions skipped
    assert stats["total_clusters"] == 2
    assert stats["total_incidents"] == 2
    assert stats["total_ingest_runs"] == 1


def test_reviews_by_brand(dashboard_db):
    rows = db.get_reviews_by_brand()
    counts = {r["brand"]: r["reviews"] for r in rows}
    assert counts == {"Anker": 3, "Soundcore": 3}


def test_absa_status_distribution(dashboard_db):
    dist = db.get_absa_status_distribution()
    assert dist["success"] == 4
    assert dist["no_mentions"] == 1
    assert dist["failed"] == 1
    assert dist["invalid"] == 0  # key always present even at zero


def test_aspect_and_sentiment_distribution(dashboard_db):
    aspects = {r["aspect_code"]: r["mentions"] for r in db.get_aspect_distribution()}
    assert aspects["reliability"] == 2
    assert aspects["charging"] == 1
    sentiments = db.get_sentiment_distribution()
    assert sentiments["negative"] == 3
    assert sentiments["positive"] == 1


def test_list_brands_and_aspects(dashboard_db):
    assert db.list_brands() == ["Anker", "Soundcore"]
    assert set(db.list_aspects()) == {"reliability", "charging", "price"}


# ---- ABSA distribution --------------------------------------------------


def test_aspect_sentiment_matrix_and_filter(dashboard_db):
    full = db.get_aspect_sentiment_matrix()
    assert {(r["aspect_code"], r["sentiment"]): r["count"] for r in full}[
        ("reliability", "negative")
    ] == 2
    anker_only = db.get_aspect_sentiment_matrix(brand="Anker")
    assert all(r["aspect_code"] != "price" for r in anker_only)


def test_severity_distribution_negatives_only(dashboard_db):
    sev = db.get_severity_distribution()
    assert sev["high"] == 1   # only the success-status reliability high
    assert sev["medium"] == 2


def test_negative_examples_carry_quotes(dashboard_db):
    examples = db.get_negative_examples(aspect="reliability")
    assert len(examples) == 2
    assert all(e["evidence_quote"] for e in examples)
    assert all(e["aspect_code"] == "reliability" for e in examples)


# ---- clusters -----------------------------------------------------------


def test_clusters_ranked_by_severity_weight(dashboard_db):
    clusters = db.get_clusters()
    assert [c["aspect_code"] for c in clusters] == ["reliability", "charging"]
    assert clusters[0]["severity_weighted_size"] == 7.0
    members = db.get_cluster_members(clusters[0]["cluster_id"])
    assert len(members) == 2
    assert {m["aspect_code"] for m in members} == {"reliability"}


def test_clusters_filtered_by_aspect(dashboard_db):
    only_charging = db.get_clusters(aspect="charging")
    assert len(only_charging) == 1
    assert only_charging[0]["aspect_code"] == "charging"


# ---- incidents ----------------------------------------------------------


def test_incidents_ranked_and_filterable(dashboard_db):
    incidents = db.get_incidents()
    assert len(incidents) == 2
    assert incidents[0]["severity_score"] >= incidents[1]["severity_score"]
    assert db.list_incident_granularities() == ["aspect", "cluster"]
    assert set(db.list_incident_aspects()) == {"reliability", "charging"}
    aspect_level = db.get_incidents(granularity="aspect")
    assert len(aspect_level) == 1
    assert aspect_level[0]["cluster_id"] is None


# ---- data quality -------------------------------------------------------


def test_ingest_runs_and_dq_summary(dashboard_db):
    runs = db.get_ingest_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    dq = db.get_dq_summary()
    assert dq["total_events"] == 3
    assert dq["total_failed"] == 3
    assert dq["failures_by_reason"]["non_english_language"] == 2
    assert dq["failures_by_reason"]["text_too_short"] == 1
    # a check with zero failures is not a "reason"
    assert "invalid_rating" not in dq["failures_by_reason"]


def test_pipeline_coverage(dashboard_db):
    cov = db.get_pipeline_coverage()
    assert cov["total_reviews"] == 6
    assert cov["absa_processed_reviews"] == 6
    assert cov["processed_coverage_rate"] == 1.0
    assert 0.0 < cov["mention_coverage_rate"] <= 1.0
    assert cov["absa_failed_rate"] == pytest.approx(1 / 6, abs=1e-3)


# ---- empty-state safety -------------------------------------------------


def test_queries_on_empty_db_do_not_crash(fresh_engine):
    """Every helper must return a benign empty value on a fresh DB."""
    assert db.get_overview_stats()["total_reviews"] == 0
    assert db.get_reviews_by_brand() == []
    assert db.get_aspect_distribution() == []
    assert db.get_sentiment_distribution() == {}
    assert db.list_brands() == []
    assert db.list_aspects() == []
    assert db.get_aspect_sentiment_matrix() == []
    assert db.get_negative_examples() == []
    assert db.get_clusters() == []
    assert db.get_cluster_members(999) == []
    assert db.get_incidents() == []
    assert db.list_incident_aspects() == []
    assert db.list_incident_granularities() == []
    assert db.get_ingest_runs() == []
    assert db.get_dq_summary()["total_events"] == 0
    assert db.get_pipeline_coverage()["processed_coverage_rate"] == 0.0
