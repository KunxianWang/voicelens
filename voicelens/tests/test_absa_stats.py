from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from scripts.absa_stats import collect_absa_stats
from scripts.seed_aspect_ontology import seed_aspect_ontology
from voicelens.pipeline.flows.absa_flow import absa_flow
from voicelens.tests._absa_fixture import seed_reviews as _seed_reviews


def test_absa_stats_on_empty_db(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s)
        s.commit()

    stats = collect_absa_stats()
    assert stats["total_aspect_mentions"] == 0
    assert stats["total_reviews"] == 0
    assert stats["processed_reviews"] == 0
    assert stats["reviews_with_mentions"] == 0
    assert stats["reviews_no_mentions"] == 0
    assert stats["invalid_reviews"] == 0
    assert stats["failed_reviews"] == 0
    assert stats["processed_coverage_rate"] == 0.0
    assert stats["mention_coverage_rate"] == 0.0
    assert stats["aspect_distribution"] == {}
    assert stats["sentiment_distribution"] == {}
    assert stats["severity_distribution"] == {}
    assert stats["top_brands_by_negative_mentions"] == {}


def test_absa_stats_after_smoke_run(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s)
        _seed_reviews(s)
        s.commit()

    summary = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    stats = collect_absa_stats()
    assert stats["total_aspect_mentions"] > 0
    assert stats["total_reviews"] >= 6
    assert stats["reviews_with_mentions"] > 0
    assert stats["processed_reviews"] == summary["processed_reviews"]
    assert 0.0 <= stats["processed_coverage_rate"] <= 1.0
    assert 0.0 <= stats["mention_coverage_rate"] <= 1.0
    assert stats["mention_coverage_rate"] <= stats["processed_coverage_rate"]
    assert stats["evidence_verbatim_rate"] == 1.0
    assert set(stats["aspect_distribution"]).issubset(
        {"battery", "charging", "overheating", "sound_quality",
         "bluetooth", "delivery", "price"}
    )
    assert set(stats["sentiment_distribution"]).issubset(
        {"positive", "neutral", "negative"}
    )


def test_absa_stats_reports_processed_reviews_without_mentions(fresh_engine):
    """A review that produced zero aspects must still count as processed."""
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s)
        _seed_reviews(s)
        s.commit()

    absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    stats = collect_absa_stats()
    # processed reviews >= reviews_with_mentions (because some may have no_mentions)
    assert stats["processed_reviews"] >= stats["reviews_with_mentions"]
