from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from voicelens.config import AMAZON_FIXTURE_PATH, BRAND_ALLOWLIST
from voicelens.db.models import Brand, DQEvent, IngestRun, Review, Sku


def test_fixture_flow_loads_expected_rows(fresh_engine):
    from voicelens.pipeline.flows.amazon_ingest_flow import amazon_ingest_flow

    summary = amazon_ingest_flow(
        input_path=AMAZON_FIXTURE_PATH,
        brands=BRAND_ALLOWLIST,
        limit=None,
    )

    assert summary["input_rows"] == 20
    assert summary["matched_rows"] == 18  # Sony + Apple dropped
    assert summary["passed_rows"] == 12
    assert summary["failed_rows"] == 6
    assert summary["loaded_reviews"] == 12

    per = summary["per_check_failed"]
    assert per.get("duplicate_review_id", 0) == 2
    assert per.get("missing_asin", 0) == 1
    assert per.get("invalid_rating", 0) == 1
    assert per.get("text_too_short", 0) == 2

    brand_counts = summary["brand_counts"]
    assert brand_counts.get("Anker", 0) >= 3
    assert brand_counts.get("Soundcore", 0) >= 2
    assert "Sony" not in brand_counts
    assert "Apple" not in brand_counts

    assert isinstance(summary["top_asin_counts"], dict)
    assert len(summary["top_asin_counts"]) > 0

    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        assert s.scalar(select(func.count()).select_from(Review)) == 12
        brand_names = {b.name for b in s.scalars(select(Brand))}
        assert "Anker" in brand_names
        assert "Soundcore" in brand_names
        assert "Sony" not in brand_names
        assert s.scalar(select(func.count()).select_from(Sku)) >= 7
        assert s.scalar(select(func.count()).select_from(IngestRun)) == 1
        assert s.scalar(select(func.count()).select_from(DQEvent)) == 7


def test_fixture_flow_brand_override(fresh_engine):
    """Override brand list to a subset and confirm only that subset loads."""
    from voicelens.pipeline.flows.amazon_ingest_flow import amazon_ingest_flow

    summary = amazon_ingest_flow(
        input_path=AMAZON_FIXTURE_PATH,
        brands=("Anker",),
    )

    brand_counts = summary["brand_counts"]
    assert set(brand_counts.keys()) == {"Anker"}
    assert summary["loaded_reviews"] == brand_counts["Anker"]


def test_fixture_flow_limit_applied(fresh_engine):
    from voicelens.pipeline.flows.amazon_ingest_flow import amazon_ingest_flow

    summary = amazon_ingest_flow(
        input_path=AMAZON_FIXTURE_PATH,
        brands=BRAND_ALLOWLIST,
        limit=5,
    )
    assert summary["matched_rows"] == 5
    assert summary["loaded_reviews"] <= 5


def test_existing_sample_ingest_still_works(fresh_engine):
    """Regression guard for Milestone 0: original ingest_flow must not break."""
    from voicelens.pipeline.flows.ingest_flow import ingest_flow

    summary = ingest_flow()
    assert summary["total_rows"] > 0
    assert summary["passed_rows"] > 0
    assert summary["loaded_reviews"] == summary["passed_rows"]
