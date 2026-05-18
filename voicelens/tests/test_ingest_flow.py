from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from voicelens.db.models import Brand, DQEvent, IngestRun, Review, Sku


@pytest.fixture
def jsonl_path(tmp_path: Path, sample_row_factory) -> Path:
    rows = [
        sample_row_factory(source_id="R-good-1"),
        sample_row_factory(source_id="R-good-2", asin="B0SND20001", brand="Soundcore",
                           model_number="Liberty-4-NC", category="earbuds",
                           text_raw="ANC is great for long flights and bluetooth pairs instantly."),
        sample_row_factory(source_id="R-bad-rating", rating=0),
        sample_row_factory(source_id="R-no-asin", asin=None),
        sample_row_factory(source_id="R-de", language="de",
                           text_raw="Sehr gutes Produkt, schnelles Aufladen."),
        sample_row_factory(source_id="R-low-conf", lang_confidence=0.3),
        sample_row_factory(source_id="R-short", text_raw="ok"),
        sample_row_factory(source_id="R-dup-1"),
        sample_row_factory(source_id="R-dup-1"),
    ]
    out = tmp_path / "reviews.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out


def test_ingest_flow_smoke(jsonl_path, fresh_engine):
    from voicelens.pipeline.flows.ingest_flow import ingest_flow

    summary = ingest_flow(path=jsonl_path)

    assert summary["total_rows"] == 9
    assert summary["passed_rows"] == 3
    assert summary["failed_rows"] == 6
    assert summary["loaded_reviews"] == 3

    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_reviews = s.scalar(select(func.count()).select_from(Review))
        n_brands = s.scalar(select(func.count()).select_from(Brand))
        n_skus = s.scalar(select(func.count()).select_from(Sku))
        n_runs = s.scalar(select(func.count()).select_from(IngestRun))
        n_dq = s.scalar(select(func.count()).select_from(DQEvent))

    assert n_reviews == 3
    assert n_brands == 2
    assert n_skus == 2
    assert n_runs == 1
    assert n_dq == 7


def test_valid_rows_loaded_into_postgres_like_store(jsonl_path, fresh_engine):
    """The 'valid sample rows are loaded' acceptance check."""
    from voicelens.pipeline.flows.ingest_flow import ingest_flow

    ingest_flow(path=jsonl_path)

    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        reviews = list(s.scalars(select(Review)))
        source_ids = sorted(r.source_id for r in reviews)
        assert source_ids == ["R-dup-1", "R-good-1", "R-good-2"]
        for r in reviews:
            assert r.char_len == len(r.text_raw)
            assert r.language == "en"
            assert r.lang_confidence >= 0.7
            assert 1 <= r.rating <= 5
            assert r.sku is not None
            assert r.sku.brand is not None


def test_idempotent_rerun(jsonl_path, fresh_engine):
    from voicelens.pipeline.flows.ingest_flow import ingest_flow

    s1 = ingest_flow(path=jsonl_path)
    s2 = ingest_flow(path=jsonl_path)

    assert s1["loaded_reviews"] == 3
    assert s2["loaded_reviews"] == 0

    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_reviews = s.scalar(select(func.count()).select_from(Review))
        n_runs = s.scalar(select(func.count()).select_from(IngestRun))
    assert n_reviews == 3
    assert n_runs == 2


def test_seed_sample_set_runs_clean(fresh_engine):
    from scripts.seed_sample import generate
    rows = generate()
    assert len(rows) >= 100

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name

    from voicelens.pipeline.flows.ingest_flow import ingest_flow
    summary = ingest_flow(path=path)

    assert summary["total_rows"] == len(rows)
    assert summary["passed_rows"] > 0
    assert summary["loaded_reviews"] == summary["passed_rows"]
    assert summary["dq_pass_rate"] > 0.5
