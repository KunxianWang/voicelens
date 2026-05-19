from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from scripts.sample_absa_holdout import sample_holdout, write_seed
from scripts.seed_aspect_ontology import seed_aspect_ontology
from voicelens.tests._absa_fixture import seed_reviews as _seed_reviews


def test_holdout_sample_is_deterministic(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s)
        _seed_reviews(s)
        s.commit()

    first = sample_holdout(target_count=4, seed=42, source="amazon_reviews_2023_mvp_subset")
    second = sample_holdout(target_count=4, seed=42, source="amazon_reviews_2023_mvp_subset")

    assert [r["review_id"] for r in first] == [r["review_id"] for r in second]


def test_holdout_sample_includes_gold_aspects_field(tmp_path: Path, fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s)
        _seed_reviews(s)
        s.commit()

    rows = sample_holdout(target_count=3, source="amazon_reviews_2023_mvp_subset")
    out_path = tmp_path / "holdout.jsonl"
    write_seed(out_path, rows)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines, "scaffold must contain at least one row"
    for line in lines:
        payload = json.loads(line)
        assert "gold_aspects" in payload
        assert payload["gold_aspects"] == []
        for key in ("review_id", "source_id", "text_raw", "brand", "asin", "rating", "posted_at"):
            assert key in payload


def test_holdout_sample_balances_across_brands(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s)
        _seed_reviews(s)
        s.commit()

    rows = sample_holdout(target_count=6, source="amazon_reviews_2023_mvp_subset")
    brands = {r["brand"] for r in rows}
    assert len(brands) >= 4, f"expected multiple brands in 6-row sample, got {brands}"


def test_holdout_sample_handles_empty_db(fresh_engine):
    rows = sample_holdout(target_count=10, source="amazon_reviews_2023_mvp_subset")
    assert rows == []
