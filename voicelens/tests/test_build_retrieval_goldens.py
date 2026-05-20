"""Schema + write-path tests for the retrieval-golden generator."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from scripts.build_retrieval_goldens import (
    QUERY_TEMPLATES,
    QueryTemplate,
    _collect_gold_ids,
    build_goldens,
    write_jsonl,
)
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


def _seed_one_review(
    session,
    *,
    text: str,
    aspect_code: str,
    sentiment: str,
    severity: str | None,
    evidence_quote: str,
    brand: str = "Anker",
    asin: str = "B0X",
    source_id: str = "S1",
    rating: int = 1,
    aspect_version: str = "v2",
    provider: str = "anthropic",
    model_name: str = "claude-opus-4.6",
) -> int:
    run = session.scalar(
        select(IngestRun).where(IngestRun.source == "amazon_reviews_2023_mvp_subset")
    )
    if run is None:
        run = IngestRun(source="amazon_reviews_2023_mvp_subset", status="completed")
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
    aspect_row = session.scalar(
        select(AspectOntology).where(
            AspectOntology.version == aspect_version,
            AspectOntology.code == aspect_code,
        )
    )
    assert aspect_row is not None, "aspect ontology must be seeded first"
    session.add(
        AspectMention(
            review_id=review.id,
            aspect_id=aspect_row.id,
            sentiment=sentiment,
            severity=severity,
            evidence_quote=evidence_quote,
            model_name=model_name,
            aspect_version=aspect_version,
        )
    )
    session.add(
        ABSAReviewStatus(
            review_id=review.id,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model_name,
            status="success",
            n_mentions=1,
            n_errors=0,
            error_codes_json=None,
        )
    )
    session.flush()
    return review.id


def test_query_templates_cover_every_v2_aspect():
    """Sanity check: each ontology v2 aspect has at least one template."""
    seen_aspects = {t.expected_aspect for t in QUERY_TEMPLATES if t.expected_aspect}
    required = {
        "battery", "charging", "overheating", "sound_quality",
        "bluetooth", "delivery", "price", "reliability",
    }
    missing = required - seen_aspects
    assert not missing, f"missing templates for aspects: {missing}"


def test_query_templates_have_unique_ids():
    ids = [t.query_id for t in QUERY_TEMPLATES]
    assert len(ids) == len(set(ids)), "QueryTemplate.query_id must be unique"


def test_collect_gold_ids_filters_by_aspect_and_quote(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s, version="v2")
        rid_match = _seed_one_review(
            s,
            text="Battery died and then it STOPPED working completely.",
            aspect_code="reliability",
            sentiment="negative",
            severity="medium",
            evidence_quote="STOPPED working completely",
            source_id="match",
        )
        # Different aspect — should NOT match a reliability template.
        _seed_one_review(
            s,
            text="Charging cable broke.",
            aspect_code="charging",
            sentiment="negative",
            severity="medium",
            evidence_quote="Charging cable broke",
            asin="B0Y",
            source_id="other_aspect",
        )
        # Right aspect, wrong evidence_quote phrase — should NOT match.
        _seed_one_review(
            s,
            text="Defective unit; sent it back.",
            aspect_code="reliability",
            sentiment="negative",
            severity="medium",
            evidence_quote="Defective unit",
            asin="B0Z",
            source_id="wrong_phrase",
        )
        s.commit()

    template = QueryTemplate(
        "rel-test", "stopped working",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("stopped working",),
    )
    with SessionLocal() as s:
        ids = _collect_gold_ids(
            s, template,
            aspect_version="v2", provider="anthropic", model_name="claude-opus-4.6",
            per_query=5,
        )
    assert ids == [rid_match]


def test_build_goldens_emits_required_schema(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        seed_aspect_ontology(s, version="v2")
        s.commit()

    rows = build_goldens(
        aspect_version="v2",
        provider="anthropic",
        model_name="claude-opus-4.6",
        per_query=3,
    )
    assert rows
    for row in rows:
        assert isinstance(row["query_id"], str)
        assert isinstance(row["query"], str)
        assert "gold_review_ids" in row
        assert isinstance(row["gold_review_ids"], list)
        assert "notes" in row


def test_write_jsonl_round_trip(tmp_path: Path):
    rows = [
        {"query_id": "x", "query": "y", "gold_review_ids": [1, 2], "notes": ""},
        {"query_id": "z", "query": "w", "gold_review_ids": [], "notes": "no hits"},
    ]
    path = tmp_path / "goldens.jsonl"
    write_jsonl(rows, path)
    contents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert contents == rows
