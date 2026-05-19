"""Shared seed helpers for ABSA-related tests.

Not a test module itself — pytest skips files prefixed with ``_``.
Centralising the seed helper here lets ``conftest.py`` and every test
module load the same review fixtures without re-importing across test
files (which trips ruff's F811).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from voicelens.db.models import Brand, IngestRun, Review, Sku

_FIXED_POSTED_AT = datetime(2024, 1, 1, 12, 0, 0)

# Last entry is intentionally keyword-free so MockABSAProvider returns
# ``{"aspects": []}`` — exercises the no_mentions status path.
REVIEW_TEXTS: tuple[tuple[str, str, str, int, str], ...] = (
    ("Anker-1", "Anker", "B0ANK10001", 5,
     "The battery lasts forever and the charging is super fast. Love it."),
    ("Anker-2", "Anker", "B0ANK10002", 1,
     "The battery is broken after one week and it stopped working completely. Junk."),
    ("Soundcore-1", "Soundcore", "B0SND20001", 2,
     "Bluetooth pairing drops constantly and the sound quality is terrible."),
    ("Bose-1", "Bose", "B0BOS30001", 5,
     "Amazing sound quality and the bluetooth pairs instantly with my phone."),
    ("JBL-1", "JBL", "B0JBL40001", 3,
     "The delivery was on time but the packaging was damaged on arrival."),
    ("UGREEN-1", "UGREEN", "B0UGR50001", 4,
     "Great value for the price and the USB-C charging works perfectly."),
    ("NoAspects-1", "Anker", "B0ANK10003", 4,
     "Bought this last weekend and have been using it ever since. Looks fine."),
)


def seed_reviews(session) -> dict[str, int]:
    """Populate brand/sku/ingest_run/review rows; return {source_id: review_id}."""
    run = IngestRun(source="amazon_reviews_2023_mvp_subset", status="completed")
    session.add(run)
    session.flush()
    extra_run = IngestRun(source="other_source", status="completed")
    session.add(extra_run)
    session.flush()

    review_ids: dict[str, int] = {}
    for source_id, brand_name, asin, rating, text in REVIEW_TEXTS:
        brand = session.scalar(select(Brand).where(Brand.name == brand_name))
        if brand is None:
            brand = Brand(name=brand_name)
            session.add(brand)
            session.flush()
        sku = session.scalar(
            select(Sku).where(Sku.brand_id == brand.id, Sku.asin == asin)
        )
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
            ingest_run_id=run.id,
        )
        session.add(review)
        session.flush()
        review_ids[source_id] = review.id

    other_run_review = Review(
        source="amazon_reviews_2023",
        source_id="OTHER-1",
        sku_id=session.scalar(select(Sku.id).limit(1)),
        language="en",
        lang_confidence=0.95,
        rating=4,
        posted_at=_FIXED_POSTED_AT,
        text_raw="The battery on this is fine.",
        char_len=27,
        ingest_run_id=extra_run.id,
    )
    session.add(other_run_review)
    session.flush()
    review_ids["OTHER-1"] = other_run_review.id
    return review_ids


# Back-compat alias for tests that still import _seed_reviews by the
# original underscore-prefixed name. Remove once those imports are gone.
_seed_reviews = seed_reviews
