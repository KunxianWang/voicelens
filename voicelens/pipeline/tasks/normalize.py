from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from voicelens.db.models import Brand, Review, Sku

_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def parse_posted_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        raise ValueError("posted_at is required")
    s = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def get_or_create_brand(session: Session, name: str) -> Brand:
    brand = session.scalar(select(Brand).where(Brand.name == name))
    if brand is None:
        brand = Brand(name=name)
        session.add(brand)
        session.flush()
    return brand


def get_or_create_sku(
    session: Session,
    brand: Brand,
    asin: str,
    model_number: str | None = None,
    category: str | None = None,
) -> Sku:
    sku = session.scalar(
        select(Sku).where(Sku.brand_id == brand.id, Sku.asin == asin)
    )
    if sku is None:
        sku = Sku(
            brand_id=brand.id,
            asin=asin,
            model_number=model_number,
            category=category,
        )
        session.add(sku)
        session.flush()
    return sku


def load_reviews(
    session: Session,
    rows: list[dict[str, Any]],
    ingest_run_id: int,
) -> int:
    """Upsert valid rows into Postgres. Returns count of newly inserted reviews."""
    inserted = 0
    for row in rows:
        brand = get_or_create_brand(session, row["brand"])
        sku = get_or_create_sku(
            session,
            brand,
            asin=row["asin"],
            model_number=row.get("model_number"),
            category=row.get("category"),
        )

        exists = session.scalar(
            select(Review.id).where(
                Review.source == row["source"],
                Review.source_id == str(row["source_id"]),
            )
        )
        if exists is not None:
            continue

        text = clean_text(row["text_raw"])
        review = Review(
            source=row["source"],
            source_id=str(row["source_id"]),
            sku_id=sku.id,
            locale=row.get("locale"),
            language=row["language"],
            lang_confidence=float(row["lang_confidence"]),
            rating=int(row["rating"]),
            verified=bool(row.get("verified", False)),
            posted_at=parse_posted_at(row["posted_at"]),
            helpful_count=int(row.get("helpful_count", 0) or 0),
            text_raw=text,
            char_len=len(text),
            ingest_run_id=ingest_run_id,
        )
        session.add(review)
        inserted += 1
    session.flush()
    return inserted
