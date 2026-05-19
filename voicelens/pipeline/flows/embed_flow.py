"""Embedding + Qdrant indexing flow (M3A).

Selects reviews from Postgres that have already been ABSA-processed under
``(aspect_version, provider, model_name)``, builds an embedding text
combining the raw review with a short aspect summary, embeds in batches,
and upserts the resulting vectors + payloads into a Qdrant collection.

By default only reviews with ``ABSAReviewStatus.status`` in
``{success, no_mentions}`` are indexed; ``failed`` and ``invalid`` rows
are deliberately skipped because their evidence is unreliable.

Out of scope for M3A:
- BERTopic clustering, anomaly detection
- Full RAG / LangGraph
- Streamlit
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from prefect import flow, get_run_logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from voicelens.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from voicelens.db.engine import get_session
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Review,
    Sku,
)
from voicelens.nlp.absa import LATEST_ONTOLOGY_VERSION
from voicelens.retrieval.embeddings import (
    EmbeddingProvider,
    chunked,
    get_embedding_provider,
)
from voicelens.retrieval.qdrant_index import (
    build_embedding_text,
    build_payload,
    build_point_id,
    ensure_collection,
    upsert_points,
)

DEFAULT_STATUS_INCLUDE = (
    ABSAReviewStatus.STATUS_SUCCESS,
    ABSAReviewStatus.STATUS_NO_MENTIONS,
)


def _resolve_review_rows(
    session: Session,
    *,
    aspect_version: str,
    provider: str,
    model_name: str,
    statuses: Iterable[str],
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], int]:
    """Pull review + status + aspect_mention rows in three focused queries.

    Returns ``(reviews, mentions_by_review_id, skipped_count)`` where
    ``skipped_count`` is the number of ABSAReviewStatus rows that
    matched ``(aspect_version, provider, model_name)`` but whose status
    was NOT in ``statuses`` (so the caller can report how many failed /
    invalid rows were excluded).
    """
    status_stmt = select(ABSAReviewStatus.review_id, ABSAReviewStatus.status).where(
        and_(
            ABSAReviewStatus.aspect_version == aspect_version,
            ABSAReviewStatus.provider == provider,
            ABSAReviewStatus.model_name == model_name,
        )
    )
    status_rows = list(session.execute(status_stmt))
    include_set = set(statuses)
    eligible_ids = [int(r.review_id) for r in status_rows if r.status in include_set]
    skipped = sum(1 for r in status_rows if r.status not in include_set)
    if not eligible_ids:
        return [], {}, skipped

    review_stmt = (
        select(
            Review.id,
            Review.source,
            Review.source_id,
            Review.sku_id,
            Review.rating,
            Review.verified,
            Review.posted_at,
            Review.text_raw,
            Sku.asin,
            Brand.name.label("brand"),
        )
        .join(Sku, Sku.id == Review.sku_id)
        .join(Brand, Brand.id == Sku.brand_id)
        .where(Review.id.in_(eligible_ids))
        .order_by(Review.id)
    )
    if limit is not None and limit > 0:
        review_stmt = review_stmt.limit(limit)
    review_rows = [
        {
            "review_id": int(row.id),
            "source": row.source,
            "source_id": row.source_id,
            "sku_id": int(row.sku_id) if row.sku_id is not None else None,
            "asin": row.asin,
            "brand": row.brand,
            "rating": int(row.rating) if row.rating is not None else None,
            "verified": bool(row.verified) if row.verified is not None else None,
            "posted_at": row.posted_at.isoformat() if row.posted_at else None,
            "text_raw": row.text_raw,
        }
        for row in session.execute(review_stmt)
    ]
    review_ids_in_scope = [r["review_id"] for r in review_rows]
    status_by_id = {
        int(r.review_id): r.status
        for r in status_rows
        if r.status in include_set and int(r.review_id) in set(review_ids_in_scope)
    }
    for row in review_rows:
        row["absa_status"] = status_by_id.get(row["review_id"], "")

    mention_stmt = (
        select(
            AspectMention.review_id,
            AspectOntology.code,
            AspectMention.sentiment,
            AspectMention.severity,
            AspectMention.evidence_quote,
        )
        .join(AspectOntology, AspectOntology.id == AspectMention.aspect_id)
        .where(
            and_(
                AspectMention.review_id.in_(review_ids_in_scope),
                AspectMention.aspect_version == aspect_version,
                AspectMention.model_name == model_name,
            )
        )
    )
    mentions_by_review: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in session.execute(mention_stmt):
        mentions_by_review[int(row.review_id)].append(
            {
                "aspect_code": row.code,
                "sentiment": row.sentiment,
                "severity": row.severity,
                "evidence_quote": row.evidence_quote,
            }
        )
    return review_rows, dict(mentions_by_review), skipped


def _build_points(
    *,
    reviews: list[dict[str, Any]],
    mentions_by_review: dict[int, list[dict[str, Any]]],
    embedder: EmbeddingProvider,
    aspect_version: str,
    provider: str,
    model_name: str,
    batch_size: int,
) -> Iterable[rest.PointStruct]:
    """Embed in chunks and yield Qdrant ``PointStruct`` instances."""
    review_texts: list[str] = []
    for review in reviews:
        text = build_embedding_text(
            review.get("text_raw") or "",
            mentions_by_review.get(review["review_id"], []),
        )
        review_texts.append(text)

    yielded = 0
    for chunk in chunked(review_texts, batch_size):
        # Slice the parallel review list to match the chunk.
        chunk_reviews = reviews[yielded : yielded + len(chunk)]
        vectors = embedder.embed_batch(chunk)
        for review, vector in zip(chunk_reviews, vectors, strict=True):
            mentions = mentions_by_review.get(review["review_id"], [])
            payload = build_payload(
                review=review,
                mentions=mentions,
                aspect_version=aspect_version,
                provider=provider,
                model_name=model_name,
                absa_status=str(review.get("absa_status") or ""),
            )
            point_id = build_point_id(
                review["review_id"], aspect_version, provider, model_name
            )
            yield rest.PointStruct(id=point_id, vector=vector, payload=payload)
        yielded += len(chunk)


def _open_client(qdrant_url: str | None) -> QdrantClient:
    target = (qdrant_url or QDRANT_URL).strip()
    if target in ("", ":memory:", "memory://"):
        return QdrantClient(":memory:")
    return QdrantClient(url=target)


@flow(name="embed_flow")
def embed_flow(
    *,
    aspect_version: str = LATEST_ONTOLOGY_VERSION,
    provider: str = "anthropic",
    model: str = "claude-opus-4.6",
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    collection: str | None = None,
    qdrant_url: str | None = None,
    statuses: tuple[str, ...] | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    recreate_collection: bool = False,
    client: QdrantClient | None = None,
    embedder: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    try:
        logger = get_run_logger()
    except Exception:  # pragma: no cover - prefect ctx missing
        logger = logging.getLogger("embed_flow")

    embedder = embedder or get_embedding_provider(
        embedding_provider or EMBEDDING_PROVIDER, model=embedding_model or EMBEDDING_MODEL
    )
    collection_name = collection or QDRANT_COLLECTION
    statuses_to_include = tuple(statuses) if statuses else DEFAULT_STATUS_INCLUDE
    batch = batch_size if (batch_size and batch_size > 0) else EMBEDDING_BATCH_SIZE

    with get_session() as session:
        reviews, mentions_by_review, skipped = _resolve_review_rows(
            session,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model,
            statuses=statuses_to_include,
            limit=limit,
        )

    summary: dict[str, Any] = {
        "selected_reviews": len(reviews),
        "embedded_reviews": 0,
        "upserted_points": 0,
        "skipped_failed_or_invalid": skipped,
        "collection": collection_name,
        "embedding_provider": embedder.name,
        "embedding_model": getattr(embedder, "model_name_str", None)
        or getattr(embedder, "model_id", None)
        or embedder.name,
        "vector_size": None,
        "aspect_version": aspect_version,
        "absa_provider": provider,
        "absa_model": model,
        "statuses_included": list(statuses_to_include),
    }

    if not reviews:
        logger.info(
            "embed_flow: no reviews matched (aspect_version=%s provider=%s model=%s)",
            aspect_version, provider, model,
        )
        return summary

    vector_size = int(embedder.dimension)
    if vector_size <= 0:
        raise RuntimeError(
            f"Embedding provider {embedder.name!r} reported non-positive dimension {vector_size}; "
            "cannot create or upsert into Qdrant."
        )
    summary["vector_size"] = vector_size

    qdrant = client or _open_client(qdrant_url)
    ensure_collection(
        qdrant, collection_name, vector_size=vector_size, recreate=recreate_collection
    )

    logger.info(
        "embed_flow: embedding %d reviews with %s into %s (vector_size=%d, batch=%d)",
        len(reviews), embedder.name, collection_name, vector_size, batch,
    )
    n_upserted = upsert_points(
        qdrant,
        collection_name,
        _build_points(
            reviews=reviews,
            mentions_by_review=mentions_by_review,
            embedder=embedder,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model,
            batch_size=batch,
        ),
    )
    summary["embedded_reviews"] = len(reviews)
    summary["upserted_points"] = n_upserted
    return summary


def _parse_statuses(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed ABSA-processed reviews into Qdrant.")
    parser.add_argument("--aspect-version", default=LATEST_ONTOLOGY_VERSION)
    parser.add_argument("--provider", default="anthropic", help="ABSA provider (filter on absa_review_status)")
    parser.add_argument("--model", default="claude-opus-4.6", help="ABSA model_name (filter on absa_review_status)")
    parser.add_argument("--embedding-provider", default=None, help="Override EMBEDDING_PROVIDER")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL")
    parser.add_argument("--collection", default=None, help="Override QDRANT_COLLECTION")
    parser.add_argument("--qdrant-url", default=None, help="Override QDRANT_URL (use ':memory:' for ephemeral)")
    parser.add_argument(
        "--statuses",
        default=None,
        help="Comma-separated ABSAReviewStatus values to include (default: success,no_mentions)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before upsert",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = embed_flow(
        aspect_version=args.aspect_version,
        provider=args.provider,
        model=args.model,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        statuses=_parse_statuses(args.statuses),
        limit=args.limit,
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
