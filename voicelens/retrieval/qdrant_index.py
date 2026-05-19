"""Qdrant collection helpers.

Three concerns lifted out of the flow so the rest of the retrieval
package and the tests can use them independently:

- :func:`build_point_id` — deterministic UUIDv5 keyed on
  ``(review_id, aspect_version, provider, model_name)``. The same
  review under the same provider+model+ontology version always maps to
  the same point, so re-running the flow upserts in place rather than
  duplicating.
- :func:`build_payload` — assembles the payload dict from a Review row
  plus its ``AspectMention`` list. ABSA outputs are stored as parallel
  lists (``aspect_codes``, ``sentiments``, ``severities``,
  ``evidence_quotes``) so a single filter on, say, ``aspect_codes``
  matches "any aspect == X".
- :func:`build_embedding_text` — packs ``text_raw`` plus an aspect
  summary so retrieval over the embedding can find reviews that share
  failure modes even when the wording differs.

Qdrant client wrappers are also here for completeness:

- :func:`ensure_collection` is idempotent.
- :func:`upsert_points` chunks into ``UPSERT_BATCH_SIZE``.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

REVIEW_NAMESPACE = uuid.UUID("8d4e0e4e-9d4b-5e0e-9e7a-1a4e2b9c0c4a")
DEFAULT_DISTANCE = rest.Distance.COSINE
UPSERT_BATCH_SIZE = 128


def build_point_id(
    review_id: int, aspect_version: str, provider: str, model_name: str
) -> str:
    """Deterministic UUIDv5 string keyed on (review_id, version, provider, model).

    Returning a string keeps the point id stable across Python sessions
    and JSON-serialisable, and Qdrant accepts UUID strings as point ids.
    """
    key = f"{int(review_id)}|{aspect_version}|{provider}|{model_name}"
    return str(uuid.uuid5(REVIEW_NAMESPACE, key))


def build_embedding_text(text_raw: str, mentions: Sequence[dict[str, Any]] | None) -> str:
    """Concatenate the review text and a short aspect summary.

    Aspect summary format::

        Aspects: reliability negative medium; price positive

    Severity is appended only when present (negative mentions). For
    reviews with no mentions the summary is ``Aspects: (none)``; this is
    important so the embedding still encodes the "no in-ontology issue"
    signal instead of being mostly indistinguishable from the raw text.
    """
    text = (text_raw or "").strip()
    if not mentions:
        aspects_str = "(none)"
    else:
        parts: list[str] = []
        for m in mentions:
            code = m.get("aspect_code") or m.get("code")
            sentiment = m.get("sentiment")
            severity = m.get("severity")
            if not isinstance(code, str):
                continue
            chunk = f"{code} {sentiment or 'neutral'}"
            if isinstance(severity, str) and severity:
                chunk += f" {severity}"
            parts.append(chunk)
        aspects_str = "; ".join(parts) if parts else "(none)"
    return f"Review: {text}\nAspects: {aspects_str}"


def build_payload(
    *,
    review: dict[str, Any],
    mentions: Sequence[dict[str, Any]],
    aspect_version: str,
    provider: str,
    model_name: str,
    absa_status: str,
) -> dict[str, Any]:
    """Construct the Qdrant point payload.

    ``review`` is a plain dict (decoupled from SQLAlchemy ORM so the
    builder is trivially testable). Required keys: ``review_id``,
    ``source_id``, ``source``, ``sku_id``, ``asin``, ``brand``,
    ``rating``, ``verified``, ``posted_at``, ``text_raw``. Missing
    optional keys fall back to safe defaults.
    """
    aspect_codes: list[str] = []
    sentiments: list[str] = []
    severities: list[str | None] = []
    evidence_quotes: list[str] = []
    for m in mentions:
        code = m.get("aspect_code") or m.get("code")
        if not isinstance(code, str):
            continue
        aspect_codes.append(code)
        sentiment = m.get("sentiment")
        sentiments.append(str(sentiment) if sentiment is not None else "")
        severity = m.get("severity")
        severities.append(severity if isinstance(severity, str) else None)
        quote = m.get("evidence_quote")
        evidence_quotes.append(str(quote) if isinstance(quote, str) else "")
    return {
        "review_id": int(review["review_id"]),
        "source_id": str(review.get("source_id") or ""),
        "source": str(review.get("source") or ""),
        "sku_id": int(review.get("sku_id") or 0) if review.get("sku_id") is not None else None,
        "asin": str(review.get("asin") or ""),
        "brand": str(review.get("brand") or ""),
        "rating": int(review["rating"]) if review.get("rating") is not None else None,
        "verified": bool(review.get("verified")) if review.get("verified") is not None else None,
        "posted_at": str(review.get("posted_at") or ""),
        "text_raw": str(review.get("text_raw") or ""),
        "aspect_version": aspect_version,
        "provider": provider,
        "model_name": model_name,
        "absa_status": absa_status,
        "aspect_codes": aspect_codes,
        "sentiments": sentiments,
        "severities": severities,
        "evidence_quotes": evidence_quotes,
        "mention_count": len(aspect_codes),
    }


def ensure_collection(
    client: QdrantClient,
    name: str,
    *,
    vector_size: int,
    distance: rest.Distance = DEFAULT_DISTANCE,
    recreate: bool = False,
) -> bool:
    """Create the collection if needed; return True if a create happened.

    Idempotent: an existing collection with the same vector size and
    distance is left untouched. A mismatch raises so we never silently
    write into an incompatible collection (mixed dimensions kill
    cosine search at query time).
    """
    if vector_size <= 0:
        raise ValueError(f"vector_size must be positive, got {vector_size!r}")
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(size=vector_size, distance=distance),
        )
        return True
    info = client.get_collection(name)
    existing_size = _existing_vector_size(info)
    if existing_size is not None and existing_size != vector_size:
        raise RuntimeError(
            f"Qdrant collection {name!r} already exists with vector_size="
            f"{existing_size}, but the current embedding provider produces "
            f"{vector_size}-dim vectors. Either delete the collection, set "
            f"--recreate, or switch to a matching embedding model."
        )
    return False


def _existing_vector_size(info: Any) -> int | None:
    cfg = getattr(info, "config", None) or getattr(info, "result", None)
    if cfg is None:
        return None
    params = getattr(cfg, "params", None)
    if params is None:
        return None
    vectors = getattr(params, "vectors", None)
    if vectors is None:
        return None
    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors, dict):
        for v in vectors.values():
            inner = getattr(v, "size", None)
            if isinstance(inner, int):
                return inner
    return None


def upsert_points(
    client: QdrantClient,
    collection: str,
    points: Iterable[rest.PointStruct],
    *,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """Upsert ``points`` into ``collection`` in chunks. Returns count.

    Qdrant's ``upsert`` payload size is capped by the server config;
    chunking keeps requests bounded even for the M3A 1k batch.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total = 0
    bucket: list[rest.PointStruct] = []
    for point in points:
        bucket.append(point)
        if len(bucket) >= batch_size:
            client.upsert(collection_name=collection, points=bucket, wait=True)
            total += len(bucket)
            bucket = []
    if bucket:
        client.upsert(collection_name=collection, points=bucket, wait=True)
        total += len(bucket)
    return total
