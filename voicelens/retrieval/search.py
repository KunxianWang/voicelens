"""Query-time retrieval helper.

A thin wrapper around ``QdrantClient.search`` that:
- embeds the natural-language query with the same embedding provider
  the indexing flow used,
- attaches an optional filter (see :mod:`voicelens.retrieval.filters`),
- normalises the response into a :class:`SearchHit` dataclass so the
  CLI smoke script and tests don't depend on Qdrant's internal types.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from voicelens.retrieval.embeddings import EmbeddingProvider


@dataclass
class SearchHit:
    score: float
    review_id: int | None
    brand: str
    asin: str
    rating: int | None
    aspect_codes: list[str] = field(default_factory=list)
    sentiments: list[str] = field(default_factory=list)
    severities: list[str | None] = field(default_factory=list)
    evidence_quotes: list[str] = field(default_factory=list)
    text_raw: str = ""
    posted_at: str = ""
    absa_status: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def text_snippet(self, max_chars: int = 200) -> str:
        text = (self.text_raw or "").strip().replace("\n", " ")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"


def retrieve(
    *,
    client: QdrantClient,
    collection: str,
    embedder: EmbeddingProvider,
    query: str,
    limit: int = 5,
    filter_: rest.Filter | None = None,
) -> list[SearchHit]:
    """Embed ``query`` and search ``collection`` returning ``limit`` hits."""
    if not query or not query.strip():
        return []
    [vector] = embedder.embed_batch([query])
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        query_filter=filter_,
        with_payload=True,
    )
    points = getattr(response, "points", response)
    hits: list[SearchHit] = []
    for hit in points:
        payload = dict(getattr(hit, "payload", None) or {})
        hits.append(
            SearchHit(
                score=float(getattr(hit, "score", 0.0) or 0.0),
                review_id=_coerce_int(payload.get("review_id")),
                brand=str(payload.get("brand") or ""),
                asin=str(payload.get("asin") or ""),
                rating=_coerce_int(payload.get("rating")),
                aspect_codes=list(_coerce_str_list(payload.get("aspect_codes"))),
                sentiments=list(_coerce_str_list(payload.get("sentiments"))),
                severities=list(_coerce_optional_str_list(payload.get("severities"))),
                evidence_quotes=list(_coerce_str_list(payload.get("evidence_quotes"))),
                text_raw=str(payload.get("text_raw") or ""),
                posted_at=str(payload.get("posted_at") or ""),
                absa_status=str(payload.get("absa_status") or ""),
                raw_payload=payload,
            )
        )
    return hits


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_list(value: Any) -> Sequence[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v is not None]


def _coerce_optional_str_list(value: Any) -> Sequence[str | None]:
    if not isinstance(value, list):
        return []
    return [str(v) if isinstance(v, str) else None for v in value]
