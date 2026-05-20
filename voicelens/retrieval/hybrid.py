"""Dense + lexical hybrid retrieval via Reciprocal Rank Fusion.

The hybrid retriever asks both backends for their top-``oversample_k``
hits, fuses them with RRF (default constant ``k=60``), and trims to
``limit``. Each fused hit is returned as a :class:`SearchHit` so the
smoke CLI + evaluator consume identical types regardless of mode.

Filters are enforced consistently: dense uses Qdrant's payload filter,
lexical uses :func:`voicelens.retrieval.lexical.filter_payloads` on
the in-memory corpus. The fusion only sees post-filter rankings, so
the result respects every predicate every retriever respected.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from voicelens.eval.retrieval_eval import (
    priority_fill_fusion,
    reciprocal_rank_fusion,
)
from voicelens.retrieval.embeddings import EmbeddingProvider
from voicelens.retrieval.lexical import BM25LexicalRetriever, filter_payloads
from voicelens.retrieval.search import SearchHit, retrieve

# Supported fusion strategies for :func:`hybrid_search`.
#   rrf_equal     - classic Reciprocal Rank Fusion, both rankers equal.
#   rrf_weighted  - RRF with per-ranker weights (favour the stronger one).
#   lexical_first - trust the lexical ranking, backfill with dense-only docs.
HYBRID_FUSIONS = ("rrf_equal", "rrf_weighted", "lexical_first")


def _hit_from_payload(score: float, payload: dict[str, Any]) -> SearchHit:
    return SearchHit(
        score=float(score),
        review_id=int(payload["review_id"]) if payload.get("review_id") is not None else None,
        brand=str(payload.get("brand") or ""),
        asin=str(payload.get("asin") or ""),
        rating=int(payload["rating"]) if payload.get("rating") is not None else None,
        aspect_codes=list(payload.get("aspect_codes") or []),
        sentiments=list(payload.get("sentiments") or []),
        severities=list(payload.get("severities") or []),
        evidence_quotes=list(payload.get("evidence_quotes") or []),
        text_raw=str(payload.get("text_raw") or ""),
        posted_at=str(payload.get("posted_at") or ""),
        absa_status=str(payload.get("absa_status") or ""),
        raw_payload=payload,
    )


def lexical_search(
    *,
    bm25: BM25LexicalRetriever,
    query: str,
    limit: int,
    brand: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
    oversample: int = 5,
) -> list[SearchHit]:
    """Rank the corpus with BM25, then post-filter and trim to ``limit``.

    Oversampling matters because the filter prunes after ranking — if
    we asked BM25 for only ``limit`` hits, a strict filter could leave
    us with too few results.
    """
    raw = bm25.rank(query, limit=max(limit * oversample, limit))
    if brand or aspect or sentiment or rating_min is not None or rating_max is not None:
        kept_payloads = filter_payloads(
            (hit.payload for hit in raw),
            brand=brand,
            aspect=aspect,
            sentiment=sentiment,
            rating_min=rating_min,
            rating_max=rating_max,
        )
        kept_ids = {int(p["review_id"]) for p in kept_payloads if p.get("review_id") is not None}
        raw = [h for h in raw if h.review_id in kept_ids]
    return [_hit_from_payload(h.score, h.payload) for h in raw[:limit]]


def hybrid_search(
    *,
    client: QdrantClient,
    collection: str,
    embedder: EmbeddingProvider,
    bm25: BM25LexicalRetriever,
    query: str,
    limit: int = 10,
    dense_filter: rest.Filter | None = None,
    lexical_brand: str | None = None,
    lexical_aspect: str | None = None,
    lexical_sentiment: str | None = None,
    lexical_rating_min: int | None = None,
    lexical_rating_max: int | None = None,
    oversample_k: int = 50,
    rrf_constant: int = 60,
    fusion: str = "rrf_equal",
    lexical_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[SearchHit]:
    """Run dense + lexical, fuse the rankings, return ``limit`` SearchHits.

    Both retrievers see the same ``query`` and the same filter
    predicates. The dense side accepts a pre-built ``rest.Filter`` so
    callers can pass whatever :func:`voicelens.retrieval.filters.build_search_filter`
    produced; the lexical side accepts the same predicates as keyword
    args because BM25 doesn't speak Qdrant filters natively.

    ``fusion`` selects how the two rankings combine (see
    :data:`HYBRID_FUSIONS`). M3B measured pure RRF underperforming the
    lexical baseline because the weaker dense ranking diluted it;
    ``rrf_weighted`` and ``lexical_first`` exist to correct that.
    """
    if fusion not in HYBRID_FUSIONS:
        raise ValueError(f"fusion must be one of {HYBRID_FUSIONS}, got {fusion!r}")
    dense_hits = retrieve(
        client=client,
        collection=collection,
        embedder=embedder,
        query=query,
        limit=oversample_k,
        filter_=dense_filter,
    )
    lexical_hits = lexical_search(
        bm25=bm25,
        query=query,
        limit=oversample_k,
        brand=lexical_brand,
        aspect=lexical_aspect,
        sentiment=lexical_sentiment,
        rating_min=lexical_rating_min,
        rating_max=lexical_rating_max,
    )

    dense_ranking = [h.review_id for h in dense_hits if h.review_id is not None]
    lexical_ranking = [h.review_id for h in lexical_hits if h.review_id is not None]

    if fusion == "lexical_first":
        fused_ids = priority_fill_fusion(
            lexical_ranking, dense_ranking, top_k=limit
        )
    else:
        # dense ranking is index 0, lexical is index 1 — keep weights aligned.
        weights = (
            [dense_weight, lexical_weight] if fusion == "rrf_weighted" else None
        )
        fused_ids = reciprocal_rank_fusion(
            [dense_ranking, lexical_ranking],
            k_constant=rrf_constant,
            top_k=limit,
            weights=weights,
        )
    payloads: dict[int, dict[str, Any]] = {}
    for hit in dense_hits:
        if hit.review_id is not None and hit.review_id not in payloads:
            payloads[hit.review_id] = hit.raw_payload
    for hit in lexical_hits:
        if hit.review_id is not None and hit.review_id not in payloads:
            payloads[hit.review_id] = hit.raw_payload

    fused_hits: list[SearchHit] = []
    for rank, rid in enumerate(fused_ids, start=1):
        payload = payloads.get(rid)
        if payload is None:
            continue
        # Fused score is a stand-in for downstream consumers — the
        # exact RRF score is meaningful only relative to the same query.
        fused_score = 1.0 / rank
        fused_hits.append(_hit_from_payload(fused_score, payload))
    return fused_hits
