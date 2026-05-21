"""Retrieval search page — a search demo over the Qdrant index.

This is **search only**: it ranks indexed reviews for a query and shows
them. It is deliberately *not* the final RAG layer — there is no LLM
answer generation here, and none is called.

``run_search`` is kept free of any ``streamlit`` calls and takes its
backends by injection, so it can be unit-tested with a mock /
in-memory Qdrant client.
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from voicelens.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from voicelens.retrieval.embeddings import EmbeddingProvider, get_embedding_provider
from voicelens.retrieval.filters import build_search_filter
from voicelens.retrieval.hybrid import hybrid_search, lexical_search
from voicelens.retrieval.lexical import BM25LexicalRetriever
from voicelens.retrieval.search import SearchHit, retrieve
from voicelens.ui import db
from voicelens.ui.components import empty_state, page_header

SEARCH_MODES = ("hybrid", "dense", "lexical")
DEFAULT_MODE = "hybrid"          # hybrid / rrf_equal — the M3C-tuned default
DEFAULT_FUSION = "rrf_equal"


@dataclass
class RetrievalBackends:
    """The three pieces ``run_search`` needs, bundled for injection."""

    client: object  # qdrant_client.QdrantClient (kept loose for test doubles)
    embedder: EmbeddingProvider
    bm25: BM25LexicalRetriever
    collection: str


def run_search(
    query: str,
    *,
    backends: RetrievalBackends,
    mode: str = DEFAULT_MODE,
    brand: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
    top_k: int = 10,
    fusion: str = DEFAULT_FUSION,
) -> list[SearchHit]:
    """Run one search and return ranked hits — no Streamlit, no LLM.

    ``mode`` is ``dense`` / ``lexical`` / ``hybrid``. An empty query
    short-circuits to ``[]`` so the page never crashes on first load.
    """
    if not query or not query.strip():
        return []
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")

    if mode == "lexical":
        return lexical_search(
            bm25=backends.bm25,
            query=query,
            limit=top_k,
            brand=brand,
            aspect=aspect,
            sentiment=sentiment,
            rating_min=rating_min,
            rating_max=rating_max,
        )

    dense_filter = build_search_filter(
        brand=brand,
        aspect=aspect,
        sentiment=sentiment,
        rating_min=rating_min,
        rating_max=rating_max,
    )
    if mode == "dense":
        return retrieve(
            client=backends.client,
            collection=backends.collection,
            embedder=backends.embedder,
            query=query,
            limit=top_k,
            filter_=dense_filter,
        )

    # hybrid
    return hybrid_search(
        client=backends.client,
        collection=backends.collection,
        embedder=backends.embedder,
        bm25=backends.bm25,
        query=query,
        limit=top_k,
        dense_filter=dense_filter,
        lexical_brand=brand,
        lexical_aspect=aspect,
        lexical_sentiment=sentiment,
        lexical_rating_min=rating_min,
        lexical_rating_max=rating_max,
        fusion=fusion,
    )


@st.cache_resource(show_spinner="Loading retrieval backends…")
def _build_backends() -> RetrievalBackends:
    """Construct the real Qdrant / embedding / BM25 backends once per session."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL)
    embedder = get_embedding_provider(EMBEDDING_PROVIDER, model=EMBEDDING_MODEL)
    bm25 = BM25LexicalRetriever.from_qdrant(client, QDRANT_COLLECTION)
    return RetrievalBackends(
        client=client, embedder=embedder, bm25=bm25, collection=QDRANT_COLLECTION
    )


def _hit_rows(hits: list[SearchHit]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, hit in enumerate(hits, start=1):
        rows.append(
            {
                "rank": rank,
                "score": round(hit.score, 4),
                "review_id": hit.review_id,
                "brand": hit.brand,
                "asin": hit.asin,
                "rating": hit.rating,
                "aspects": ", ".join(hit.aspect_codes),
                "sentiments": ", ".join(hit.sentiments),
                "evidence_quotes": " | ".join(q for q in hit.evidence_quotes if q),
                "snippet": hit.text_snippet(220),
            }
        )
    return rows


def render() -> None:
    page_header(
        "🔎 Retrieval Search",
        "Search-only demo over the Qdrant review index — no LLM answer "
        "generation (that is a later milestone).",
    )

    with st.sidebar:
        st.subheader("Search filters")
        mode = st.selectbox("Mode", SEARCH_MODES, index=SEARCH_MODES.index(DEFAULT_MODE))
        top_k = st.slider("Top K", min_value=1, max_value=25, value=10)
        brands = ["(any)", *db.list_brands()]
        aspects = ["(any)", *db.list_aspects()]
        brand = st.selectbox("Brand", brands)
        aspect = st.selectbox("Aspect", aspects)
        sentiment = st.selectbox("Sentiment", ["(any)", "negative", "neutral", "positive"])
        rating_min, rating_max = st.slider(
            "Rating range", min_value=1, max_value=5, value=(1, 5)
        )

    query = st.text_input(
        "Query", placeholder="e.g. cable stopped charging after a few weeks"
    )
    if not query.strip():
        empty_state("Enter a query above to search the indexed reviews.")
        return

    try:
        backends = _build_backends()
    except Exception as exc:  # Qdrant down / collection missing
        st.error(
            f"Could not reach the Qdrant index ({exc}). "
            "Start it with `make up` and index with `make embed-v2-1k`."
        )
        return

    try:
        hits = run_search(
            query,
            backends=backends,
            mode=mode,
            brand=None if brand == "(any)" else brand,
            aspect=None if aspect == "(any)" else aspect,
            sentiment=None if sentiment == "(any)" else sentiment,
            rating_min=rating_min,
            rating_max=rating_max,
            top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001 - surface any backend error in-page
        st.error(f"Search failed: {exc}")
        return

    st.caption(f"Mode `{mode}` · {len(hits)} hit(s)")
    if not hits:
        empty_state("No reviews matched this query and filter combination.")
        return

    st.dataframe(_hit_rows(hits), use_container_width=True, hide_index=True)
    for rank, hit in enumerate(hits, start=1):
        with st.expander(
            f"#{rank} · score={hit.score:.4f} · {hit.brand} {hit.asin} "
            f"· rating={hit.rating}"
        ):
            if hit.aspect_codes:
                st.write("**Aspects:** " + ", ".join(hit.aspect_codes))
            for quote in hit.evidence_quotes:
                if quote:
                    st.write(f"> {quote}")
            st.write(hit.text_snippet(600))
