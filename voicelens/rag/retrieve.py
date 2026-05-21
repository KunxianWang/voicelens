"""Retrieval helper shared by the RAG CLI and the M6B agent.

A single ``streamlit``-free entry point that wires the embedder, the
Qdrant client and the BM25 retriever, then runs dense / lexical /
hybrid search. Both ``scripts/ask_voicelens.py`` and
``voicelens/agent`` call this so the retrieval path is defined once.
"""
from __future__ import annotations

import os

from qdrant_client import QdrantClient

from voicelens.config import QDRANT_COLLECTION, QDRANT_URL
from voicelens.retrieval.embeddings import get_embedding_provider
from voicelens.retrieval.filters import build_search_filter
from voicelens.retrieval.hybrid import hybrid_search, lexical_search
from voicelens.retrieval.lexical import BM25LexicalRetriever
from voicelens.retrieval.search import SearchHit, retrieve

SEARCH_MODES = ("hybrid", "dense", "lexical")

# The working ``reviews_v2`` collection was indexed with the local
# bge-small embedder (384-dim), so retrieval must embed queries the same
# way. ``config.EMBEDDING_PROVIDER`` defaults to ``mock`` (32-dim) for
# the test path — wrong here — so resolve to ``local`` unless the env
# explicitly overrides it.
DEFAULT_EMBED_PROVIDER = os.getenv("EMBEDDING_PROVIDER") or "local"
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"


def open_client(url: str | None) -> QdrantClient:
    """Open a Qdrant client; ``:memory:`` / blank yields an in-memory one."""
    target = (url or QDRANT_URL).strip()
    if target in ("", ":memory:", "memory://"):
        return QdrantClient(":memory:")
    return QdrantClient(url=target)


def retrieve_reviews(
    question: str,
    *,
    mode: str = "hybrid",
    collection: str | None = None,
    qdrant_url: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    brand: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
    top_k: int = 8,
    client: QdrantClient | None = None,
) -> list[SearchHit]:
    """Retrieve reviews for ``question`` with the chosen mode + filters.

    A pre-built ``client`` may be injected (used by tests with an
    in-memory Qdrant); otherwise one is opened from ``qdrant_url``.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")

    collection = collection or QDRANT_COLLECTION
    if client is None:
        client = open_client(qdrant_url or QDRANT_URL)
    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Qdrant collection {collection!r} does not exist. "
            "Run `make embed-v2-1k` first."
        )
    embedder = get_embedding_provider(
        embedding_provider or DEFAULT_EMBED_PROVIDER,
        model=embedding_model or DEFAULT_EMBED_MODEL,
    )
    bm25 = BM25LexicalRetriever.from_qdrant(client, collection)

    if mode == "lexical":
        return lexical_search(
            bm25=bm25, query=question, limit=top_k, brand=brand, aspect=aspect,
            sentiment=sentiment, rating_min=rating_min, rating_max=rating_max,
        )
    dense_filter = build_search_filter(
        brand=brand, aspect=aspect, sentiment=sentiment,
        rating_min=rating_min, rating_max=rating_max,
    )
    if mode == "dense":
        return retrieve(
            client=client, collection=collection, embedder=embedder,
            query=question, limit=top_k, filter_=dense_filter,
        )
    return hybrid_search(
        client=client, collection=collection, embedder=embedder, bm25=bm25,
        query=question, limit=top_k, dense_filter=dense_filter,
        lexical_brand=brand, lexical_aspect=aspect, lexical_sentiment=sentiment,
        lexical_rating_min=rating_min, lexical_rating_max=rating_max,
        fusion="rrf_equal",
    )
