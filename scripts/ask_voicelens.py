"""Ask VoiceLens a question and get a citation-grounded answer (M6A).

Runs the existing hybrid retrieval layer, then the M6A answer
generator. This is *not* an agent — there is no planner, no tool
routing, no memory. It is a single retrieve-then-answer pass.

Example::

    python scripts/ask_voicelens.py \
      --question "What are customers saying about products that stopped working?" \
      --mode hybrid --aspect reliability --sentiment negative \
      --top-k 8 --provider mock

Use ``--provider mock`` for offline runs; ``--provider anthropic`` /
``openai`` need the matching API key in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from qdrant_client import QdrantClient

from voicelens.config import QDRANT_COLLECTION, QDRANT_URL
from voicelens.rag.answer import AnswerResult, generate_answer
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


def _open_client(url: str) -> QdrantClient:
    target = (url or QDRANT_URL).strip()
    if target in ("", ":memory:", "memory://"):
        return QdrantClient(":memory:")
    return QdrantClient(url=target)


def retrieve_hits(
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

    Streamlit-free so the CLI and the RAG smoke script can both call it.
    A pre-built ``client`` may be injected (used by tests with an
    in-memory Qdrant); otherwise one is opened from ``qdrant_url``.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")

    collection = collection or QDRANT_COLLECTION
    if client is None:
        client = _open_client(qdrant_url or QDRANT_URL)
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


def _print_result(result: AnswerResult, *, mode: str, top_k: int) -> None:
    print("=" * 70)
    print(f"Q: {result.question}")
    print("=" * 70)
    print("\nANSWER")
    print("-" * 70)
    print(result.answer)
    if result.insufficient_evidence:
        print("\n[insufficient evidence — answer not grounded in retrieved reviews]")

    print("\nCITATIONS")
    print("-" * 70)
    if not result.citations:
        print("(none)")
    for c in result.citations:
        rating = c.rating if c.rating is not None else "-"
        print(
            f"[{c.citation_id}] review_id={c.review_id} {c.brand} {c.asin} "
            f"rating={rating} score={c.retrieval_score:.4f}"
        )
        if c.aspect_codes:
            print(f"     aspects: {', '.join(c.aspect_codes)}")
        if c.evidence_quote:
            print(f"     quote  : {c.evidence_quote}")

    print("\nRETRIEVAL SUMMARY")
    print("-" * 70)
    print(
        f"mode={mode} top_k={top_k} retrieved={len(result.retrieved_review_ids)} "
        f"provider={result.provider} model={result.model}"
    )
    if result.guardrail_flags:
        print(f"guardrails: {json.dumps(result.guardrail_flags, sort_keys=True)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask VoiceLens a question (retrieve + cite-grounded answer)."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hybrid")
    parser.add_argument("--brand", default=None)
    parser.add_argument("--aspect", default=None)
    parser.add_argument("--sentiment", default=None)
    parser.add_argument("--rating-min", type=int, default=None)
    parser.add_argument("--rating-max", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--provider", default=None,
        help="mock | openai | anthropic (default: RAG_PROVIDER env, else mock)",
    )
    parser.add_argument("--model", default=None, help="override RAG_MODEL")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument(
        "--embedding-provider", default=DEFAULT_EMBED_PROVIDER,
        help=f"query embedder (default: {DEFAULT_EMBED_PROVIDER})",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        hits = retrieve_hits(
            args.question, mode=args.mode, collection=args.collection,
            qdrant_url=args.qdrant_url,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            brand=args.brand, aspect=args.aspect,
            sentiment=args.sentiment, rating_min=args.rating_min,
            rating_max=args.rating_max, top_k=args.top_k,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = generate_answer(
        args.question, hits, provider=args.provider, model=args.model,
        filters={
            "brand": args.brand, "aspect": args.aspect,
            "sentiment": args.sentiment, "rating_min": args.rating_min,
            "rating_max": args.rating_max, "mode": args.mode,
        },
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        _print_result(result, mode=args.mode, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
