"""Quick retrieval smoke against the Qdrant collection.

Embeds a natural-language query with the same embedding provider used
for indexing, optionally narrows with payload filters, and prints the
top hits. Not a substitute for the real RAG layer (still scoped to a
later milestone) — this script just confirms the index is queryable.
"""
from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient

from voicelens.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from voicelens.retrieval.embeddings import get_embedding_provider
from voicelens.retrieval.filters import build_search_filter
from voicelens.retrieval.search import SearchHit, retrieve


def _open_client(url: str) -> QdrantClient:
    target = (url or QDRANT_URL).strip()
    if target in ("", ":memory:", "memory://"):
        return QdrantClient(":memory:")
    return QdrantClient(url=target)


def _print_hits(hits: list[SearchHit]) -> None:
    if not hits:
        print("(no hits)")
        return
    for i, hit in enumerate(hits, start=1):
        rating = hit.rating if hit.rating is not None else "-"
        print(
            f"#{i} score={hit.score:.4f} review_id={hit.review_id} "
            f"brand={hit.brand!r} asin={hit.asin!r} rating={rating} "
            f"status={hit.absa_status}"
        )
        if hit.aspect_codes:
            aspects = []
            for code, sent, sev in zip(
                hit.aspect_codes,
                hit.sentiments + [""] * len(hit.aspect_codes),
                hit.severities + [None] * len(hit.aspect_codes),
                strict=False,
            ):
                tag = f"{code} {sent or ''}".strip()
                if sev:
                    tag += f" {sev}"
                aspects.append(tag)
            print(f"     aspects : {', '.join(aspects)}")
        snippet = hit.text_snippet(220)
        if snippet:
            print(f"     snippet : {snippet}")
        for quote in (hit.evidence_quotes or [])[:2]:
            if quote:
                print(f"     quote   : {quote}")
        print()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval smoke against Qdrant.")
    parser.add_argument("--query", required=True, help="Natural-language query")
    parser.add_argument("--collection", default=None, help=f"Default: {QDRANT_COLLECTION}")
    parser.add_argument("--qdrant-url", default=None, help=f"Default: {QDRANT_URL}")
    parser.add_argument("--embedding-provider", default=None, help=f"Default: {EMBEDDING_PROVIDER}")
    parser.add_argument("--embedding-model", default=None, help=f"Default: {EMBEDDING_MODEL}")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--brand", default=None)
    parser.add_argument("--asin", default=None)
    parser.add_argument("--aspect", default=None)
    parser.add_argument("--sentiment", default=None)
    parser.add_argument("--rating-min", type=int, default=None)
    parser.add_argument("--rating-max", type=int, default=None)
    parser.add_argument("--aspect-version", default=None)
    parser.add_argument("--absa-provider", default=None)
    parser.add_argument("--absa-model", default=None)
    parser.add_argument("--absa-status", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    embedder = get_embedding_provider(
        args.embedding_provider or EMBEDDING_PROVIDER,
        model=args.embedding_model or EMBEDDING_MODEL,
    )
    client = _open_client(args.qdrant_url or QDRANT_URL)
    collection = args.collection or QDRANT_COLLECTION
    if not client.collection_exists(collection):
        print(
            f"ERROR: Qdrant collection {collection!r} does not exist. "
            "Run `make embed-v2-1k` (or `python -m voicelens.pipeline.flows.embed_flow`) first.",
            file=sys.stderr,
        )
        return 2
    filter_ = build_search_filter(
        brand=args.brand,
        asin=args.asin,
        aspect=args.aspect,
        sentiment=args.sentiment,
        rating_min=args.rating_min,
        rating_max=args.rating_max,
        aspect_version=args.aspect_version,
        provider=args.absa_provider,
        model_name=args.absa_model,
        absa_status=args.absa_status,
    )
    hits = retrieve(
        client=client,
        collection=collection,
        embedder=embedder,
        query=args.query,
        limit=args.limit,
        filter_=filter_,
    )
    print(
        f"== retrieval_smoke: collection={collection!r} embedder={embedder.name!r} "
        f"limit={args.limit} query={args.query!r} =="
    )
    if filter_ is not None:
        print(f"   filter: {filter_.model_dump(exclude_none=True)}")
    print()
    _print_hits(hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
