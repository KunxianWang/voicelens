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
import sys

from voicelens.rag.answer import AnswerResult, generate_answer
from voicelens.rag.retrieve import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_PROVIDER,
    SEARCH_MODES,
    retrieve_reviews,
)

# Backwards-compatible alias: the retrieval helper now lives in
# ``voicelens.rag.retrieve`` (shared with the M6B agent).
retrieve_hits = retrieve_reviews


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
