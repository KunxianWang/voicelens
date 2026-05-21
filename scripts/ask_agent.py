"""Ask the VoiceLens agent a question (M6B LangGraph workflow).

The agent routes the question to one of four tools — retrieval answer,
incident summary, analytics summary, or insufficient scope — and
returns the result. It is a single deterministic routing pass, not an
autonomous agent: no memory, no actions, no multi-step planning.

Examples::

    python scripts/ask_agent.py --question "What are the main reliability complaints?"
    python scripts/ask_agent.py --question "Which issues spiked recently?"
    python scripts/ask_agent.py --question "Which aspect has the most negative mentions?"

``--provider mock`` (default) needs no API key; ``anthropic`` / ``openai``
need the matching key in the environment.
"""
from __future__ import annotations

import argparse
import json
import sys

from voicelens.agent import result_summary, run_agent
from voicelens.agent.prompts import ROUTE_DESCRIPTIONS


def _print_state(state) -> None:
    summary = result_summary(state)
    route = summary["route"]
    print("=" * 70)
    print(f"Q: {summary['question']}")
    print("=" * 70)
    print(f"\nROUTE : {route}  —  {ROUTE_DESCRIPTIONS.get(route, '')}")
    if summary["filters"]:
        print(f"FILTERS: {json.dumps(summary['filters'], sort_keys=True)}")

    print("\nANSWER")
    print("-" * 70)
    print(summary["answer"])

    citations = summary["citations"]
    if citations:
        print("\nCITATIONS")
        print("-" * 70)
        for c in citations:
            rating = c["rating"] if c["rating"] is not None else "-"
            print(
                f"[{c['citation_id']}] review_id={c['review_id']} "
                f"{c['brand']} {c['asin']} rating={rating} "
                f"score={c['retrieval_score']}"
            )
            if c["evidence_quote"]:
                print(f"     quote: {c['evidence_quote']}")

    if summary["retrieved_review_ids"]:
        print(f"\nretrieved review IDs: {summary['retrieved_review_ids']}")

    warnings = summary["warnings"]
    print("\nWARNINGS")
    print("-" * 70)
    print("\n".join(f"- {w}" for w in warnings) if warnings else "(none)")
    print(f"\nprovider={summary['provider']} model={summary['model_name'] or '-'}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the VoiceLens agent (route -> tool -> answer)."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--provider", default="mock",
        help="mock | openai | anthropic (default: mock — no API key needed)",
    )
    parser.add_argument("--model", default=None, help="override RAG_MODEL")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    state = run_agent(
        args.question, provider=args.provider, model=args.model, top_k=args.top_k
    )
    if args.json:
        print(json.dumps(result_summary(state), indent=2, ensure_ascii=False))
    else:
        _print_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
