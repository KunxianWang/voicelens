"""RAG answer-generation smoke test (Milestone 6A).

Runs five fixed VoC questions plus one nonsense query through the
retrieve-then-answer pipeline and checks the answer-generation
guardrails hold:

- a real question returns a non-empty, fully-grounded citation set,
- every cited review_id came from the retrieved results,
- the answer contains no unsupported citation markers,
- a nonsense query is reported as insufficient evidence.

Defaults to the offline ``mock`` provider so it can run in CI / without
an API key. Pass ``--provider anthropic`` (with a key) for a real run.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts only ``scripts/`` on sys.path; add the
# repo root so the ``scripts`` package (sibling modules) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

from scripts.ask_voicelens import retrieve_hits  # noqa: E402
from voicelens.rag.answer import generate_answer  # noqa: E402

# (question, aspect filter, sentiment filter)
SMOKE_QUESTIONS: tuple[tuple[str, str | None, str | None], ...] = (
    ("What are the main reliability complaints?", "reliability", "negative"),
    ("Are users complaining about Bluetooth disconnects?", "bluetooth", "negative"),
    ("What do users say about price/value?", "price", None),
    ("Are there charging failures?", "charging", "negative"),
    ("What delivery issues appear?", "delivery", "negative"),
)

# A query with no plausible support in the review corpus — must trip the
# insufficient-evidence guardrail rather than fabricate an answer.
NONSENSE_QUERY = "qwxyz blorptastic frobnicate zffflorp nonsense token"


def _check_question(
    question: str,
    aspect: str | None,
    sentiment: str | None,
    *,
    provider: str | None,
    model: str | None,
    top_k: int,
) -> tuple[bool, list[str]]:
    """Run one real question; return ``(passed, failure_messages)``."""
    hits = retrieve_hits(
        question, mode="hybrid", aspect=aspect, sentiment=sentiment, top_k=top_k
    )
    result = generate_answer(question, hits, provider=provider, model=model)
    retrieved_ids = {h.review_id for h in hits if h.review_id is not None}

    failures: list[str] = []
    if not result.citations:
        failures.append("no citations were produced")
    bad_ids = [
        c.review_id
        for c in result.citations
        if c.review_id is not None and c.review_id not in retrieved_ids
    ]
    if bad_ids:
        failures.append(f"citations reference non-retrieved review_ids {bad_ids}")
    unsupported = result.guardrail_flags.get("unsupported_citations") or []
    if unsupported:
        failures.append(f"answer cites unsupported ids {unsupported}")
    return (not failures), failures


def _check_nonsense(
    *, provider: str | None, model: str | None, top_k: int
) -> tuple[bool, list[str]]:
    """The nonsense query must come back as insufficient evidence."""
    hits = retrieve_hits(NONSENSE_QUERY, mode="hybrid", top_k=top_k)
    result = generate_answer(NONSENSE_QUERY, hits, provider=provider, model=model)
    if not result.insufficient_evidence:
        return False, ["nonsense query was NOT flagged insufficient_evidence"]
    return True, []


def run_smoke(
    *, provider: str | None = "mock", model: str | None = None, top_k: int = 8
) -> int:
    """Run the full smoke suite; return a process exit code (0 = pass)."""
    print(f"=== RAG answer-generation smoke (provider={provider or 'env/mock'}) ===\n")
    all_ok = True

    for question, aspect, sentiment in SMOKE_QUESTIONS:
        ok, failures = _check_question(
            question, aspect, sentiment,
            provider=provider, model=model, top_k=top_k,
        )
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {question}")
        for msg in failures:
            print(f"        - {msg}")
        all_ok = all_ok and ok

    ok, failures = _check_nonsense(provider=provider, model=model, top_k=top_k)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] (nonsense query -> insufficient evidence)")
    for msg in failures:
        print(f"        - {msg}")
    all_ok = all_ok and ok

    print(f"\n--- {'ALL SMOKE CHECKS PASSED' if all_ok else 'SMOKE CHECKS FAILED'} ---")
    return 0 if all_ok else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG answer-generation smoke test.")
    parser.add_argument(
        "--provider", default="mock",
        help="mock | openai | anthropic (default: mock — no API key needed)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run_smoke(provider=args.provider, model=args.model, top_k=args.top_k)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
