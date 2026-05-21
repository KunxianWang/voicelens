"""Agent workflow smoke test (Milestone 6B).

Runs ~10 fixed questions through the LangGraph agent and checks the
routing + per-route behaviour:

- the router picks the expected route,
- every route produces a non-empty answer,
- the retrieval route attaches citations,
- the analytics route returns numeric / statistical content,
- the incident route mentions incident / anomaly fields,
- the unsupported question is routed to insufficient scope and does
  not hallucinate an answer.

Defaults to the offline ``mock`` answer provider so it runs without an
API key. Needs Postgres + Qdrant up (retrieval / analytics / incident
routes read live data).
"""
from __future__ import annotations

import argparse
import sys

from voicelens.agent import (
    ROUTE_ANALYTICS,
    ROUTE_INCIDENT,
    ROUTE_INSUFFICIENT,
    ROUTE_RETRIEVAL,
    result_summary,
    run_agent,
)

# (question, expected_route)
SMOKE_QUESTIONS: tuple[tuple[str, str], ...] = (
    # --- retrieval (4) ---
    ("What are the main reliability complaints?", ROUTE_RETRIEVAL),
    ("What are customers saying about bluetooth?", ROUTE_RETRIEVAL),
    ("Show me example reviews about charging failures", ROUTE_RETRIEVAL),
    ("What complaints do users have about price?", ROUTE_RETRIEVAL),
    # --- analytics (3) ---
    ("Which aspect has the most negative mentions?", ROUTE_ANALYTICS),
    ("What is the sentiment distribution across reviews?", ROUTE_ANALYTICS),
    ("Show the severity breakdown of negative mentions", ROUTE_ANALYTICS),
    # --- incident (2) ---
    ("Which issues spiked recently?", ROUTE_INCIDENT),
    ("Are there any emerging anomalies?", ROUTE_INCIDENT),
    # --- unsupported (1) ---
    ("What is the capital of France?", ROUTE_INSUFFICIENT),
)

_INCIDENT_FIELD_HINTS = ("incident", "spiked", "baseline", "z=", "severity")


def _check(summary: dict, expected_route: str) -> list[str]:
    """Return a list of failure messages for one agent result ([] = pass)."""
    failures: list[str] = []
    route = summary["route"]
    answer = summary["answer"] or ""

    if route != expected_route:
        failures.append(f"route {route!r}, expected {expected_route!r}")
    if not answer.strip():
        failures.append("empty answer")

    if expected_route == ROUTE_RETRIEVAL:
        if not summary["citations"]:
            failures.append("retrieval route returned no citations")
    elif expected_route == ROUTE_ANALYTICS:
        if not any(ch.isdigit() for ch in answer):
            failures.append("analytics route answer has no numeric content")
    elif expected_route == ROUTE_INCIDENT:
        low = answer.lower()
        if not any(h in low for h in _INCIDENT_FIELD_HINTS):
            failures.append("incident route answer mentions no incident fields")
    elif expected_route == ROUTE_INSUFFICIENT:
        if summary["citations"]:
            failures.append("insufficient route should not produce citations")
        if "can't answer" not in answer.lower():
            failures.append("insufficient route did not return the safe message")
    return failures


def run_smoke(*, provider: str = "mock", model: str | None = None) -> int:
    """Run the smoke suite; return a process exit code (0 = all passed)."""
    print(f"=== Agent workflow smoke (provider={provider}) ===\n")
    all_ok = True
    for question, expected_route in SMOKE_QUESTIONS:
        state = run_agent(question, provider=provider, model=model)
        summary = result_summary(state)
        failures = _check(summary, expected_route)
        mark = "PASS" if not failures else "FAIL"
        print(f"[{mark}] ({summary['route']:>18}) {question}")
        for msg in failures:
            print(f"        - {msg}")
        all_ok = all_ok and not failures

    print(f"\n--- {'ALL AGENT SMOKE CHECKS PASSED' if all_ok else 'SMOKE CHECKS FAILED'} ---")
    return 0 if all_ok else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent workflow smoke test.")
    parser.add_argument(
        "--provider", default="mock",
        help="mock | openai | anthropic (default: mock — no API key needed)",
    )
    parser.add_argument("--model", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run_smoke(provider=args.provider, model=args.model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
