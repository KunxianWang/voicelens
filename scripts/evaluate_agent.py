"""Evaluate the LangGraph agent workflow against a golden set (M6C).

Loads ``data/eval/agent_smoke_goldens.jsonl``, runs every question
through the agent, scores routing / citation / content expectations,
and writes:

- ``data/eval/agent_eval_results.csv``  — one row per question
- ``data/eval/agent_eval_summary.json`` — the aggregate metric set

Defaults to the offline ``mock`` answer provider. Needs Postgres +
Qdrant up (the retrieval / analytics / incident routes read live data).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from voicelens.agent import result_summary, run_agent
from voicelens.config import PROJECT_ROOT
from voicelens.eval.agent_eval import (
    AgentGolden,
    aggregate_metrics,
    load_goldens,
    score_record,
)

_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
GOLDENS_PATH = _EVAL_DIR / "agent_smoke_goldens.jsonl"
RESULTS_CSV = _EVAL_DIR / "agent_eval_results.csv"
SUMMARY_JSON = _EVAL_DIR / "agent_eval_summary.json"

_CSV_COLUMNS = [
    "question", "expected_route", "actual_route", "route_match",
    "n_citations", "citation_required", "citation_ok", "contains_ok",
    "answer_nonempty", "has_warning", "passed",
]

# A scorer-facing summary function: question -> result_summary dict.
SummaryFn = Callable[[str], dict[str, Any]]


def run_evaluation(
    goldens: list[AgentGolden],
    *,
    provider: str = "mock",
    model: str | None = None,
    summary_fn: SummaryFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run every golden through the agent and score it.

    ``summary_fn`` may be injected (tests pass canned summaries);
    otherwise each question is run through the real agent workflow.
    """
    if summary_fn is None:
        def summary_fn(question: str) -> dict[str, Any]:
            return result_summary(
                run_agent(question, provider=provider, model=model)
            )

    scored = [score_record(g, summary_fn(g.question)) for g in goldens]
    return scored, aggregate_metrics(scored)


def write_results(
    scored: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    csv_path: Path = RESULTS_CSV,
    json_path: Path = SUMMARY_JSON,
) -> None:
    """Write the per-question CSV and the aggregate-metrics JSON."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in scored:
            writer.writerow({k: row[k] for k in _CSV_COLUMNS})
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def _print_summary(metrics: dict[str, Any]) -> None:
    print("=== Agent workflow evaluation ===")
    print(
        f"questions            : {metrics['n_questions']} "
        f"(retrieval={metrics['n_retrieval']} analytics={metrics['n_analytics']} "
        f"incident={metrics['n_incident']} unsupported={metrics['n_unsupported']})"
    )
    print(f"route_accuracy       : {metrics['route_accuracy']}")
    print(f"retrieval_citation_rate : {metrics['retrieval_citation_rate']}")
    print(f"unsupported_refusal_rate: {metrics['unsupported_refusal_rate']}")
    print(f"analytics_nonempty_rate : {metrics['analytics_answer_nonempty_rate']}")
    print(f"incident_nonempty_rate  : {metrics['incident_answer_nonempty_rate']}")
    print(f"avg_citations_retrieval : {metrics['average_citations_for_retrieval']}")
    print(f"warning_rate         : {metrics['warning_rate']}")
    print(f"pass_rate            : {metrics['pass_rate']} "
          f"({metrics['n_passed']}/{metrics['n_questions']})")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the agent workflow.")
    parser.add_argument("--goldens", default=str(GOLDENS_PATH))
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    goldens = load_goldens(args.goldens)
    if not goldens:
        print(f"ERROR: no goldens loaded from {args.goldens}", file=sys.stderr)
        return 2
    scored, metrics = run_evaluation(
        goldens, provider=args.provider, model=args.model
    )
    write_results(scored, metrics)
    _print_summary(metrics)
    print(f"\nwrote {RESULTS_CSV}")
    print(f"wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
