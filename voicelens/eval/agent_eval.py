"""Agent-workflow evaluation metrics (Milestone 6C).

Pure functions: a golden row plus the agent's ``result_summary`` dict
in, a per-question score dict or an aggregate-metrics dict out. No DB,
no Qdrant, no LLM — the runner in ``scripts/evaluate_agent.py`` is
responsible for producing the summaries this module scores.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicelens.agent.router import (
    ROUTE_ANALYTICS,
    ROUTE_INCIDENT,
    ROUTE_INSUFFICIENT,
    ROUTE_RETRIEVAL,
)


@dataclass
class AgentGolden:
    """One expected-behaviour row from ``agent_smoke_goldens.jsonl``."""

    question: str
    expected_route: str
    expected_contains: list[str] = field(default_factory=list)
    expected_citation_required: bool = False
    expected_min_citations: int = 1
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> AgentGolden:
        return cls(
            question=str(row["question"]),
            expected_route=str(row["expected_route"]),
            expected_contains=list(row.get("expected_contains") or []),
            expected_citation_required=bool(row.get("expected_citation_required")),
            expected_min_citations=int(row.get("expected_min_citations") or 1),
            notes=str(row.get("notes") or ""),
        )


def load_goldens(path: str | Path) -> list[AgentGolden]:
    """Load the agent golden set from a JSONL file."""
    goldens: list[AgentGolden] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                goldens.append(AgentGolden.from_dict(json.loads(line)))
    return goldens


def score_record(golden: AgentGolden, summary: dict[str, Any]) -> dict[str, Any]:
    """Score one agent run against its golden row.

    ``summary`` is a :func:`voicelens.agent.state.result_summary` dict.
    """
    actual_route = str(summary.get("route") or "")
    answer = str(summary.get("answer") or "")
    citations = summary.get("citations") or []
    warnings = summary.get("warnings") or []
    n_citations = len(citations)

    route_match = actual_route == golden.expected_route
    answer_nonempty = bool(answer.strip())

    if golden.expected_citation_required:
        citation_ok = n_citations >= golden.expected_min_citations
    else:
        citation_ok = True

    contains_ok = all(
        token.lower() in answer.lower() for token in golden.expected_contains
    )

    passed = route_match and citation_ok and contains_ok and answer_nonempty
    return {
        "question": golden.question,
        "expected_route": golden.expected_route,
        "actual_route": actual_route,
        "route_match": route_match,
        "n_citations": n_citations,
        "citation_required": golden.expected_citation_required,
        "citation_ok": citation_ok,
        "contains_ok": contains_ok,
        "answer_nonempty": answer_nonempty,
        "has_warning": bool(warnings),
        "passed": passed,
    }


def _rate(numerator: int, denominator: int) -> float:
    """Safe ratio rounded to 4dp; 0.0 when the denominator is 0."""
    return round(numerator / denominator, 4) if denominator else 0.0


def aggregate_metrics(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question score dicts into the M6C metric set."""
    total = len(scored)
    retrieval = [s for s in scored if s["expected_route"] == ROUTE_RETRIEVAL]
    analytics = [s for s in scored if s["expected_route"] == ROUTE_ANALYTICS]
    incident = [s for s in scored if s["expected_route"] == ROUTE_INCIDENT]
    unsupported = [s for s in scored if s["expected_route"] == ROUTE_INSUFFICIENT]

    route_correct = sum(1 for s in scored if s["route_match"])
    retrieval_with_citations = sum(
        1 for s in retrieval if s["n_citations"] > 0
    )
    unsupported_refused = sum(
        1 for s in unsupported if s["actual_route"] == ROUTE_INSUFFICIENT
    )
    analytics_nonempty = sum(1 for s in analytics if s["answer_nonempty"])
    incident_nonempty = sum(1 for s in incident if s["answer_nonempty"])
    total_retrieval_citations = sum(s["n_citations"] for s in retrieval)
    with_warnings = sum(1 for s in scored if s["has_warning"])
    passed = sum(1 for s in scored if s["passed"])

    return {
        "n_questions": total,
        "n_retrieval": len(retrieval),
        "n_analytics": len(analytics),
        "n_incident": len(incident),
        "n_unsupported": len(unsupported),
        "route_accuracy": _rate(route_correct, total),
        "retrieval_citation_rate": _rate(retrieval_with_citations, len(retrieval)),
        "unsupported_refusal_rate": _rate(unsupported_refused, len(unsupported)),
        "analytics_answer_nonempty_rate": _rate(analytics_nonempty, len(analytics)),
        "incident_answer_nonempty_rate": _rate(incident_nonempty, len(incident)),
        "average_citations_for_retrieval": _rate(
            total_retrieval_citations, len(retrieval)
        ),
        "warning_rate": _rate(with_warnings, total),
        "pass_rate": _rate(passed, total),
        "n_passed": passed,
    }
