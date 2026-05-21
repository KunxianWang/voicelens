"""Graph state for the M6B agent workflow.

The state is a plain ``TypedDict`` so LangGraph can merge node updates
by key. Every field is optional (``total=False``) — a node only writes
the keys it owns, and the router/tool that runs decides which of
``retrieved_results`` / ``analytics_result`` / ``incidents_result`` is
populated.
"""
from __future__ import annotations

from typing import Any, TypedDict

from voicelens.rag.citations import Citation
from voicelens.retrieval.search import SearchHit


class AgentState(TypedDict, total=False):
    """The single state object threaded through the LangGraph workflow."""

    question: str
    route: str
    filters: dict[str, Any]
    retrieved_results: list[SearchHit]
    retrieved_review_ids: list[int]
    analytics_result: dict[str, Any]
    incidents_result: list[dict[str, Any]]
    answer: str
    citations: list[Citation]
    warnings: list[str]
    provider: str
    model_name: str


def new_state(
    question: str, *, provider: str = "mock", model: str | None = None
) -> AgentState:
    """Build the initial state for one agent run."""
    return AgentState(
        question=(question or "").strip(),
        route="",
        filters={},
        warnings=[],
        provider=provider,
        model_name=model or "",
    )


def result_summary(state: AgentState) -> dict[str, Any]:
    """JSON-friendly view of a finished run — for the CLI and dashboard."""
    return {
        "question": state.get("question", ""),
        "route": state.get("route", ""),
        "filters": state.get("filters", {}),
        "answer": state.get("answer", ""),
        "warnings": list(state.get("warnings", []) or []),
        "provider": state.get("provider", ""),
        "model_name": state.get("model_name", ""),
        "retrieved_review_ids": list(state.get("retrieved_review_ids", []) or []),
        "citations": [c.as_dict() for c in state.get("citations", []) or []],
        "analytics_result": state.get("analytics_result", {}),
        "incidents_result": state.get("incidents_result", []),
    }
