"""Lightweight LangGraph router/workflow over the VoiceLens analytics.

Milestone 6B: a *controlled* agentic-RAG workflow — a deterministic
keyword router picks one of four routes (retrieval answer, incident
summary, analytics summary, or insufficient scope), the matching tool
runs, and the response is formatted. One pass, one tool.

It is deliberately not an autonomous agent: no memory, no autonomous
actions, no multi-agent collaboration, and no multi-step planning
beyond the single routing decision.
"""
from voicelens.agent.graph import build_agent_graph, run_agent
from voicelens.agent.router import (
    ALL_ROUTES,
    ROUTE_ANALYTICS,
    ROUTE_INCIDENT,
    ROUTE_INSUFFICIENT,
    ROUTE_RETRIEVAL,
    classify_route,
    extract_filters,
)
from voicelens.agent.state import AgentState, result_summary
from voicelens.agent.tools import AgentConfig

__all__ = [
    "ALL_ROUTES",
    "ROUTE_ANALYTICS",
    "ROUTE_INCIDENT",
    "ROUTE_INSUFFICIENT",
    "ROUTE_RETRIEVAL",
    "AgentConfig",
    "AgentState",
    "build_agent_graph",
    "classify_route",
    "extract_filters",
    "result_summary",
    "run_agent",
]
