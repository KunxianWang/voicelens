"""The M6B LangGraph workflow: route -> one tool -> format -> END.

A single deterministic routing decision picks exactly one tool node;
there is no looping, no re-planning, no multi-agent fan-out. LangGraph
supplies the ``StateGraph`` plumbing only — every decision in the graph
is plain Python.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from voicelens.agent.router import (
    ROUTE_ANALYTICS,
    ROUTE_INCIDENT,
    ROUTE_INSUFFICIENT,
    ROUTE_RETRIEVAL,
)
from voicelens.agent.state import AgentState, new_state
from voicelens.agent.tools import (
    AgentConfig,
    RetrieveFn,
    analytics_summary_node,
    format_response_node,
    incident_summary_node,
    insufficient_scope_node,
    retrieval_answer_node,
    route_question_node,
)

# route name -> graph node name (they are equal, kept explicit for clarity).
_ROUTE_TO_NODE: dict[str, str] = {
    ROUTE_RETRIEVAL: "retrieval_answer",
    ROUTE_INCIDENT: "incident_summary",
    ROUTE_ANALYTICS: "analytics_summary",
    ROUTE_INSUFFICIENT: "insufficient_scope",
}


def _route_selector(state: AgentState) -> str:
    """Conditional-edge function: read the route the router chose."""
    return state.get("route", ROUTE_INSUFFICIENT)


def build_agent_graph(config: AgentConfig | None = None):
    """Compile the agent workflow for one :class:`AgentConfig`.

    The tool nodes are bound to ``config`` with ``partial`` so the
    compiled graph carries its provider / retrieval settings.
    """
    cfg = config or AgentConfig()
    builder: StateGraph = StateGraph(AgentState)

    # Bind ``cfg`` with one-arg closures: LangGraph only injects extras
    # (config / store / runtime) into nodes that name those params, so a
    # single-``state`` closure keeps the tool config out of the graph API.
    builder.add_node("route_question", route_question_node)
    builder.add_node(
        "retrieval_answer", lambda s: retrieval_answer_node(s, cfg)
    )
    builder.add_node(
        "incident_summary", lambda s: incident_summary_node(s, cfg)
    )
    builder.add_node(
        "analytics_summary", lambda s: analytics_summary_node(s, cfg)
    )
    builder.add_node(
        "insufficient_scope", lambda s: insufficient_scope_node(s, cfg)
    )
    builder.add_node("format_response", format_response_node)

    builder.add_edge(START, "route_question")
    builder.add_conditional_edges(
        "route_question", _route_selector, _ROUTE_TO_NODE
    )
    for node in _ROUTE_TO_NODE.values():
        builder.add_edge(node, "format_response")
    builder.add_edge("format_response", END)

    return builder.compile()


def run_agent(
    question: str,
    *,
    provider: str = "mock",
    model: str | None = None,
    top_k: int = 8,
    retrieve_fn: RetrieveFn | None = None,
    config: AgentConfig | None = None,
) -> AgentState:
    """Run one question through the agent workflow and return final state.

    ``retrieve_fn`` may be injected (tests use an in-memory Qdrant);
    otherwise the retrieval route hits the live Qdrant index.
    """
    if config is None:
        config = AgentConfig(
            provider=provider, model=model, top_k=top_k, retrieve_fn=retrieve_fn
        )
    graph = build_agent_graph(config)
    initial = new_state(question, provider=config.provider, model=config.model)
    return graph.invoke(initial)  # type: ignore[return-value]
