"""Route nodes for the M6B agent graph.

Each route is a plain function ``state -> partial-state-update``. Three
of them are non-LLM (incident / analytics read Postgres directly via
:mod:`voicelens.ui.db`); the retrieval route reuses the M6A
citation-grounded answer generator. The graph in :mod:`voicelens.agent.graph`
wires them together.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from voicelens.agent.prompts import INSUFFICIENT_SCOPE_MESSAGE
from voicelens.agent.router import classify_route, extract_filters
from voicelens.agent.state import AgentState
from voicelens.rag.answer import generate_answer
from voicelens.rag.retrieve import retrieve_reviews
from voicelens.retrieval.search import SearchHit
from voicelens.ui import db

# A retrieval function: (question, filters, top_k) -> hits. Injectable so
# tests can supply hand-built hits without a live Qdrant.
RetrieveFn = Callable[[str, dict, int], list[SearchHit]]


@dataclass
class AgentConfig:
    """Run-time configuration for one agent invocation."""

    provider: str = "mock"
    model: str | None = None
    top_k: int = 8
    retrieve_fn: RetrieveFn | None = None
    max_incidents: int = 5


def _default_retrieve(question: str, filters: dict, top_k: int) -> list[SearchHit]:
    """Default retrieval: hybrid search with the extracted filters."""
    return retrieve_reviews(
        question,
        mode="hybrid",
        brand=filters.get("brand"),
        aspect=filters.get("aspect"),
        sentiment=filters.get("sentiment"),
        top_k=top_k,
    )


# ---- routing ------------------------------------------------------------


def route_question_node(state: AgentState) -> dict:
    """Classify the question and, for the retrieval route, extract filters."""
    question = state.get("question", "")
    route = classify_route(question)
    filters = extract_filters(question) if route == "retrieval_answer" else {}
    return {"route": route, "filters": filters}


# ---- retrieval route ----------------------------------------------------


def retrieval_answer_node(state: AgentState, config: AgentConfig) -> dict:
    """Retrieve reviews, then generate a citation-grounded answer."""
    question = state.get("question", "")
    filters = state.get("filters", {})
    retrieve_fn = config.retrieve_fn or _default_retrieve

    try:
        hits = retrieve_fn(question, filters, config.top_k)
    except RuntimeError as exc:
        return {
            "answer": (
                "Retrieval is unavailable, so I can't ground an answer right "
                f"now ({exc})."
            ),
            "citations": [],
            "retrieved_results": [],
            "retrieved_review_ids": [],
            "warnings": [f"retrieval unavailable: {exc}"],
        }

    result = generate_answer(
        question, hits, provider=config.provider, model=config.model,
        filters=filters,
    )
    warnings: list[str] = []
    if result.insufficient_evidence:
        warnings.append("answer flagged insufficient evidence")
    unsupported = result.guardrail_flags.get("unsupported_citations") or []
    if unsupported:
        warnings.append(f"stripped unsupported citation markers {unsupported}")

    return {
        "answer": result.answer,
        "citations": result.citations,
        "retrieved_results": hits,
        "retrieved_review_ids": result.retrieved_review_ids,
        "warnings": warnings,
    }


# ---- incident route -----------------------------------------------------


def incident_summary_node(state: AgentState, config: AgentConfig) -> dict:
    """Summarise the top emerging-issue incidents from the incident table."""
    incidents = db.get_incidents()
    if not incidents:
        return {
            "incidents_result": [],
            "answer": (
                "No emerging-issue incidents have been detected. The incident "
                "table is empty — run `make anomaly-v2` to populate it."
            ),
            "warnings": ["incident table is empty"],
        }

    top = incidents[: config.max_incidents]
    lines: list[str] = [
        f"Top {len(top)} emerging-issue incident(s) by severity score:"
    ]
    for i, inc in enumerate(top, start=1):
        unit = inc["cluster_label"] or f"{inc['aspect_code']} (aspect-level)"
        lines.append(
            f"{i}. [{inc['granularity']}] {unit} — week {inc['week_start']}: "
            f"observed {inc['observed_volume']} vs baseline "
            f"{inc['baseline_volume']} (z={inc['z_score']}, "
            f"severity {inc['severity_score']})."
        )
        if inc.get("summary"):
            lines.append(f"   {inc['summary']}")
    return {
        "incidents_result": top,
        "answer": "\n".join(lines),
        "warnings": [],
    }


# ---- analytics route ----------------------------------------------------


def _format_counts(counts: dict[str, int]) -> str:
    """Render a ``{label: count}`` mapping as ``a=1, b=2`` (desc)."""
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{label}={n}" for label, n in ordered) or "(none)"


def analytics_summary_node(state: AgentState, config: AgentConfig) -> dict:
    """Report aggregate ABSA statistics chosen from the question wording."""
    question = state.get("question", "").lower()

    if "severity" in question:
        kind = "severity_distribution"
        counts = db.get_severity_distribution()
        body = f"Severity distribution of negative mentions: {_format_counts(counts)}."
        result: dict = {"kind": kind, "severity_distribution": counts}
    elif "sentiment" in question:
        kind = "sentiment_distribution"
        counts = db.get_sentiment_distribution()
        body = f"Sentiment distribution of aspect mentions: {_format_counts(counts)}."
        result = {"kind": kind, "sentiment_distribution": counts}
    elif "status" in question:
        kind = "absa_status_distribution"
        counts = db.get_absa_status_distribution()
        body = f"ABSA processing-status distribution: {_format_counts(counts)}."
        result = {"kind": kind, "absa_status_distribution": counts}
    else:
        # Default: aspect distribution, with the top negative aspect called out.
        kind = "aspect_distribution"
        aspect_rows = db.get_aspect_distribution()
        negative_rows = db.get_aspect_sentiment_matrix(sentiment="negative")
        neg_by_aspect: dict[str, int] = {}
        for row in negative_rows:
            neg_by_aspect[row["aspect_code"]] = (
                neg_by_aspect.get(row["aspect_code"], 0) + row["count"]
            )
        total = sum(r["mentions"] for r in aspect_rows)
        result = {
            "kind": kind,
            "aspect_distribution": aspect_rows,
            "negative_by_aspect": neg_by_aspect,
            "total_mentions": total,
        }
        if not aspect_rows:
            body = "No aspect mentions are present in the database yet."
        else:
            top = aspect_rows[0]
            top_neg = (
                max(neg_by_aspect.items(), key=lambda kv: kv[1])
                if neg_by_aspect
                else None
            )
            body = (
                f"Aspect distribution across {total} mentions: "
                f"{_format_counts({r['aspect_code']: r['mentions'] for r in aspect_rows})}. "
                f"Largest aspect: {top['aspect_code']} ({top['mentions']} mentions)."
            )
            if top_neg is not None:
                body += (
                    f" Most negative aspect: {top_neg[0]} "
                    f"({top_neg[1]} negative mentions)."
                )

    return {"analytics_result": result, "answer": body, "warnings": []}


# ---- insufficient-scope route ------------------------------------------


def insufficient_scope_node(state: AgentState, config: AgentConfig) -> dict:
    """Return a safe, honest out-of-scope message — no hallucinated answer."""
    return {
        "answer": INSUFFICIENT_SCOPE_MESSAGE,
        "citations": [],
        "warnings": ["question routed to insufficient_scope"],
    }


# ---- response formatting ------------------------------------------------


def format_response_node(state: AgentState) -> dict:
    """Final node: guarantee ``answer`` and ``warnings`` keys are present."""
    update: dict = {}
    if not state.get("answer"):
        update["answer"] = "(no answer was produced)"
    if state.get("warnings") is None:
        update["warnings"] = []
    return update
