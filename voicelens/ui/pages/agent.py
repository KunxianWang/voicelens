"""Agent Q&A page (Beta) — the M6B LangGraph workflow in the dashboard.

A natural-language question is routed by the deterministic agent to one
of four tools (retrieval answer / incident summary / analytics summary
/ insufficient scope); the page shows the chosen route, the answer, any
citations and any warnings. It is a controlled routing workflow — not
an autonomous agent.
"""
from __future__ import annotations

import streamlit as st

from voicelens.agent import result_summary, run_agent
from voicelens.agent.prompts import ROUTE_DESCRIPTIONS
from voicelens.rag.providers import KNOWN_PROVIDERS
from voicelens.ui.components import empty_state, page_header

_EXAMPLES = (
    "What are the main reliability complaints?",
    "Which issues spiked recently?",
    "Which aspect has the most negative mentions?",
)

_HELP_TEXT = """\
Ask one question in plain language. A deterministic keyword router
sends it to **one** of these tools:

- **Retrieval answer** — *complaints, what customers say, examples,
  quotes* about an aspect (reliability, bluetooth, charging, price,
  delivery, sound quality, overheating, battery). Returns a
  citation-grounded answer.
- **Incident summary** — *incidents, spikes, anomalies, emerging
  issues*. Returns the top anomaly incidents.
- **Analytics summary** — *how many, counts, distribution, top aspect,
  most common*. Returns aspect / sentiment / severity statistics.
- **Insufficient scope** — anything outside VoC analytics gets a safe
  refusal rather than a guessed answer.

This is a single routing pass — no memory, no actions, no planning.
"""


def _render_result(summary: dict) -> None:
    route = summary["route"]
    st.info(f"**Route selected:** `{route}` — {ROUTE_DESCRIPTIONS.get(route, '')}")

    meta = st.columns(3)
    meta[0].metric("Route", route)
    meta[1].metric("Citations", len(summary["citations"]))
    meta[2].metric("Warnings", len(summary["warnings"]))
    if summary["filters"]:
        st.caption(f"Extracted filters: `{summary['filters']}`")
    else:
        st.caption("Extracted filters: (none)")

    st.subheader("Answer")
    if route == "insufficient_scope":
        st.warning(summary["answer"])
    else:
        st.success(summary["answer"])

    citations = summary["citations"]
    if citations:
        with st.expander(f"Citations ({len(citations)})"):
            for c in citations:
                rating = c["rating"] if c["rating"] is not None else "-"
                st.markdown(
                    f"**[{c['citation_id']}]** {c['brand']} {c['asin']} · "
                    f"rating {rating} · score {c['retrieval_score']} · "
                    f"review_id {c['review_id']}"
                )
                if c["evidence_quote"]:
                    st.write(f"> {c['evidence_quote']}")

    warnings = summary["warnings"]
    if warnings:
        st.markdown("**Warnings**")
        for w in warnings:
            st.caption(f"⚠ {w}")
    st.caption(
        f"Agent Q&A Beta · provider={summary['provider']} "
        f"· {len(summary['retrieved_review_ids'])} reviews retrieved"
    )


def render() -> None:
    page_header(
        "🤖 Agent Q&A (Beta)",
        "Ask a question in plain language. A deterministic LangGraph router "
        "sends it to retrieval, incident, or analytics — one routing pass, "
        "no memory, no autonomous actions.",
    )

    with st.sidebar:
        st.subheader("Agent settings")
        provider = st.selectbox(
            "Answer provider", list(KNOWN_PROVIDERS), key="agent_provider"
        )
        st.caption(
            "`mock` works offline. `anthropic` / `openai` need an API key "
            "and only affect the retrieval route."
        )

    with st.expander("ℹ️ What can I ask?"):
        st.markdown(_HELP_TEXT)

    st.caption("Example questions: " + " · ".join(f"*{e}*" for e in _EXAMPLES))
    question = st.text_input(
        "Your question", placeholder=_EXAMPLES[0], key="agent_question"
    )

    if not question.strip():
        empty_state("Enter a question above and click **Ask the agent**.")
        return

    if not st.button("Ask the agent", key="agent_run"):
        return

    try:
        state = run_agent(question, provider=provider)
    except (RuntimeError, ValueError) as exc:
        st.warning(
            f"Could not complete the request with provider `{provider}`: "
            f"{exc}\n\nTry the `mock` provider for an offline answer."
        )
        return

    _render_result(result_summary(state))
