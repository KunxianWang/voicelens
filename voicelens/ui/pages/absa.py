"""ABSA distribution page — aspect/sentiment/severity breakdowns."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from voicelens.ui import db
from voicelens.ui.components import (
    bar_chart_from_counts,
    dataframe,
    empty_state,
    metric_row,
    page_header,
)


def _opt(value: str) -> str | None:
    """Map the sentinel ``(any)`` option back to ``None``."""
    return None if value == "(any)" else value


def render() -> None:
    page_header(
        "🏷️ ABSA Distribution",
        "Aspect-based sentiment over the ABSA-processed reviews. "
        "Negative mentions are the quality signal the rest of the "
        "pipeline (clusters, incidents) is built on.",
    )

    with st.sidebar:
        st.subheader("ABSA filters")
        brand = st.selectbox("Brand", ["(any)", *db.list_brands()])
        aspect = st.selectbox("Aspect", ["(any)", *db.list_aspects()])
        sentiment = st.selectbox(
            "Sentiment", ["(any)", "negative", "neutral", "positive"]
        )
        severity = st.selectbox("Severity", ["(any)", "low", "medium", "high"])
        rating_min, rating_max = st.slider(
            "Rating range", min_value=1, max_value=5, value=(1, 5)
        )

    filters = dict(
        brand=_opt(brand),
        aspect=_opt(aspect),
        rating_min=rating_min,
        rating_max=rating_max,
    )

    # ---- aspect x sentiment matrix --------------------------------------
    st.subheader("Aspect × sentiment counts")
    matrix = db.get_aspect_sentiment_matrix(
        sentiment=_opt(sentiment), severity=_opt(severity), **filters
    )
    if matrix:
        pivot = (
            pd.DataFrame(matrix)
            .pivot_table(
                index="aspect_code", columns="sentiment", values="count",
                fill_value=0, aggfunc="sum",
            )
            .astype(int)
        )
        st.dataframe(pivot, use_container_width=True)
        total = sum(r["count"] for r in matrix)
        neg = sum(r["count"] for r in matrix if r["sentiment"] == "negative")
        metric_row(
            [
                ("Mentions (filtered)", f"{total:,}"),
                ("Negative", f"{neg:,}"),
                ("Negative share", f"{(neg / total * 100):.1f}%" if total else "0%"),
            ]
        )
    else:
        empty_state("No mentions match the current filters.")

    st.divider()

    # ---- severity distribution for negatives ----------------------------
    st.subheader("Severity distribution (negative mentions)")
    severities = db.get_severity_distribution(**filters)
    bar_chart_from_counts(severities, label="severity", value="negative mentions")

    st.divider()

    # ---- top negative examples ------------------------------------------
    st.subheader("Negative examples")
    examples = db.get_negative_examples(
        severity=_opt(severity), limit=50, **filters
    )
    dataframe(
        examples,
        empty_message="No negative mentions match the current filters.",
    )

    st.divider()

    # ---- reliability spotlight ------------------------------------------
    st.subheader("🔧 Reliability spotlight")
    st.caption(
        "Reliability is the largest negative-quality signal in the 1k "
        "subset — it dominates both clustering and the detected incidents."
    )
    rel_examples = db.get_negative_examples(
        brand=_opt(brand),
        aspect="reliability",
        severity=_opt(severity),
        rating_min=rating_min,
        rating_max=rating_max,
        limit=30,
    )
    if rel_examples:
        metric_row([("Reliability negatives shown", len(rel_examples))])
        for row in rel_examples[:10]:
            st.markdown(
                f"**{row['brand']} {row['asin']}** · rating {row['rating']} · "
                f"severity `{row['severity']}`"
            )
            if row["evidence_quote"]:
                st.write(f"> {row['evidence_quote']}")
    else:
        empty_state("No reliability negatives for the current filters.")
