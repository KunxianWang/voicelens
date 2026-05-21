"""Data quality / pipeline stats page."""
from __future__ import annotations

import streamlit as st

from voicelens.ui import db
from voicelens.ui.components import (
    bar_chart_from_counts,
    dataframe,
    empty_state,
    metric_row,
    page_header,
)


def render() -> None:
    page_header(
        "🩺 Data Quality & Pipeline Stats",
        "Ingestion DQ checks and ABSA processing reliability — the "
        "guardrails that keep the downstream analytics trustworthy.",
    )

    # ---- ingest runs ----------------------------------------------------
    st.subheader("Ingest runs")
    dataframe(
        db.get_ingest_runs(),
        empty_message="No ingest runs recorded yet.",
    )

    st.divider()

    # ---- DQ events ------------------------------------------------------
    st.subheader("Data-quality checks")
    dq = db.get_dq_summary()
    metric_row(
        [
            ("DQ events", dq["total_events"]),
            ("Rows checked", f"{dq['total_rows_checked']:,}"),
            ("Rows passed", f"{dq['total_passed']:,}"),
            ("Rows failed", f"{dq['total_failed']:,}"),
        ]
    )
    if dq["by_check"]:
        dataframe(
            [
                {
                    "check": name,
                    "rows": vals["rows"],
                    "failed": vals["failed"],
                    "pass_rate": (
                        round(1 - vals["failed"] / vals["rows"], 4)
                        if vals["rows"]
                        else None
                    ),
                }
                for name, vals in sorted(dq["by_check"].items())
            ],
            empty_message="No DQ checks recorded.",
        )
    else:
        empty_state("No DQ events recorded yet.")

    st.subheader("DQ failures by reason")
    bar_chart_from_counts(
        dq["failures_by_reason"], label="reason", value="failed rows"
    )

    st.divider()

    # ---- ABSA reliability ----------------------------------------------
    st.subheader("ABSA processing reliability")
    cov = db.get_pipeline_coverage()
    bar_chart_from_counts(
        cov["status_counts"], label="status", value="reviews"
    )
    metric_row(
        [
            ("Processed coverage", f"{cov['processed_coverage_rate'] * 100:.1f}%"),
            ("Mention coverage", f"{cov['mention_coverage_rate'] * 100:.1f}%"),
            ("ABSA success rate", f"{cov['absa_success_rate'] * 100:.1f}%"),
        ]
    )
    metric_row(
        [
            ("ABSA failed rate", f"{cov['absa_failed_rate'] * 100:.1f}%"),
            ("ABSA invalid rate", f"{cov['absa_invalid_rate'] * 100:.1f}%"),
            ("Total mentions", f"{cov['total_mentions']:,}"),
        ]
    )
    st.caption(
        "Processed coverage = ABSA-processed ÷ total reviews. "
        "Mention coverage = reviews with ≥1 aspect mention ÷ processed reviews. "
        "failed / invalid rates flag LLM-batch reliability problems."
    )
