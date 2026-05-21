"""Emerging incidents page — M4B anomaly detections."""
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
        "🚨 Emerging Incidents",
        "Weeks where an issue cluster (or aspect) spiked above its EWMA "
        "baseline (Milestone 4B). Each summary is generated deterministically "
        "by the anomaly pipeline — no LLM involved.",
    )

    granularities = db.list_incident_granularities()
    aspects = db.list_incident_aspects()

    if not granularities:
        empty_state(
            "No incidents yet. Run `make anomaly-v2` (cluster-level) and the "
            "aspect-level flow to populate the incident table."
        )
        return

    with st.sidebar:
        st.subheader("Incident filters")
        granularity = st.selectbox("Granularity", ["(any)", *granularities])
        aspect = st.selectbox("Aspect", ["(any)", *aspects])

    incidents = db.get_incidents(
        aspect=None if aspect == "(any)" else aspect,
        granularity=None if granularity == "(any)" else granularity,
    )
    if not incidents:
        empty_state("No incidents match the selected filters.")
        return

    metric_row(
        [
            ("Incidents", len(incidents)),
            ("Top severity score", incidents[0]["severity_score"]),
            ("Top z-score", max(i["z_score"] for i in incidents)),
        ]
    )

    by_aspect: dict[str, int] = {}
    for inc in incidents:
        by_aspect[inc["aspect_code"]] = by_aspect.get(inc["aspect_code"], 0) + 1
    st.subheader("Incidents by aspect")
    bar_chart_from_counts(by_aspect, label="aspect", value="incidents")

    st.divider()
    st.subheader("Incidents by severity score")
    dataframe(
        [
            {
                "incident_id": i["incident_id"],
                "granularity": i["granularity"],
                "aspect": i["aspect_code"],
                "cluster_label": i["cluster_label"] or "(aspect-level)",
                "week_start": i["week_start"],
                "observed": i["observed_volume"],
                "baseline": i["baseline_volume"],
                "z_score": i["z_score"],
                "severity_score": i["severity_score"],
            }
            for i in incidents
        ],
        empty_message="No incidents to show.",
    )

    st.divider()
    st.subheader("Incident detail")
    for inc in incidents:
        label = inc["cluster_label"] or f"{inc['aspect_code']} (aspect-level)"
        with st.expander(
            f"[{inc['granularity']}] {label} · week {inc['week_start']} · "
            f"z={inc['z_score']} · severity={inc['severity_score']}"
        ):
            st.write(inc["summary"] or "(no summary)")
            metric_row(
                [
                    ("Observed volume", inc["observed_volume"]),
                    ("Baseline volume", inc["baseline_volume"]),
                    ("z-score", inc["z_score"]),
                    ("Unique reviews", inc["unique_review_count"]),
                ]
            )
            if inc["example_quotes"]:
                st.write("**Representative quotes:**")
                for quote in inc["example_quotes"]:
                    if quote:
                        st.write(f"> {quote}")
