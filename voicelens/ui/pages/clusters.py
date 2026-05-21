"""Issue clusters page — M4A topic clusters ranked by severity weight."""
from __future__ import annotations

import streamlit as st

from voicelens.ui import db
from voicelens.ui.components import dataframe, empty_state, metric_row, page_header


def render() -> None:
    page_header(
        "🧩 Issue Clusters",
        "Negative ABSA mentions grouped into emergent issue clusters "
        "(Milestone 4A). Ranked by severity-weighted size — a few severe "
        "complaints out-rank a pile of minor ones.",
    )

    with st.sidebar:
        st.subheader("Cluster filters")
        aspect = st.selectbox("Aspect", ["(any)", *db.list_aspects()])

    clusters = db.get_clusters(aspect=None if aspect == "(any)" else aspect)
    if not clusters:
        empty_state(
            "No clusters yet. Run `make cluster-v2` to populate the "
            "cluster table."
        )
        return

    metric_row(
        [
            ("Clusters", len(clusters)),
            ("Total members", sum(c["size"] for c in clusters)),
            (
                "Top severity-weighted",
                clusters[0]["severity_weighted_size"],
            ),
        ]
    )

    st.subheader("Clusters by severity-weighted size")
    dataframe(
        [
            {
                "cluster_id": c["cluster_id"],
                "aspect": c["aspect_code"],
                "label": c["label"],
                "size": c["size"],
                "severity_weighted_size": c["severity_weighted_size"],
                "algorithm": c["algorithm"],
            }
            for c in clusters
        ],
        empty_message="No clusters for this aspect.",
    )

    st.divider()
    st.subheader("Inspect a cluster")
    options = {
        f"#{c['cluster_id']} · [{c['aspect_code']}] {c['label']} "
        f"(size {c['size']})": c["cluster_id"]
        for c in clusters
    }
    chosen = st.selectbox("Cluster", list(options.keys()))
    cluster_id = options[chosen]
    selected = next(c for c in clusters if c["cluster_id"] == cluster_id)

    metric_row(
        [
            ("Size", selected["size"]),
            ("Severity-weighted", selected["severity_weighted_size"]),
            ("Aspect", selected["aspect_code"]),
        ]
    )
    if selected["topic_keywords"]:
        st.write("**Topic keywords:** " + ", ".join(selected["topic_keywords"]))
    if selected["representative_quotes"]:
        st.write("**Representative quotes:**")
        for quote in selected["representative_quotes"]:
            if quote:
                st.write(f"> {quote}")

    st.subheader("Reviews in this cluster")
    members = db.get_cluster_members(cluster_id, limit=200)
    dataframe(
        members,
        empty_message="No member reviews found for this cluster.",
    )
