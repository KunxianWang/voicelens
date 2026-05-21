"""VoiceLens analytics dashboard — Streamlit entry point (Milestone 5A).

Run it with::

    make dashboard          # streamlit run voicelens/ui/app.py

The app is a read-only view over Postgres + Qdrant. It visualises the
existing pipeline outputs (ABSA, clusters, incidents) and offers a
search demo over the retrieval index. It never calls a real LLM and is
**not** the final RAG layer — that is a later milestone.

Importing this module is side-effect free: the navigation only runs
when a Streamlit runtime is active, so ``import voicelens.ui.app``
stays a safe smoke check.
"""
from __future__ import annotations

import streamlit as st

from voicelens.ui.pages import (
    absa,
    clusters,
    data_quality,
    incidents,
    overview,
    retrieval,
)


def build_navigation():
    """Assemble the six dashboard pages and return the selected page."""
    pages = [
        st.Page(
            overview.render, title="Overview", icon="📊",
            url_path="overview", default=True,
        ),
        st.Page(
            absa.render, title="ABSA Distribution", icon="🏷️",
            url_path="absa",
        ),
        st.Page(
            clusters.render, title="Issue Clusters", icon="🧩",
            url_path="clusters",
        ),
        st.Page(
            incidents.render, title="Emerging Incidents", icon="🚨",
            url_path="incidents",
        ),
        st.Page(
            retrieval.render, title="Retrieval Search", icon="🔎",
            url_path="retrieval",
        ),
        st.Page(
            data_quality.render, title="Data Quality", icon="🩺",
            url_path="data-quality",
        ),
    ]
    return st.navigation(pages)


def main() -> None:
    """Configure the page and run the selected dashboard page."""
    st.set_page_config(
        page_title="VoiceLens Analytics",
        page_icon="📊",
        layout="wide",
    )
    st.sidebar.title("VoiceLens")
    st.sidebar.caption("VoC analytics dashboard · Milestone 5A")
    build_navigation().run()


if st.runtime.exists():  # only when launched via `streamlit run`
    main()
