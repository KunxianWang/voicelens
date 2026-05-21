"""Overview page — headline counts and distribution charts."""
from __future__ import annotations

import streamlit as st

from voicelens.config import QDRANT_COLLECTION, QDRANT_URL
from voicelens.ui import db
from voicelens.ui.components import (
    bar_chart_from_counts,
    bar_chart_from_rows,
    metric_row,
    page_header,
)


def _qdrant_point_count() -> int | None:
    """Indexed-point count from Qdrant, or ``None`` if it is unreachable."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(QDRANT_COLLECTION):
            return None
        return int(client.count(QDRANT_COLLECTION).count)
    except Exception:  # noqa: BLE001 - Qdrant is optional for this page
        return None


def render() -> None:
    page_header(
        "📊 VoiceLens Overview",
        "A read-only snapshot of the current pipeline outputs — "
        "ingestion, ABSA, clustering and anomaly detection.",
    )

    stats = db.get_overview_stats()
    qdrant_points = _qdrant_point_count()

    metric_row(
        [
            ("Total reviews", f"{stats['total_reviews']:,}"),
            ("ABSA-processed", f"{stats['absa_processed_reviews']:,}"),
            ("Aspect mentions", f"{stats['total_aspect_mentions']:,}"),
            ("Brands", stats["total_brands"]),
        ]
    )
    metric_row(
        [
            ("Issue clusters", stats["total_clusters"]),
            ("Incidents", stats["total_incidents"]),
            ("Ingest runs", stats["total_ingest_runs"]),
            (
                "Qdrant points",
                f"{qdrant_points:,}" if qdrant_points is not None else "n/a",
            ),
        ]
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Reviews by brand")
        bar_chart_from_rows(
            db.get_reviews_by_brand(), label="brand", value="reviews"
        )
    with right:
        st.subheader("ABSA status distribution")
        bar_chart_from_counts(
            db.get_absa_status_distribution(), label="status", value="reviews"
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Aspect distribution")
        bar_chart_from_rows(
            db.get_aspect_distribution(), label="aspect_code", value="mentions"
        )
    with right:
        st.subheader("Sentiment distribution")
        bar_chart_from_counts(
            db.get_sentiment_distribution(), label="sentiment", value="mentions"
        )

    if qdrant_points is None:
        st.caption(
            "Qdrant index not reachable — start it with `make up` and index "
            "with `make embed-v2-1k` to enable the retrieval page."
        )
