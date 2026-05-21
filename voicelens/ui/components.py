"""Small shared Streamlit widgets for the dashboard pages.

Kept deliberately thin — these wrap repetitive ``st.*`` calls (metric
rows, empty states, bar charts) so the page modules stay readable.
Nothing here touches the database.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st


def page_header(title: str, subtitle: str = "") -> None:
    """Render a page title and an optional one-line description."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def metric_row(metrics: Sequence[tuple[str, Any]]) -> None:
    """Render a horizontal row of ``st.metric`` cards."""
    if not metrics:
        return
    for col, (label, value) in zip(st.columns(len(metrics)), metrics, strict=True):
        col.metric(label, value)


def empty_state(message: str) -> None:
    """Uniform 'nothing to show' banner so empty filters never crash a page."""
    st.info(message, icon="ℹ️")


def bar_chart_from_counts(
    counts: Mapping[str, Any],
    *,
    label: str = "category",
    value: str = "count",
) -> None:
    """Draw a bar chart from a ``{name: count}`` mapping (empty-safe)."""
    if not counts:
        empty_state("No data for this chart yet.")
        return
    frame = pd.DataFrame(
        {label: list(counts.keys()), value: list(counts.values())}
    )
    st.bar_chart(frame, x=label, y=value)


def bar_chart_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    value: str,
) -> None:
    """Draw a bar chart from a list of dict rows (empty-safe)."""
    if not rows:
        empty_state("No data for this chart yet.")
        return
    st.bar_chart(pd.DataFrame(list(rows)), x=label, y=value)


def dataframe(rows: Sequence[Mapping[str, Any]], *, empty_message: str) -> None:
    """Render a list of dict rows as a full-width table, or an empty state."""
    if not rows:
        empty_state(empty_message)
        return
    st.dataframe(pd.DataFrame(list(rows)), use_container_width=True, hide_index=True)
