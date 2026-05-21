"""Analytics layer: topic clustering (M4A) + anomaly detection (M4B).

The dashboard (M5) is deliberately not here yet. ``clustering`` and
``anomaly`` both define ``SEVERITY_WEIGHTS`` / ``severity_weight`` with
different weights — import those from the submodule you mean.
"""
from voicelens.analytics.anomaly import (
    AnomalyResult,
    MentionEvent,
    WeeklyBucket,
    aggregate_weekly,
    build_incident_summary,
    detect_anomalies,
    ewma_baseline,
)
from voicelens.analytics.clustering import (
    ClusterRecord,
    ClusterResult,
    build_clustering_records,
    cluster_records,
)
from voicelens.analytics.labels import build_cluster_label

__all__ = [
    "AnomalyResult",
    "ClusterRecord",
    "ClusterResult",
    "MentionEvent",
    "WeeklyBucket",
    "aggregate_weekly",
    "build_cluster_label",
    "build_clustering_records",
    "build_incident_summary",
    "cluster_records",
    "detect_anomalies",
    "ewma_baseline",
]
