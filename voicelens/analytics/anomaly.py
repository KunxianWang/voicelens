"""EWMA-based anomaly detection over clustered negative ABSA mentions (M4B).

The pipeline turns clustered negative mentions into weekly volume series
and flags weeks that spike above an exponentially-weighted baseline:

1. :func:`aggregate_weekly` groups mention events into ``WeeklyBucket``
   rows per series (a cluster, or an aspect for the sparse-data fallback).
2. :func:`detect_anomalies` walks each series in week order, computing an
   EWMA baseline and a rolling std over *previous* weeks only, then a
   z-score for the current week. Weeks clearing the z / volume / severity
   thresholds become ``AnomalyResult`` incidents.

Everything here is pure (no DB, no LLM, no pandas) so it unit-tests on
hand-built event lists. ``anomaly_flow`` supplies the events from
Postgres and persists the incidents.

Series use **observed weeks only** — gaps between active weeks are not
zero-filled. On the 1k MVP subset that keeps the baseline meaningful
(a zero-filled series would make every busy week look anomalous); proper
calendar-week gap-filling is deferred.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# Severity weights for M4B incident scoring. Note these are steeper than
# the clustering weights (M4A used 3/2/1) — anomaly scoring deliberately
# punishes "high" severity harder so a few severe complaints register.
SEVERITY_WEIGHTS: dict[str, float] = {"low": 1.0, "medium": 2.0, "high": 4.0}
_DEFAULT_SEVERITY_WEIGHT = 1.0

GRANULARITY_CLUSTER = "cluster"
GRANULARITY_ASPECT = "aspect"


def severity_weight(severity: str | None) -> float:
    """Map an ABSA severity label to its anomaly-scoring weight."""
    if not severity:
        return _DEFAULT_SEVERITY_WEIGHT
    return SEVERITY_WEIGHTS.get(severity.strip().lower(), _DEFAULT_SEVERITY_WEIGHT)


def week_start(value: date | datetime) -> date:
    """Monday (ISO) of the week containing ``value``."""
    day = value.date() if isinstance(value, datetime) else value
    return day - timedelta(days=day.weekday())


@dataclass
class MentionEvent:
    """One clustered negative mention, the atomic unit of the weekly series."""

    review_id: int
    cluster_id: int | None
    aspect_code: str
    severity: str | None
    posted_at: date | datetime
    rating: int | None
    evidence_quote: str = ""


@dataclass
class WeeklyBucket:
    """Aggregated weekly volume for one series (cluster or aspect)."""

    series_key: str
    granularity: str
    cluster_id: int | None
    aspect_code: str
    week_start: date
    observed_volume: int
    severity_weighted_volume: float
    unique_review_count: int
    avg_rating: float | None
    example_quotes: list[str] = field(default_factory=list)


@dataclass
class AnomalyResult:
    """A weekly bucket flagged as an incident, with its baseline + z-score."""

    series_key: str
    granularity: str
    cluster_id: int | None
    aspect_code: str
    week_start: date
    observed_volume: int
    baseline_volume: float
    rolling_std: float
    z_score: float
    severity_score: float
    unique_review_count: int
    avg_rating: float | None
    example_quotes: list[str] = field(default_factory=list)
    summary: str = ""


def _series_key(granularity: str, cluster_id: int | None, aspect_code: str) -> str:
    """Stable string key for a series (clusters keyed by id, aspects by code)."""
    if granularity == GRANULARITY_CLUSTER:
        return f"cluster:{cluster_id}"
    return f"aspect:{aspect_code}"


def aggregate_weekly(
    events: list[MentionEvent],
    *,
    granularity: str = GRANULARITY_CLUSTER,
    max_example_quotes: int = 3,
) -> dict[str, list[WeeklyBucket]]:
    """Group mention events into per-series, week-sorted ``WeeklyBucket`` lists.

    Cluster-level keys on ``cluster_id``; aspect-level keys on
    ``aspect_code``. Events with no ``cluster_id`` are skipped under
    cluster granularity (they cannot belong to a cluster series).
    """
    grouped: dict[tuple[str, date], list[MentionEvent]] = defaultdict(list)
    for event in events:
        if granularity == GRANULARITY_CLUSTER and event.cluster_id is None:
            continue
        key = _series_key(granularity, event.cluster_id, event.aspect_code)
        grouped[(key, week_start(event.posted_at))].append(event)

    buckets_by_series: dict[str, list[WeeklyBucket]] = defaultdict(list)
    for (key, wk), bucket_events in grouped.items():
        ratings = [e.rating for e in bucket_events if e.rating is not None]
        quotes: list[str] = []
        seen_quotes: set[str] = set()
        for event in bucket_events:
            quote = (event.evidence_quote or "").strip()
            if quote and quote.lower() not in seen_quotes and len(quotes) < max_example_quotes:
                seen_quotes.add(quote.lower())
                quotes.append(quote)
        first = bucket_events[0]
        buckets_by_series[key].append(
            WeeklyBucket(
                series_key=key,
                granularity=granularity,
                cluster_id=first.cluster_id if granularity == GRANULARITY_CLUSTER else None,
                aspect_code=first.aspect_code,
                week_start=wk,
                observed_volume=len(bucket_events),
                severity_weighted_volume=round(
                    sum(severity_weight(e.severity) for e in bucket_events), 4
                ),
                unique_review_count=len({e.review_id for e in bucket_events}),
                avg_rating=round(statistics.fmean(ratings), 4) if ratings else None,
                example_quotes=quotes,
            )
        )

    for key in buckets_by_series:
        buckets_by_series[key].sort(key=lambda b: b.week_start)
    return dict(buckets_by_series)


def ewma_baseline(prev_volumes: list[float], *, span: int) -> float | None:
    """EWMA over *previous* weeks; ``None`` when there is no history.

    ``span`` follows the pandas convention: ``alpha = 2 / (span + 1)``.
    The current week is never part of its own baseline — the caller
    passes only the weeks before it.
    """
    if not prev_volumes:
        return None
    alpha = 2.0 / (span + 1.0)
    estimate = float(prev_volumes[0])
    for volume in prev_volumes[1:]:
        estimate = alpha * float(volume) + (1.0 - alpha) * estimate
    return estimate


def rolling_std(prev_volumes: list[float], *, window: int | None) -> float | None:
    """Sample std over the last ``window`` previous weeks (all if ``None``).

    Returns ``None`` when fewer than two weeks are available, so the
    caller can fall back to a std floor instead of dividing by nothing.
    """
    sample = prev_volumes[-window:] if window else prev_volumes
    if len(sample) < 2:
        return None
    return statistics.stdev(float(v) for v in sample)


def z_score(
    observed: float,
    baseline: float,
    std: float | None,
    *,
    min_std: float,
) -> float:
    """``(observed - baseline) / max(std, min_std)``.

    ``min_std`` is a floor that both prevents division by zero and stops
    a near-flat series from producing an explosive z on a small bump.
    """
    effective_std = max(std or 0.0, min_std)
    return (observed - baseline) / effective_std


def detect_anomalies(
    buckets_by_series: dict[str, list[WeeklyBucket]],
    *,
    min_history_weeks: int = 3,
    z_threshold: float = 2.0,
    min_volume: int = 3,
    min_severity_score: float = 5.0,
    ewma_span: int = 4,
    rolling_window: int | None = 8,
    min_std: float = 1.0,
) -> list[AnomalyResult]:
    """Flag weekly spikes across every series.

    A week is an incident when **all** hold:

    - it has at least ``min_history_weeks`` earlier weeks in its series,
    - ``z_score >= z_threshold`` against the EWMA baseline,
    - ``observed_volume >= min_volume`` (guardrail vs one-off blips),
    - ``severity_weighted_volume >= min_severity_score``.
    """
    incidents: list[AnomalyResult] = []
    for series_key in sorted(buckets_by_series):
        buckets = buckets_by_series[series_key]
        volumes = [float(b.observed_volume) for b in buckets]
        for i, bucket in enumerate(buckets):
            prev = volumes[:i]
            if len(prev) < min_history_weeks:
                continue  # not enough history to judge this week
            baseline = ewma_baseline(prev, span=ewma_span)
            if baseline is None:
                continue
            std = rolling_std(prev, window=rolling_window)
            z = z_score(volumes[i], baseline, std, min_std=min_std)

            if (
                z >= z_threshold
                and bucket.observed_volume >= min_volume
                and bucket.severity_weighted_volume >= min_severity_score
            ):
                incidents.append(
                    AnomalyResult(
                        series_key=series_key,
                        granularity=bucket.granularity,
                        cluster_id=bucket.cluster_id,
                        aspect_code=bucket.aspect_code,
                        week_start=bucket.week_start,
                        observed_volume=bucket.observed_volume,
                        baseline_volume=round(baseline, 4),
                        rolling_std=round(std, 4) if std is not None else 0.0,
                        z_score=round(z, 4),
                        severity_score=bucket.severity_weighted_volume,
                        unique_review_count=bucket.unique_review_count,
                        avg_rating=bucket.avg_rating,
                        example_quotes=list(bucket.example_quotes),
                    )
                )
    incidents.sort(key=lambda r: (-r.z_score, -r.severity_score, r.week_start))
    return incidents


def build_incident_summary(result: AnomalyResult, *, cluster_label: str | None) -> str:
    """Deterministic one-paragraph incident summary (no LLM)."""
    aspect = result.aspect_code.replace("_", " ").capitalize()
    if result.granularity == GRANULARITY_CLUSTER and cluster_label:
        where = f"in cluster '{cluster_label}'"
    else:
        where = "overall"
    summary = (
        f"{aspect} complaints {where} spiked to {result.observed_volume} "
        f"mentions during week {result.week_start.isoformat()}, above EWMA "
        f"baseline {result.baseline_volume:.1f} (z={result.z_score:.1f}, "
        f"severity score {result.severity_score:.0f})."
    )
    if result.example_quotes:
        summary += f" Representative quote: '{result.example_quotes[0]}'"
    return summary
