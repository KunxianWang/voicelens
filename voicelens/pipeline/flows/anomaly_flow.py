"""EWMA anomaly-detection flow over clustered negative mentions (M4B).

Builds weekly volume series from the ``review_cluster`` table, runs
:mod:`voicelens.analytics.anomaly`, and persists flagged weeks into the
``incident`` table.

Two granularities: ``cluster`` (one series per issue cluster — the
preferred, finest view) and ``aspect`` (one series per aspect — the
fallback when cluster-level weekly counts are too sparse to baseline).

Re-runs are idempotent per ``(aspect_version, provider, model_name,
granularity)``: prior incidents for that scope are cleared first, and
each incident is written through :func:`upsert_incident`, which is
itself idempotent on ``(granularity, cluster_id, aspect_code,
week_start)``.

CLI::

    python -m voicelens.pipeline.flows.anomaly_flow \
      --aspect-version v2 --provider anthropic --model claude-opus-4.6 \
      --granularity cluster --min-history-weeks 3 --z-threshold 2.0 \
      --min-volume 3

Out of scope for M4B: RAG, LangGraph, Streamlit, real LLM calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from typing import Any

from prefect import flow, get_run_logger
from sqlalchemy import and_, delete, select
from sqlalchemy.orm import Session

from voicelens.analytics.anomaly import (
    GRANULARITY_ASPECT,
    GRANULARITY_CLUSTER,
    AnomalyResult,
    MentionEvent,
    aggregate_weekly,
    build_incident_summary,
    detect_anomalies,
)
from voicelens.db.engine import get_session
from voicelens.db.models import (
    AspectMention,
    AspectOntology,
    Cluster,
    Incident,
    Review,
    ReviewCluster,
)
from voicelens.nlp.absa import LATEST_ONTOLOGY_VERSION


def _quote_map(
    session: Session, *, aspect_version: str, model_name: str
) -> dict[tuple[int, str], str]:
    """Map ``(review_id, aspect_code) -> evidence_quote`` for negative mentions.

    Looked up separately from the main event query so the join cannot
    multiply ``review_cluster`` rows when a review has several mentions.
    """
    rows = session.execute(
        select(
            AspectMention.review_id,
            AspectOntology.code,
            AspectMention.evidence_quote,
        )
        .join(AspectOntology, AspectOntology.id == AspectMention.aspect_id)
        .where(
            and_(
                AspectMention.aspect_version == aspect_version,
                AspectMention.model_name == model_name,
                AspectMention.sentiment == "negative",
            )
        )
    )
    quotes: dict[tuple[int, str], str] = {}
    for review_id, code, quote in rows:
        key = (int(review_id), code)
        if key not in quotes and quote:
            quotes[key] = quote
    return quotes


def build_mention_events(
    session: Session,
    *,
    aspect_version: str = "v2",
    provider: str = "anthropic",
    model_name: str = "claude-opus-4.6",
) -> tuple[list[MentionEvent], dict[int, str]]:
    """Load clustered negative mentions as ``MentionEvent``s + a label map.

    Only ``review_cluster`` rows whose ``cluster`` is in the requested
    ``(aspect_version, provider, model_name)`` scope are returned.
    """
    rows = session.execute(
        select(
            ReviewCluster.review_id,
            ReviewCluster.cluster_id,
            ReviewCluster.aspect_code,
            ReviewCluster.severity,
            Review.posted_at,
            Review.rating,
        )
        .join(Cluster, Cluster.id == ReviewCluster.cluster_id)
        .join(Review, Review.id == ReviewCluster.review_id)
        .where(
            and_(
                Cluster.aspect_version == aspect_version,
                Cluster.provider == provider,
                Cluster.model_name == model_name,
            )
        )
    ).all()

    quotes = _quote_map(session, aspect_version=aspect_version, model_name=model_name)
    events: list[MentionEvent] = []
    for review_id, cluster_id, aspect_code, severity, posted_at, rating in rows:
        events.append(
            MentionEvent(
                review_id=int(review_id),
                cluster_id=int(cluster_id) if cluster_id is not None else None,
                aspect_code=aspect_code,
                severity=severity,
                posted_at=posted_at,
                rating=int(rating) if rating is not None else None,
                evidence_quote=quotes.get((int(review_id), aspect_code), ""),
            )
        )

    label_rows = session.execute(
        select(Cluster.id, Cluster.label).where(
            and_(
                Cluster.aspect_version == aspect_version,
                Cluster.provider == provider,
                Cluster.model_name == model_name,
            )
        )
    )
    cluster_labels = {int(cid): label for cid, label in label_rows}
    return events, cluster_labels


def upsert_incident(
    session: Session,
    *,
    run_id: str,
    granularity: str,
    aspect_version: str,
    provider: str,
    model_name: str,
    cluster_id: int | None,
    aspect_code: str,
    week_start: date,
    cluster_label: str | None,
    observed_volume: int,
    baseline_volume: float,
    z_score: float,
    severity_score: float,
    unique_review_count: int,
    avg_rating: float | None,
    example_quotes: list[str],
    summary: str,
) -> Incident:
    """Insert or update one incident, idempotent on its natural key.

    The natural key is ``(aspect_version, provider, model_name,
    granularity, cluster_id, aspect_code, week_start)`` — calling this
    twice with the same key updates the existing row instead of
    duplicating it.
    """
    existing = session.scalar(
        select(Incident).where(
            and_(
                Incident.aspect_version == aspect_version,
                Incident.provider == provider,
                Incident.model_name == model_name,
                Incident.granularity == granularity,
                Incident.cluster_id.is_(cluster_id)
                if cluster_id is None
                else Incident.cluster_id == cluster_id,
                Incident.aspect_code == aspect_code,
                Incident.week_start == week_start,
            )
        )
    )
    if existing is None:
        incident = Incident(
            run_id=run_id,
            granularity=granularity,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model_name,
            cluster_id=cluster_id,
            aspect_code=aspect_code,
            week_start=week_start,
            status=Incident.STATUS_OPEN,
        )
        session.add(incident)
    else:
        incident = existing

    incident.run_id = run_id
    incident.cluster_label = cluster_label
    incident.observed_volume = observed_volume
    incident.baseline_volume = baseline_volume
    incident.z_score = z_score
    incident.severity_score = severity_score
    incident.unique_review_count = unique_review_count
    incident.avg_rating = avg_rating
    incident.example_quotes = list(example_quotes)
    incident.summary = summary
    session.flush()
    return incident


def _clear_incidents(
    session: Session,
    *,
    aspect_version: str,
    provider: str,
    model_name: str,
    granularity: str,
) -> int:
    """Delete incidents from earlier runs of this scope + granularity."""
    ids = list(
        session.execute(
            select(Incident.id).where(
                and_(
                    Incident.aspect_version == aspect_version,
                    Incident.provider == provider,
                    Incident.model_name == model_name,
                    Incident.granularity == granularity,
                )
            )
        ).scalars()
    )
    if ids:
        session.execute(delete(Incident).where(Incident.id.in_(ids)))
    return len(ids)


def _top_incidents(results: list[AnomalyResult], *, n: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results[:n]:
        rows.append(
            {
                "aspect_code": result.aspect_code,
                "cluster_id": result.cluster_id,
                "week_start": result.week_start.isoformat(),
                "observed_volume": result.observed_volume,
                "baseline_volume": result.baseline_volume,
                "z_score": result.z_score,
                "severity_score": result.severity_score,
                "summary": result.summary,
            }
        )
    return rows


@flow(name="anomaly_flow")
def anomaly_flow(
    *,
    aspect_version: str = LATEST_ONTOLOGY_VERSION,
    provider: str = "anthropic",
    model: str = "claude-opus-4.6",
    granularity: str = GRANULARITY_CLUSTER,
    min_history_weeks: int = 3,
    z_threshold: float = 2.0,
    min_volume: int = 3,
    min_severity_score: float = 5.0,
    ewma_span: int = 4,
    rolling_window: int | None = 8,
    min_std: float = 1.0,
) -> dict[str, Any]:
    """Detect weekly issue spikes and persist them as incidents."""
    try:
        logger = get_run_logger()
    except Exception:  # pragma: no cover - prefect ctx missing
        logger = logging.getLogger("anomaly_flow")

    if granularity not in (GRANULARITY_CLUSTER, GRANULARITY_ASPECT):
        raise ValueError(
            f"granularity must be 'cluster' or 'aspect', got {granularity!r}"
        )

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "aspect_version": aspect_version,
        "provider": provider,
        "model": model,
        "granularity": granularity,
        "z_threshold": z_threshold,
        "min_volume": min_volume,
        "total_series": 0,
        "total_weeks": 0,
        "incidents_detected": 0,
        "incidents_by_aspect": {},
        "top_incidents": [],
    }

    with get_session() as session:
        events, cluster_labels = build_mention_events(
            session,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model,
        )
        buckets = aggregate_weekly(events, granularity=granularity)
        summary["total_series"] = len(buckets)
        summary["total_weeks"] = sum(len(v) for v in buckets.values())

        if not buckets:
            logger.info(
                "anomaly_flow: no clustered mentions for "
                "aspect_version=%s provider=%s model=%s",
                aspect_version, provider, model,
            )
            return summary

        results = detect_anomalies(
            buckets,
            min_history_weeks=min_history_weeks,
            z_threshold=z_threshold,
            min_volume=min_volume,
            min_severity_score=min_severity_score,
            ewma_span=ewma_span,
            rolling_window=rolling_window,
            min_std=min_std,
        )
        for result in results:
            result.summary = build_incident_summary(
                result, cluster_label=cluster_labels.get(result.cluster_id)
            )

        _clear_incidents(
            session,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model,
            granularity=granularity,
        )
        for result in results:
            upsert_incident(
                session,
                run_id=run_id,
                granularity=granularity,
                aspect_version=aspect_version,
                provider=provider,
                model_name=model,
                cluster_id=result.cluster_id,
                aspect_code=result.aspect_code,
                week_start=result.week_start,
                cluster_label=cluster_labels.get(result.cluster_id),
                observed_volume=result.observed_volume,
                baseline_volume=result.baseline_volume,
                z_score=result.z_score,
                severity_score=result.severity_score,
                unique_review_count=result.unique_review_count,
                avg_rating=result.avg_rating,
                example_quotes=result.example_quotes,
                summary=result.summary,
            )

    by_aspect: dict[str, int] = {}
    for result in results:
        by_aspect[result.aspect_code] = by_aspect.get(result.aspect_code, 0) + 1
    summary["incidents_detected"] = len(results)
    summary["incidents_by_aspect"] = dict(sorted(by_aspect.items()))
    summary["top_incidents"] = _top_incidents(results)

    logger.info(
        "anomaly_flow: %d series, %d series-weeks -> %d incidents",
        summary["total_series"], summary["total_weeks"], len(results),
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect emerging issue anomalies from weekly cluster volumes."
    )
    parser.add_argument("--aspect-version", default=LATEST_ONTOLOGY_VERSION)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-opus-4.6")
    parser.add_argument(
        "--granularity", choices=(GRANULARITY_CLUSTER, GRANULARITY_ASPECT),
        default=GRANULARITY_CLUSTER,
    )
    parser.add_argument("--min-history-weeks", type=int, default=3)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--min-volume", type=int, default=3)
    parser.add_argument("--min-severity-score", type=float, default=5.0)
    parser.add_argument("--ewma-span", type=int, default=4)
    parser.add_argument(
        "--rolling-window", type=int, default=8,
        help="Weeks of history for rolling std; 0 = expanding (all history)",
    )
    parser.add_argument("--min-std", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = anomaly_flow(
        aspect_version=args.aspect_version,
        provider=args.provider,
        model=args.model,
        granularity=args.granularity,
        min_history_weeks=args.min_history_weeks,
        z_threshold=args.z_threshold,
        min_volume=args.min_volume,
        min_severity_score=args.min_severity_score,
        ewma_span=args.ewma_span,
        rolling_window=args.rolling_window or None,
        min_std=args.min_std,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
