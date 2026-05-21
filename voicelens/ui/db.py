"""Read-only Postgres query helpers for the dashboard.

Every function opens its own short-lived session via
:func:`voicelens.db.engine.get_session` and returns plain Python
(``dict`` / ``list[dict]``) so the Streamlit pages — and the tests —
never touch the ORM directly. Nothing here writes; nothing here calls
an LLM.

The 1k ABSA subset is small enough that un-cached queries are instant;
pages may still wrap calls in ``st.cache_data`` for snappier reruns.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from voicelens.db.engine import get_session
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Cluster,
    DQEvent,
    Incident,
    IngestRun,
    Review,
    ReviewCluster,
    Sku,
)

# ABSA scope used everywhere the dashboard reads pipeline output. The 1k
# subset was processed under exactly this triple (see README_DEV M2/M4).
DEFAULT_ASPECT_VERSION = "v2"
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-4.6"

_ABSA_STATUSES = ("success", "no_mentions", "invalid", "failed")


# ---- small helpers ------------------------------------------------------


def _scalar(session: Session, stmt: Any) -> int:
    """Run a ``select(func.count())``-style statement, coerce to int."""
    return int(session.scalar(stmt) or 0)


def _negative_mention_filter(
    *,
    brand: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    severity: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
) -> list[Any]:
    """Build the WHERE predicates shared by the ABSA-page queries."""
    clauses: list[Any] = []
    if brand:
        clauses.append(Brand.name == brand)
    if aspect:
        clauses.append(AspectOntology.code == aspect)
    if sentiment:
        clauses.append(AspectMention.sentiment == sentiment)
    if severity:
        clauses.append(AspectMention.severity == severity)
    if rating_min is not None:
        clauses.append(Review.rating >= rating_min)
    if rating_max is not None:
        clauses.append(Review.rating <= rating_max)
    return clauses


# ---- filter option lists ------------------------------------------------


def list_brands() -> list[str]:
    """Brands that actually have at least one review."""
    with get_session() as s:
        rows = s.execute(
            select(Brand.name)
            .join(Sku, Sku.brand_id == Brand.id)
            .join(Review, Review.sku_id == Sku.id)
            .distinct()
            .order_by(Brand.name)
        ).scalars()
        return [r for r in rows]


def list_aspects() -> list[str]:
    """Aspect codes that appear in at least one mention."""
    with get_session() as s:
        rows = s.execute(
            select(AspectOntology.code)
            .join(AspectMention, AspectMention.aspect_id == AspectOntology.id)
            .distinct()
            .order_by(AspectOntology.code)
        ).scalars()
        return [r for r in rows]


# ---- overview page ------------------------------------------------------


def get_overview_stats() -> dict[str, int]:
    """Headline counts for the overview page."""
    with get_session() as s:
        return {
            "total_reviews": _scalar(s, select(func.count()).select_from(Review)),
            "total_brands": _scalar(s, select(func.count()).select_from(Brand)),
            "total_skus": _scalar(s, select(func.count()).select_from(Sku)),
            "absa_processed_reviews": _scalar(
                s,
                select(func.count(func.distinct(ABSAReviewStatus.review_id))),
            ),
            "total_aspect_mentions": _scalar(
                s, select(func.count()).select_from(AspectMention)
            ),
            "total_clusters": _scalar(s, select(func.count()).select_from(Cluster)),
            "total_incidents": _scalar(s, select(func.count()).select_from(Incident)),
            "total_ingest_runs": _scalar(
                s, select(func.count()).select_from(IngestRun)
            ),
        }


def get_reviews_by_brand() -> list[dict[str, Any]]:
    """Review counts per brand, most reviews first."""
    with get_session() as s:
        rows = s.execute(
            select(Brand.name, func.count(Review.id))
            .join(Sku, Sku.brand_id == Brand.id)
            .join(Review, Review.sku_id == Sku.id)
            .group_by(Brand.name)
            .order_by(func.count(Review.id).desc())
        ).all()
        return [{"brand": name, "reviews": int(n)} for name, n in rows]


def get_absa_status_distribution() -> dict[str, int]:
    """Count of ABSA review-status rows by status (success/.../failed)."""
    with get_session() as s:
        rows = s.execute(
            select(ABSAReviewStatus.status, func.count())
            .group_by(ABSAReviewStatus.status)
        ).all()
    counts = {status: 0 for status in _ABSA_STATUSES}
    for status, n in rows:
        counts[status] = counts.get(status, 0) + int(n)
    return counts


def get_aspect_distribution() -> list[dict[str, Any]]:
    """Mention count per aspect code, largest first."""
    with get_session() as s:
        rows = s.execute(
            select(AspectOntology.code, func.count(AspectMention.id))
            .join(AspectMention, AspectMention.aspect_id == AspectOntology.id)
            .group_by(AspectOntology.code)
            .order_by(func.count(AspectMention.id).desc())
        ).all()
        return [{"aspect_code": code, "mentions": int(n)} for code, n in rows]


def get_sentiment_distribution() -> dict[str, int]:
    """Mention count per sentiment label."""
    with get_session() as s:
        rows = s.execute(
            select(AspectMention.sentiment, func.count())
            .group_by(AspectMention.sentiment)
        ).all()
        return {sentiment: int(n) for sentiment, n in rows}


# ---- ABSA distribution page ---------------------------------------------


def get_aspect_sentiment_matrix(
    *,
    brand: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    severity: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
) -> list[dict[str, Any]]:
    """``(aspect_code, sentiment) -> count`` after applying the filters."""
    clauses = _negative_mention_filter(
        brand=brand, aspect=aspect, sentiment=sentiment, severity=severity,
        rating_min=rating_min, rating_max=rating_max,
    )
    with get_session() as s:
        rows = s.execute(
            select(
                AspectOntology.code,
                AspectMention.sentiment,
                func.count(AspectMention.id),
            )
            .join(AspectMention, AspectMention.aspect_id == AspectOntology.id)
            .join(Review, Review.id == AspectMention.review_id)
            .join(Sku, Sku.id == Review.sku_id)
            .join(Brand, Brand.id == Sku.brand_id)
            .where(and_(*clauses) if clauses else True)
            .group_by(AspectOntology.code, AspectMention.sentiment)
            .order_by(AspectOntology.code, AspectMention.sentiment)
        ).all()
        return [
            {"aspect_code": code, "sentiment": sent, "count": int(n)}
            for code, sent, n in rows
        ]


def get_severity_distribution(
    *,
    brand: str | None = None,
    aspect: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
) -> dict[str, int]:
    """Severity counts for *negative* mentions only (low/medium/high)."""
    clauses = _negative_mention_filter(
        brand=brand, aspect=aspect, sentiment="negative",
        rating_min=rating_min, rating_max=rating_max,
    )
    with get_session() as s:
        rows = s.execute(
            select(AspectMention.severity, func.count(AspectMention.id))
            .join(AspectOntology, AspectOntology.id == AspectMention.aspect_id)
            .join(Review, Review.id == AspectMention.review_id)
            .join(Sku, Sku.id == Review.sku_id)
            .join(Brand, Brand.id == Sku.brand_id)
            .where(and_(*clauses))
            .group_by(AspectMention.severity)
        ).all()
    counts = {"low": 0, "medium": 0, "high": 0}
    for severity, n in rows:
        key = severity or "(none)"
        counts[key] = counts.get(key, 0) + int(n)
    return counts


def get_negative_examples(
    *,
    brand: str | None = None,
    aspect: str | None = None,
    severity: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Negative aspect mentions with their evidence quote, newest first."""
    clauses = _negative_mention_filter(
        brand=brand, aspect=aspect, sentiment="negative", severity=severity,
        rating_min=rating_min, rating_max=rating_max,
    )
    with get_session() as s:
        rows = s.execute(
            select(
                Review.id,
                Brand.name,
                Sku.asin,
                Review.rating,
                Review.posted_at,
                AspectOntology.code,
                AspectMention.severity,
                AspectMention.evidence_quote,
                Review.text_raw,
            )
            .join(AspectMention, AspectMention.review_id == Review.id)
            .join(AspectOntology, AspectOntology.id == AspectMention.aspect_id)
            .join(Sku, Sku.id == Review.sku_id)
            .join(Brand, Brand.id == Sku.brand_id)
            .where(and_(*clauses))
            .order_by(Review.posted_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "review_id": int(rid),
                "brand": brand_name,
                "asin": asin,
                "rating": int(rating) if rating is not None else None,
                "posted_at": posted_at.isoformat() if posted_at else "",
                "aspect_code": code,
                "severity": severity_val,
                "evidence_quote": quote,
                "text_snippet": _snippet(text_raw),
            }
            for (
                rid, brand_name, asin, rating, posted_at, code, severity_val,
                quote, text_raw,
            ) in rows
        ]


# ---- clusters page ------------------------------------------------------


def get_clusters(
    *,
    aspect: str | None = None,
    aspect_version: str = DEFAULT_ASPECT_VERSION,
    provider: str = DEFAULT_PROVIDER,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Issue clusters in scope, ranked by severity-weighted size."""
    clauses = [
        Cluster.aspect_version == aspect_version,
        Cluster.provider == provider,
        Cluster.model_name == model_name,
    ]
    if aspect:
        clauses.append(Cluster.aspect_code == aspect)
    with get_session() as s:
        rows = s.execute(
            select(Cluster).where(and_(*clauses))
        ).scalars()
        clusters = sorted(
            rows,
            key=lambda c: (-(c.severity_weighted_size or 0.0), -(c.size or 0)),
        )
        return [
            {
                "cluster_id": c.id,
                "aspect_code": c.aspect_code,
                "label": c.label,
                "size": int(c.size or 0),
                "severity_weighted_size": round(c.severity_weighted_size or 0.0, 2),
                "algorithm": c.algorithm,
                "topic_keywords": list(c.topic_keywords or []),
                "representative_quotes": list(c.representative_quotes or []),
            }
            for c in clusters
        ]


def get_cluster_members(cluster_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
    """Reviews that belong to one cluster, with their evidence quote."""
    with get_session() as s:
        rows = s.execute(
            select(
                Review.id,
                Brand.name,
                Sku.asin,
                Review.rating,
                Review.posted_at,
                ReviewCluster.aspect_code,
                ReviewCluster.severity,
                AspectMention.evidence_quote,
                Review.text_raw,
            )
            .join(Review, Review.id == ReviewCluster.review_id)
            .join(Sku, Sku.id == Review.sku_id)
            .join(Brand, Brand.id == Sku.brand_id)
            .join(
                AspectOntology,
                AspectOntology.code == ReviewCluster.aspect_code,
                isouter=True,
            )
            .join(
                AspectMention,
                and_(
                    AspectMention.review_id == ReviewCluster.review_id,
                    AspectMention.aspect_id == AspectOntology.id,
                    AspectMention.sentiment == "negative",
                ),
                isouter=True,
            )
            .where(ReviewCluster.cluster_id == cluster_id)
            .order_by(Review.posted_at.desc())
            .limit(limit)
        ).all()
        seen: set[int] = set()
        members: list[dict[str, Any]] = []
        for (
            rid, brand_name, asin, rating, posted_at, aspect_code, severity,
            quote, text_raw,
        ) in rows:
            rid = int(rid)
            if rid in seen:  # the outer join can repeat a review
                continue
            seen.add(rid)
            members.append(
                {
                    "review_id": rid,
                    "brand": brand_name,
                    "asin": asin,
                    "rating": int(rating) if rating is not None else None,
                    "posted_at": posted_at.isoformat() if posted_at else "",
                    "aspect_code": aspect_code,
                    "severity": severity,
                    "evidence_quote": quote or "",
                    "text_snippet": _snippet(text_raw),
                }
            )
        return members


# ---- incidents page -----------------------------------------------------


def get_incidents(
    *,
    aspect: str | None = None,
    granularity: str | None = None,
    aspect_version: str = DEFAULT_ASPECT_VERSION,
    provider: str = DEFAULT_PROVIDER,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Anomaly incidents in scope, ranked by severity score."""
    clauses = [
        Incident.aspect_version == aspect_version,
        Incident.provider == provider,
        Incident.model_name == model_name,
    ]
    if aspect:
        clauses.append(Incident.aspect_code == aspect)
    if granularity:
        clauses.append(Incident.granularity == granularity)
    with get_session() as s:
        rows = s.execute(select(Incident).where(and_(*clauses))).scalars()
        incidents = sorted(
            rows,
            key=lambda i: (-(i.severity_score or 0.0), -(i.z_score or 0.0), i.week_start),
        )
        return [
            {
                "incident_id": i.id,
                "granularity": i.granularity,
                "aspect_code": i.aspect_code,
                "cluster_id": i.cluster_id,
                "cluster_label": i.cluster_label,
                "week_start": i.week_start.isoformat(),
                "observed_volume": int(i.observed_volume or 0),
                "baseline_volume": round(i.baseline_volume or 0.0, 2),
                "z_score": round(i.z_score or 0.0, 2),
                "severity_score": round(i.severity_score or 0.0, 2),
                "unique_review_count": int(i.unique_review_count or 0),
                "avg_rating": round(i.avg_rating, 2) if i.avg_rating is not None else None,
                "example_quotes": list(i.example_quotes or []),
                "summary": i.summary or "",
                "status": i.status,
            }
            for i in incidents
        ]


def list_incident_aspects(
    *,
    aspect_version: str = DEFAULT_ASPECT_VERSION,
    provider: str = DEFAULT_PROVIDER,
    model_name: str = DEFAULT_MODEL,
) -> list[str]:
    """Distinct aspect codes that have at least one incident."""
    with get_session() as s:
        rows = s.execute(
            select(Incident.aspect_code)
            .where(
                and_(
                    Incident.aspect_version == aspect_version,
                    Incident.provider == provider,
                    Incident.model_name == model_name,
                )
            )
            .distinct()
            .order_by(Incident.aspect_code)
        ).scalars()
        return [r for r in rows]


def list_incident_granularities(
    *,
    aspect_version: str = DEFAULT_ASPECT_VERSION,
    provider: str = DEFAULT_PROVIDER,
    model_name: str = DEFAULT_MODEL,
) -> list[str]:
    """Distinct granularities (cluster/aspect) present in the incident table."""
    with get_session() as s:
        rows = s.execute(
            select(Incident.granularity)
            .where(
                and_(
                    Incident.aspect_version == aspect_version,
                    Incident.provider == provider,
                    Incident.model_name == model_name,
                )
            )
            .distinct()
            .order_by(Incident.granularity)
        ).scalars()
        return [r for r in rows]


# ---- data-quality page --------------------------------------------------


def get_ingest_runs(*, limit: int = 50) -> list[dict[str, Any]]:
    """Ingest-run rows, newest first."""
    with get_session() as s:
        rows = s.execute(
            select(IngestRun).order_by(IngestRun.started_at.desc()).limit(limit)
        ).scalars()
        return [
            {
                "run_id": r.id,
                "source": r.source,
                "status": r.status,
                "n_rows": int(r.n_rows or 0),
                "started_at": r.started_at.isoformat() if r.started_at else "",
                "completed_at": r.completed_at.isoformat() if r.completed_at else "",
            }
            for r in rows
        ]


def get_dq_summary() -> dict[str, Any]:
    """Aggregate the ``dq_event`` table: totals + failures by reason.

    Each ``check_name`` *is* a failure reason (``text_too_short``,
    ``duplicate_review_id``, …); ``reason_codes`` only carries example
    rows. ``failures_by_reason`` therefore sums ``n_failed`` per check.
    """
    with get_session() as s:
        events = list(s.execute(select(DQEvent)).scalars())
    total_rows = sum(int(e.n_rows or 0) for e in events)
    total_failed = sum(int(e.n_failed or 0) for e in events)
    by_check: dict[str, dict[str, int]] = {}
    for e in events:
        bucket = by_check.setdefault(e.check_name, {"rows": 0, "failed": 0})
        bucket["rows"] += int(e.n_rows or 0)
        bucket["failed"] += int(e.n_failed or 0)
    by_reason = {
        name: vals["failed"]
        for name, vals in by_check.items()
        if vals["failed"] > 0
    }
    return {
        "total_events": len(events),
        "total_rows_checked": total_rows,
        "total_failed": total_failed,
        "total_passed": max(total_rows - total_failed, 0),
        "by_check": by_check,
        "failures_by_reason": dict(
            sorted(by_reason.items(), key=lambda kv: -kv[1])
        ),
    }


def get_pipeline_coverage() -> dict[str, Any]:
    """ABSA reliability + coverage rates for the data-quality page."""
    with get_session() as s:
        total_reviews = _scalar(s, select(func.count()).select_from(Review))
        status_rows = s.execute(
            select(ABSAReviewStatus.status, func.count())
            .group_by(ABSAReviewStatus.status)
        ).all()
        processed = _scalar(
            s, select(func.count(func.distinct(ABSAReviewStatus.review_id)))
        )
        reviews_with_mentions = _scalar(
            s, select(func.count(func.distinct(AspectMention.review_id)))
        )
        total_mentions = _scalar(s, select(func.count()).select_from(AspectMention))

    status_counts = {status: 0 for status in _ABSA_STATUSES}
    for status, n in status_rows:
        status_counts[status] = status_counts.get(status, 0) + int(n)
    status_total = sum(status_counts.values()) or 1

    return {
        "total_reviews": total_reviews,
        "absa_processed_reviews": processed,
        "reviews_with_mentions": reviews_with_mentions,
        "total_mentions": total_mentions,
        "status_counts": status_counts,
        "processed_coverage_rate": _rate(processed, total_reviews),
        "mention_coverage_rate": _rate(reviews_with_mentions, processed),
        "absa_failed_rate": _rate(status_counts["failed"], status_total),
        "absa_invalid_rate": _rate(status_counts["invalid"], status_total),
        "absa_success_rate": _rate(status_counts["success"], status_total),
    }


# ---- formatting helpers -------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    """Safe ratio, rounded to 4 dp; 0.0 when the denominator is 0."""
    if not denominator:
        return 0.0
    return round(numerator / denominator, 4)


def _snippet(text: str | None, max_chars: int = 240) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
