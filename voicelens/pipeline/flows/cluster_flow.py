"""Topic-clustering flow over negative ABSA mentions (M4A).

Pulls negative aspect mentions from Postgres, clusters them per aspect
(:mod:`voicelens.analytics.clustering`), and persists the result into
the ``cluster`` / ``review_cluster`` tables.

Re-runs are idempotent at the ``(aspect_version, provider, model_name)``
grain: every prior cluster for that tuple is deleted before the new run
is written, so ``cluster_stats`` always reflects exactly one run.

CLI::

    python -m voicelens.pipeline.flows.cluster_flow \
      --aspect-version v2 --provider anthropic \
      --model claude-opus-4.6 --min-cluster-size 5

Out of scope for M4A: anomaly detection (M4B), RAG, LangGraph, Streamlit.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

from prefect import flow, get_run_logger
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from voicelens.analytics.clustering import (
    ClusterResult,
    build_clustering_records,
    cluster_records,
)
from voicelens.db.engine import get_session
from voicelens.db.models import Cluster, ReviewCluster
from voicelens.nlp.absa import LATEST_ONTOLOGY_VERSION


def _clear_previous_clusters(
    session: Session, *, aspect_version: str, provider: str, model_name: str
) -> int:
    """Delete clusters (and their memberships) from earlier runs of this tuple."""
    cluster_ids = list(
        session.execute(
            select(Cluster.id).where(
                Cluster.aspect_version == aspect_version,
                Cluster.provider == provider,
                Cluster.model_name == model_name,
            )
        ).scalars()
    )
    if not cluster_ids:
        return 0
    session.execute(
        delete(ReviewCluster).where(ReviewCluster.cluster_id.in_(cluster_ids))
    )
    session.execute(delete(Cluster).where(Cluster.id.in_(cluster_ids)))
    return len(cluster_ids)


def persist_clusters(
    session: Session,
    results: list[ClusterResult],
    *,
    run_id: str,
    aspect_version: str,
    provider: str,
    model_name: str,
) -> dict[str, int]:
    """Write cluster + review_cluster rows; replace any prior run's rows."""
    removed = _clear_previous_clusters(
        session, aspect_version=aspect_version, provider=provider, model_name=model_name
    )
    n_clusters = 0
    n_memberships = 0
    for result in results:
        cluster = Cluster(
            run_id=run_id,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model_name,
            aspect_code=result.aspect_code,
            label=result.label,
            algorithm=result.algorithm,
            size=result.size,
            severity_weighted_size=result.severity_weighted_size,
            topic_keywords=list(result.keywords),
            representative_review_ids=list(result.representative_review_ids),
            representative_quotes=list(result.representative_quotes),
        )
        session.add(cluster)
        session.flush()  # assign cluster.id
        n_clusters += 1
        seen: set[int] = set()
        for member in result.members:
            if member.review_id in seen:
                continue  # one review can't join the same cluster twice
            seen.add(member.review_id)
            session.add(
                ReviewCluster(
                    cluster_id=cluster.id,
                    review_id=member.review_id,
                    aspect_code=result.aspect_code,
                    severity=member.severity,
                )
            )
            n_memberships += 1
    return {
        "removed_prior_clusters": removed,
        "clusters_written": n_clusters,
        "memberships_written": n_memberships,
    }


@flow(name="cluster_flow")
def cluster_flow(
    *,
    aspect_version: str = LATEST_ONTOLOGY_VERSION,
    provider: str = "anthropic",
    model: str = "claude-opus-4.6",
    min_cluster_size: int = 5,
    max_clusters_per_aspect: int = 8,
    limit: int | None = None,
    random_state: int = 42,
    algorithm: str = "auto",
) -> dict[str, Any]:
    """Cluster negative ABSA mentions and persist the clusters."""
    try:
        logger = get_run_logger()
    except Exception:  # pragma: no cover - prefect ctx missing
        logger = logging.getLogger("cluster_flow")

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "aspect_version": aspect_version,
        "provider": provider,
        "model": model,
        "min_cluster_size": min_cluster_size,
        "negative_mentions": 0,
        "clusters": 0,
        "clustered_mentions": 0,
        "clusters_by_aspect": {},
        "skipped_aspects": [],
        "algorithm": None,
    }

    with get_session() as session:
        records = build_clustering_records(
            session,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model,
            limit=limit,
        )
        summary["negative_mentions"] = len(records)
        if not records:
            logger.info(
                "cluster_flow: no negative mentions for "
                "aspect_version=%s provider=%s model=%s",
                aspect_version, provider, model,
            )
            return summary

        results = cluster_records(
            records,
            min_cluster_size=min_cluster_size,
            max_clusters_per_aspect=max_clusters_per_aspect,
            random_state=random_state,
            algorithm=algorithm,
        )

        # Which aspects had mentions but produced no cluster (below floor).
        aspects_in = {r.aspect_code for r in records}
        aspects_out = {c.aspect_code for c in results}
        summary["skipped_aspects"] = sorted(aspects_in - aspects_out)

        persisted = persist_clusters(
            session,
            results,
            run_id=run_id,
            aspect_version=aspect_version,
            provider=provider,
            model_name=model,
        )

    by_aspect: dict[str, int] = {}
    for result in results:
        by_aspect[result.aspect_code] = by_aspect.get(result.aspect_code, 0) + 1
    summary["clusters"] = len(results)
    summary["clustered_mentions"] = persisted["memberships_written"]
    summary["clusters_by_aspect"] = dict(sorted(by_aspect.items()))
    summary["removed_prior_clusters"] = persisted["removed_prior_clusters"]
    summary["algorithm"] = results[0].algorithm if results else None

    logger.info(
        "cluster_flow: %d negative mentions -> %d clusters across %d aspects",
        len(records), len(results), len(by_aspect),
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster negative ABSA mentions into emerging issue topics."
    )
    parser.add_argument("--aspect-version", default=LATEST_ONTOLOGY_VERSION)
    parser.add_argument("--provider", default="anthropic", help="ABSA provider")
    parser.add_argument("--model", default="claude-opus-4.6", help="ABSA model_name")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--max-clusters-per-aspect", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--algorithm",
        choices=("auto", "tfidf_kmeans", "bertopic"),
        default="auto",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = cluster_flow(
        aspect_version=args.aspect_version,
        provider=args.provider,
        model=args.model,
        min_cluster_size=args.min_cluster_size,
        max_clusters_per_aspect=args.max_clusters_per_aspect,
        limit=args.limit,
        random_state=args.random_state,
        algorithm=args.algorithm,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
