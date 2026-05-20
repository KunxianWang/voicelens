"""Print a summary of the ``cluster`` table after ``make cluster-v2``.

Shows total cluster count, clusters per aspect, the largest clusters by
raw size and by severity-weighted size, and a couple of example
evidence quotes per top cluster — a quick eyeball check that clustering
produced sensible issue groups.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from sqlalchemy import func, select

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import get_session
from voicelens.db.models import Cluster


def collect_cluster_stats(
    *,
    aspect_version: str = "v2",
    provider: str = "anthropic",
    model_name: str = "claude-opus-4.6",
    top_n: int = 5,
) -> dict[str, Any]:
    """Aggregate the cluster table for one ``(version, provider, model)``."""
    out: dict[str, Any] = {
        "aspect_version": aspect_version,
        "provider": provider,
        "model_name": model_name,
    }
    with get_session() as s:
        scope = (
            (Cluster.aspect_version == aspect_version)
            & (Cluster.provider == provider)
            & (Cluster.model_name == model_name)
        )
        clusters = list(
            s.execute(select(Cluster).where(scope)).scalars()
        )
        out["total_clusters"] = len(clusters)
        out["total_clustered_mentions"] = sum(c.size for c in clusters)

        by_aspect_rows = s.execute(
            select(Cluster.aspect_code, func.count())
            .where(scope)
            .group_by(Cluster.aspect_code)
            .order_by(Cluster.aspect_code)
        ).all()
        out["clusters_by_aspect"] = {code: int(n) for code, n in by_aspect_rows}

        run_ids = sorted({c.run_id for c in clusters})
        out["run_ids"] = run_ids

        def _row(c: Cluster) -> dict[str, Any]:
            return {
                "aspect_code": c.aspect_code,
                "label": c.label,
                "size": c.size,
                "severity_weighted_size": round(c.severity_weighted_size, 2),
                "keywords": list(c.topic_keywords or [])[:6],
                "example_quotes": list(c.representative_quotes or [])[:2],
            }

        out["top_by_size"] = [
            _row(c) for c in sorted(clusters, key=lambda c: (-c.size, c.label))[:top_n]
        ]
        out["top_by_severity_weighted_size"] = [
            _row(c)
            for c in sorted(
                clusters, key=lambda c: (-c.severity_weighted_size, c.label)
            )[:top_n]
        ]
    return out


def _print_report(stats: dict[str, Any]) -> None:
    print("=== Cluster stats ===")
    print(
        f"scope                    : {stats['aspect_version']} / "
        f"{stats['provider']} / {stats['model_name']}"
    )
    print(f"total clusters           : {stats['total_clusters']}")
    print(f"total clustered mentions : {stats['total_clustered_mentions']}")

    print("\nclusters by aspect:")
    if stats["clusters_by_aspect"]:
        for code, n in stats["clusters_by_aspect"].items():
            print(f"  {code:16} {n}")
    else:
        print("  (none — run `make cluster-v2` first)")

    def _dump(title: str, rows: list[dict[str, Any]]) -> None:
        print(f"\n{title}:")
        if not rows:
            print("  (none)")
            return
        for i, row in enumerate(rows, start=1):
            print(
                f"  {i}. [{row['aspect_code']}] \"{row['label']}\"  "
                f"size={row['size']} sev_weighted={row['severity_weighted_size']}"
            )
            if row["keywords"]:
                print(f"      keywords: {', '.join(row['keywords'])}")
            for quote in row["example_quotes"]:
                print(f"      quote: {quote}")

    _dump(f"top {len(stats['top_by_size'])} clusters by size", stats["top_by_size"])
    _dump(
        f"top {len(stats['top_by_severity_weighted_size'])} clusters "
        "by severity-weighted size",
        stats["top_by_severity_weighted_size"],
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise the cluster table.")
    parser.add_argument("--aspect-version", default="v2")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-opus-4.6")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stats = collect_cluster_stats(
        aspect_version=args.aspect_version,
        provider=args.provider,
        model_name=args.model,
        top_n=args.top_n,
    )
    _print_report(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
