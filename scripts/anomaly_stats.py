"""Print a summary of the ``incident`` table after ``make anomaly-v2``.

Shows total incidents, incidents per aspect, and the top incidents by
severity score — week, cluster label, observed vs baseline volume,
z-score and a couple of example quotes each.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from sqlalchemy import func, select

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import get_session
from voicelens.db.models import Incident


def collect_anomaly_stats(
    *,
    aspect_version: str = "v2",
    provider: str = "anthropic",
    model_name: str = "claude-opus-4.6",
    granularity: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Aggregate the incident table for one ``(version, provider, model)``."""
    out: dict[str, Any] = {
        "aspect_version": aspect_version,
        "provider": provider,
        "model_name": model_name,
        "granularity": granularity or "all",
    }
    with get_session() as s:
        scope = (
            (Incident.aspect_version == aspect_version)
            & (Incident.provider == provider)
            & (Incident.model_name == model_name)
        )
        if granularity:
            scope = scope & (Incident.granularity == granularity)

        incidents = list(s.execute(select(Incident).where(scope)).scalars())
        out["total_incidents"] = len(incidents)

        by_aspect_rows = s.execute(
            select(Incident.aspect_code, func.count())
            .where(scope)
            .group_by(Incident.aspect_code)
            .order_by(Incident.aspect_code)
        ).all()
        out["incidents_by_aspect"] = {code: int(n) for code, n in by_aspect_rows}

        by_gran_rows = s.execute(
            select(Incident.granularity, func.count())
            .where(scope)
            .group_by(Incident.granularity)
        ).all()
        out["incidents_by_granularity"] = {g: int(n) for g, n in by_gran_rows}

        top = sorted(
            incidents,
            key=lambda i: (-i.severity_score, -i.z_score, i.week_start),
        )[:top_n]
        out["top_incidents"] = [
            {
                "aspect_code": i.aspect_code,
                "cluster_label": i.cluster_label,
                "week_start": i.week_start.isoformat(),
                "observed_volume": i.observed_volume,
                "baseline_volume": round(i.baseline_volume, 2),
                "z_score": round(i.z_score, 2),
                "severity_score": round(i.severity_score, 2),
                "example_quotes": list(i.example_quotes or [])[:2],
                "summary": i.summary,
            }
            for i in top
        ]
    return out


def _print_report(stats: dict[str, Any]) -> None:
    print("=== Anomaly / incident stats ===")
    print(
        f"scope                : {stats['aspect_version']} / "
        f"{stats['provider']} / {stats['model_name']} "
        f"(granularity={stats['granularity']})"
    )
    print(f"total incidents      : {stats['total_incidents']}")
    print(f"by granularity       : {stats['incidents_by_granularity'] or '(none)'}")

    print("\nincidents by aspect:")
    if stats["incidents_by_aspect"]:
        for code, n in stats["incidents_by_aspect"].items():
            print(f"  {code:16} {n}")
    else:
        print("  (none — run `make anomaly-v2` first)")

    print(f"\ntop {len(stats['top_incidents'])} incidents by severity score:")
    if not stats["top_incidents"]:
        print("  (none)")
    for i, row in enumerate(stats["top_incidents"], start=1):
        label = row["cluster_label"] or "(aspect-level)"
        print(
            f"  {i}. [{row['aspect_code']}] \"{label}\"  week={row['week_start']}"
        )
        print(
            f"      observed={row['observed_volume']} "
            f"baseline={row['baseline_volume']} z={row['z_score']} "
            f"severity={row['severity_score']}"
        )
        for quote in row["example_quotes"]:
            print(f"      quote: {quote}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise the incident table.")
    parser.add_argument("--aspect-version", default="v2")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-opus-4.6")
    parser.add_argument(
        "--granularity", choices=("cluster", "aspect"), default=None,
        help="Filter to one granularity (default: show all)",
    )
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stats = collect_anomaly_stats(
        aspect_version=args.aspect_version,
        provider=args.provider,
        model_name=args.model,
        granularity=args.granularity,
        top_n=args.top_n,
    )
    _print_report(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
