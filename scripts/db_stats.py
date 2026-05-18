"""Print a quick summary of what's in the VoiceLens DB.

Run after `make ingest-sample` or `make ingest-amazon-*` to confirm
how many reviews loaded, brand distribution, DQ failures, etc.
"""
from __future__ import annotations

import sys

from sqlalchemy import func, select

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import get_session
from voicelens.db.models import Brand, DQEvent, IngestRun, Review, Sku


def collect_stats() -> dict:
    out: dict = {}
    with get_session() as s:
        out["total_reviews"] = s.scalar(select(func.count()).select_from(Review)) or 0
        out["total_brands"] = s.scalar(select(func.count()).select_from(Brand)) or 0
        out["total_skus"] = s.scalar(select(func.count()).select_from(Sku)) or 0

        rows = s.execute(
            select(Brand.name, func.count(Review.id))
            .join(Sku, Sku.brand_id == Brand.id)
            .join(Review, Review.sku_id == Sku.id)
            .group_by(Brand.name)
            .order_by(func.count(Review.id).desc())
        ).all()
        out["reviews_by_brand"] = {name: int(n) for name, n in rows}

        rows = s.execute(
            select(Review.rating, func.count())
            .group_by(Review.rating)
            .order_by(Review.rating)
        ).all()
        out["reviews_by_rating"] = {int(rating): int(n) for rating, n in rows}

        rows = s.execute(
            select(DQEvent.check_name, func.sum(DQEvent.n_failed))
            .group_by(DQEvent.check_name)
            .order_by(func.sum(DQEvent.n_failed).desc())
        ).all()
        out["dq_failures_by_check"] = {name: int(n or 0) for name, n in rows}

        latest = s.execute(
            select(IngestRun).order_by(IngestRun.started_at.desc()).limit(1)
        ).scalar_one_or_none()
        if latest is not None:
            out["latest_ingest_run"] = {
                "id": latest.id,
                "source": latest.source,
                "status": latest.status,
                "started_at": latest.started_at.isoformat() if latest.started_at else None,
                "completed_at": latest.completed_at.isoformat() if latest.completed_at else None,
                "n_rows": latest.n_rows,
            }
        else:
            out["latest_ingest_run"] = None
    return out


def _print(stats: dict) -> None:
    print("== VoiceLens DB stats ==")
    print(f"  total_reviews : {stats['total_reviews']}")
    print(f"  total_brands  : {stats['total_brands']}")
    print(f"  total_skus    : {stats['total_skus']}")

    print("\n-- reviews by brand --")
    for name, n in stats["reviews_by_brand"].items():
        print(f"  {name:<16} {n}")

    print("\n-- reviews by rating --")
    for rating, n in stats["reviews_by_rating"].items():
        print(f"  {rating} stars  {n}")

    print("\n-- DQ failures (sum across runs) --")
    if stats["dq_failures_by_check"]:
        for name, n in stats["dq_failures_by_check"].items():
            print(f"  {name:<24} {n}")
    else:
        print("  (no dq_event rows)")

    print("\n-- latest ingest_run --")
    latest = stats["latest_ingest_run"]
    if latest is None:
        print("  (no runs)")
    else:
        for k, v in latest.items():
            print(f"  {k:<13} {v}")


def main() -> int:
    stats = collect_stats()
    _print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
