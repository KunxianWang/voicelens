from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import get_session
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Review,
    Sku,
)

DEFAULT_HOLDOUT = Path("data/labeling/absa_holdout_seed.jsonl")
DEFAULT_OUTPUT = Path("data/labeling/absa_holdout_predictions.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_predictions(
    *,
    holdout_path: Path = DEFAULT_HOLDOUT,
    output_path: Path = DEFAULT_OUTPUT,
    provider: str = "openai",
    model_name: str | None = None,
    aspect_version: str = "v1",
) -> list[dict[str, Any]]:
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"Holdout seed is missing: {holdout_path}. Run scripts/sample_absa_holdout.py first."
        )
    model = model_name or os.getenv("ABSA_MODEL") or (
        "gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-latest"
    )
    holdout_rows = _read_jsonl(holdout_path)
    review_ids = [int(row["review_id"]) for row in holdout_rows if row.get("review_id") is not None]
    by_review_id: dict[int, dict[str, Any]] = {}
    with get_session() as s:
        review_rows = s.execute(
            select(
                Review.id,
                Review.source_id,
                Review.rating,
                Review.text_raw,
                Sku.asin,
                Brand.name.label("brand"),
            )
            .join(Sku, Sku.id == Review.sku_id)
            .join(Brand, Brand.id == Sku.brand_id)
            .where(Review.id.in_(review_ids))
        ).all()
        review_info = {int(row.id): row for row in review_rows}

        mention_rows = s.execute(
            select(
                AspectMention.review_id,
                AspectOntology.code,
                AspectMention.sentiment,
                AspectMention.severity,
                AspectMention.evidence_quote,
            )
            .join(AspectOntology, AspectOntology.id == AspectMention.aspect_id)
            .where(
                AspectMention.review_id.in_(review_ids),
                AspectMention.model_name == model,
                AspectMention.aspect_version == aspect_version,
            )
        ).all()
        mentions_by_review: dict[int, list[dict[str, Any]]] = {rid: [] for rid in review_ids}
        for row in mention_rows:
            mentions_by_review[int(row.review_id)].append(
                {
                    "aspect_code": row.code,
                    "sentiment": row.sentiment,
                    "severity": row.severity,
                    "evidence_quote": row.evidence_quote,
                }
            )

        statuses = s.execute(
            select(ABSAReviewStatus).where(
                ABSAReviewStatus.review_id.in_(review_ids),
                ABSAReviewStatus.provider == provider,
                ABSAReviewStatus.model_name == model,
                ABSAReviewStatus.aspect_version == aspect_version,
            )
        ).scalars()
        status_by_review = {int(row.review_id): row for row in statuses}

    for seed_row in holdout_rows:
        rid = int(seed_row["review_id"])
        info = review_info.get(rid)
        status = status_by_review.get(rid)
        by_review_id[rid] = {
            "review_id": rid,
            "source_id": info.source_id if info else seed_row.get("source_id"),
            "brand": info.brand if info else seed_row.get("brand"),
            "asin": info.asin if info else seed_row.get("asin"),
            "rating": int(info.rating) if info else seed_row.get("rating"),
            "text_raw": info.text_raw if info else seed_row.get("text_raw"),
            "gold_aspects": seed_row.get("gold_aspects", []),
            "predicted_aspects": mentions_by_review.get(rid, []),
            "validation_errors": status.error_codes_json if status else None,
        }

    rows = [by_review_id[int(row["review_id"])] for row in holdout_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ABSA predictions for labeled holdout review.")
    parser.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--aspect-version", default="v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        rows = export_predictions(
            holdout_path=Path(args.holdout),
            output_path=Path(args.output),
            provider=args.provider,
            model_name=args.model,
            aspect_version=args.aspect_version,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} prediction rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
