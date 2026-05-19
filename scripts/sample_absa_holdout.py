"""Deterministically sample reviews into a hand-label scaffold.

Writes ``data/labeling/absa_holdout_seed.jsonl``. Each line is a single
review with empty ``gold_aspects`` ready for manual annotation. Sampling
is:

- deterministic (BLAKE2b-seeded score on review.id + source_id), so the
  same DB state always produces the same sample
- balanced across brands when possible (round-robin top-N brands)
- balanced across ratings 1..5 when possible (round-robin within a brand)

No labels are written; the file is a scaffold. Sampling does not require
the ``aspect_ontology`` to be seeded or the ABSA flow to have run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path

from sqlalchemy import select

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import get_session
from voicelens.db.models import Brand, IngestRun, Review, Sku

DEFAULT_OUTPUT_PATH = Path("data/labeling/absa_holdout_seed.jsonl")
DEFAULT_TARGET_COUNT = 200
DEFAULT_SEED = 42


def _score(seed: int, source_id: str, review_id: int) -> int:
    payload = f"{seed}|{source_id}|{review_id}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")


def sample_holdout(
    *,
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
    source: str | None = None,
) -> list[dict[str, object]]:
    with get_session() as s:
        stmt = (
            select(
                Review.id,
                Review.source,
                Review.source_id,
                Review.rating,
                Review.posted_at,
                Review.text_raw,
                Sku.asin,
                Brand.name.label("brand"),
            )
            .join(Sku, Sku.id == Review.sku_id)
            .join(Brand, Brand.id == Sku.brand_id)
        )
        if source is not None:
            stmt = stmt.where(
                Review.ingest_run_id.in_(
                    select(IngestRun.id).where(IngestRun.source == source)
                )
            )
        rows = [
            {
                "review_id": int(row.id),
                "source": row.source,
                "source_id": row.source_id,
                "rating": int(row.rating),
                "posted_at": row.posted_at.isoformat() if row.posted_at else None,
                "text_raw": row.text_raw,
                "asin": row.asin,
                "brand": row.brand,
            }
            for row in s.execute(stmt)
        ]

    if not rows:
        return []

    by_brand_rating: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["brand"]), int(row["rating"]))
        by_brand_rating[key].append(row)
    for bucket in by_brand_rating.values():
        bucket.sort(key=lambda r: _score(seed, str(r["source_id"]), int(r["review_id"])))

    brand_counts: dict[str, int] = defaultdict(int)
    for (brand, _rating), bucket in by_brand_rating.items():
        brand_counts[brand] += len(bucket)
    brands_sorted = sorted(brand_counts.keys(), key=lambda b: (-brand_counts[b], b))

    ratings_order = (1, 2, 3, 4, 5)
    cursors: dict[tuple[str, int], int] = defaultdict(int)
    selected: list[dict[str, object]] = []
    seen_ids: set[int] = set()

    rounds = 0
    while len(selected) < target_count and rounds < target_count * 10:
        progressed = False
        for brand in brands_sorted:
            if len(selected) >= target_count:
                break
            for rating in ratings_order:
                if len(selected) >= target_count:
                    break
                bucket = by_brand_rating.get((brand, rating), [])
                idx = cursors[(brand, rating)]
                if idx >= len(bucket):
                    continue
                row = bucket[idx]
                cursors[(brand, rating)] = idx + 1
                rid = int(row["review_id"])
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                selected.append(row)
                progressed = True
        if not progressed:
            break
        rounds += 1

    selected.sort(key=lambda r: (str(r["brand"]), int(r["rating"]), int(r["review_id"])))
    return selected


def write_seed(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            payload = dict(row)
            payload["gold_aspects"] = []
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _print_summary(rows: list[dict[str, object]], path: Path) -> None:
    print(f"Wrote {len(rows)} holdout rows to {path}")
    if not rows:
        print("(Database has no reviews matching the source filter.)")
        return
    by_brand: dict[str, int] = defaultdict(int)
    by_rating: dict[int, int] = defaultdict(int)
    for r in rows:
        by_brand[str(r["brand"])] += 1
        by_rating[int(r["rating"])] += 1
    print("-- by brand --")
    for brand, n in sorted(by_brand.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {brand:<16} {n}")
    print("-- by rating --")
    for rating in (1, 2, 3, 4, 5):
        print(f"  {rating} stars  {by_rating.get(rating, 0)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample ABSA holdout seed for hand-labeling.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", default=None, help="Filter by ingest_run.source")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = sample_holdout(target_count=args.count, seed=args.seed, source=args.source)
    output_path = Path(args.output)
    write_seed(output_path, rows)
    _print_summary(rows, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
