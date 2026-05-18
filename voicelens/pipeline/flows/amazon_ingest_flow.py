from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger

from voicelens.config import (
    AMAZON_METADATA_PATH,
    AMAZON_REVIEWS_PATH,
    BRAND_ALLOWLIST,
    INGEST_LIMIT,
)
from voicelens.db.engine import get_session
from voicelens.ingest.amazon_reviews_2023 import (
    iter_amazon_reviews,
    load_asin_brand_map,
)
from voicelens.pipeline.flows.ingest_flow import (
    close_ingest_run,
    open_ingest_run,
    write_dq_events,
)
from voicelens.pipeline.tasks.dq import run_dq
from voicelens.pipeline.tasks.normalize import load_reviews


@flow(name="amazon_ingest_flow")
def amazon_ingest_flow(
    input_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    brands: tuple[str, ...] | None = None,
    limit: int | None = None,
    source: str = "amazon_reviews_2023",
) -> dict[str, Any]:
    logger = get_run_logger()

    input_path = Path(input_path) if input_path else AMAZON_REVIEWS_PATH
    if metadata_path is None and AMAZON_METADATA_PATH is not None:
        metadata_path = AMAZON_METADATA_PATH
    elif metadata_path is not None:
        metadata_path = Path(metadata_path)

    allowlist: tuple[str, ...] = tuple(brands) if brands else BRAND_ALLOWLIST
    effective_limit = limit if limit is not None else INGEST_LIMIT

    if not input_path.exists():
        raise FileNotFoundError(
            f"\nAmazon Reviews file not found at: {input_path}\n"
            f"Either drop the dataset there or pass --input <path>.\n"
            f"See data/README.md for download instructions.\n"
        )

    asin_to_brand: dict[str, str] | None = None
    if metadata_path:
        if not Path(metadata_path).exists():
            raise FileNotFoundError(
                f"Metadata file not found at: {metadata_path}\n"
                f"Either drop it there or unset AMAZON_METADATA_PATH."
            )
        logger.info(f"Loading metadata from {metadata_path}")
        asin_to_brand = load_asin_brand_map(Path(metadata_path), allowlist=allowlist)

    logger.info(
        f"Ingesting {input_path} | brands={list(allowlist)} | limit={effective_limit} | "
        f"metadata={'yes' if asin_to_brand else 'no'}"
    )

    input_rows = _count_lines_cheap(Path(input_path))
    matched: list[dict[str, Any]] = list(
        iter_amazon_reviews(
            input_path,
            asin_to_brand=asin_to_brand,
            brand_allowlist=allowlist,
            limit=effective_limit,
        )
    )

    outcome = run_dq(matched)
    logger.info(
        f"DQ summary: matched={len(matched)}, passed={outcome.passed_rows}, "
        f"failed={outcome.failed_rows}, pass_rate={outcome.pass_rate:.3f}"
    )

    with get_session() as session:
        run = open_ingest_run(session, source)
        run_id = run.id
        loaded = load_reviews(session, outcome.valid_rows, ingest_run_id=run_id)
        write_dq_events(session, run_id, outcome)
        close_ingest_run(session, run, n_rows=loaded, status="completed")

    brand_counts = Counter(r["brand"] for r in outcome.valid_rows if r.get("brand"))
    asin_counts = Counter(r["asin"] for r in outcome.valid_rows if r.get("asin"))

    summary = {
        "input_rows": input_rows,
        "matched_rows": len(matched),
        "passed_rows": outcome.passed_rows,
        "failed_rows": outcome.failed_rows,
        "loaded_reviews": loaded,
        "dq_pass_rate": round(outcome.pass_rate, 4),
        "per_check_failed": dict(outcome.per_check_failed),
        "brand_counts": dict(brand_counts.most_common()),
        "top_asin_counts": dict(asin_counts.most_common(10)),
    }
    logger.info(f"Ingest summary: {summary}")
    return summary


def _count_lines_cheap(path: Path) -> int:
    """Count non-empty lines in a JSONL or JSONL.GZ file. O(file)."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            if line.strip():
                n += 1
    return n


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Amazon Reviews 2023 subset into Postgres.")
    p.add_argument("--input", default=None, help="Path to reviews JSONL or JSONL.GZ")
    p.add_argument("--metadata", default=None, help="Optional path to metadata JSONL or JSONL.GZ")
    p.add_argument("--brands", default=None, help="Comma-separated brand allowlist (overrides .env)")
    p.add_argument("--limit", type=int, default=None, help="Max rows to ingest after brand filter")
    p.add_argument("--source", default="amazon_reviews_2023", help="ingest_run.source tag")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    brands = tuple(b.strip() for b in args.brands.split(",")) if args.brands else None
    started = datetime.utcnow()
    summary = amazon_ingest_flow(
        input_path=args.input,
        metadata_path=args.metadata,
        brands=brands,
        limit=args.limit,
        source=args.source,
    )
    elapsed = (datetime.utcnow() - started).total_seconds()
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"\nElapsed: {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
