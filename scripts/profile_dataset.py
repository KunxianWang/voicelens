from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from voicelens.config import BRAND_CANDIDATES
from voicelens.ingest.brand_review_scan import (
    resolve_reviews_path,
    review_count_csv_rows,
    scan_brand_reviews,
)
from voicelens.ingest.brand_scan import (
    inventory_csv_rows,
    resolve_metadata_path,
    scan_metadata,
)


def profile_dataset(
    metadata_path: Path,
    reviews_path: Path,
    *,
    candidates: tuple[str, ...] | None = None,
    min_items: int = 20,
    min_reviews: int = 1000,
    limit: int | None = None,
    metadata_limit: int | None = None,
) -> dict[str, Any]:
    candidate_tuple = candidates or BRAND_CANDIDATES
    metadata_rows, inventory = scan_metadata(
        metadata_path,
        candidates=candidate_tuple,
        limit=metadata_limit,
    )
    review_result = scan_brand_reviews(
        metadata_path,
        reviews_path,
        candidates=candidate_tuple,
        limit=limit,
        metadata_limit=metadata_limit,
    )

    inventory_rows = inventory_csv_rows(inventory)
    review_rows = review_count_csv_rows(review_result.stats)
    recommended = [
        row["candidate_brand"]
        for row in sorted(review_rows, key=lambda r: int(r["matched_reviews"]), reverse=True)
        if _as_int(row["matched_reviews"]) >= min_reviews
        and _inventory_items(inventory_rows, str(row["candidate_brand"])) >= min_items
    ]
    dropped = _dropped_reasons(inventory_rows, review_rows, recommended, min_items, min_reviews)
    recommended_asin_counts: dict[str, int] = {}
    for brand in recommended:
        for asin, count in review_result.brand_asin_review_counts.get(brand, {}).items():
            recommended_asin_counts[asin] = recommended_asin_counts.get(asin, 0) + count
    top_asins = [
        {"asin": asin, "review_count": count}
        for asin, count in sorted(
            recommended_asin_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:20]
    ]
    warnings = [
        f"{brand}: {reason}"
        for brand, reason in dropped.items()
        if "not found" in reason or "too few" in reason or "below thresholds" in reason
    ]
    return {
        "total_scanned_metadata_rows": metadata_rows,
        "total_scanned_review_rows": review_result.review_rows_scanned,
        "candidate_brand_inventory": inventory_rows,
        "candidate_brand_review_counts": review_rows,
        "recommended_mvp_brands": recommended,
        "dropped": dropped,
        "top_20_asins_by_review_count_within_recommended_brands": top_asins,
        "warnings": warnings,
    }


def _as_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except ValueError:
        return 0


def _inventory_items(rows: list[dict[str, object]], brand: str) -> int:
    for row in rows:
        if row.get("candidate_brand") == brand:
            return _as_int(row.get("matched_items"))
    return 0


def _dropped_reasons(
    inventory_rows: list[dict[str, object]],
    review_rows: list[dict[str, object]],
    recommended: list[str],
    min_items: int,
    min_reviews: int,
) -> dict[str, str]:
    item_counts = {str(r["candidate_brand"]): _as_int(r["matched_items"]) for r in inventory_rows}
    review_counts = {str(r["candidate_brand"]): _as_int(r["matched_reviews"]) for r in review_rows}
    dropped: dict[str, str] = {}
    for brand, items in item_counts.items():
        if brand in recommended:
            continue
        reviews = review_counts.get(brand, 0)
        if items <= 0:
            dropped[brand] = "not found in metadata"
        elif reviews <= 0:
            dropped[brand] = "no matched reviews after metadata join"
        elif items < min_items and reviews < min_reviews:
            dropped[brand] = (
                f"below thresholds: {items} items < {min_items} and "
                f"{reviews} reviews < {min_reviews}"
            )
        elif items < min_items:
            dropped[brand] = f"too few matched metadata items: {items} < {min_items}"
        else:
            dropped[brand] = f"too few matched reviews: {reviews} < {min_reviews}"
    return dropped


def _print_profile(profile: dict[str, Any]) -> None:
    print(f"Total scanned metadata rows: {profile['total_scanned_metadata_rows']}")
    print(f"Total scanned review rows: {profile['total_scanned_review_rows']}")
    print("\nCandidate brand inventory:")
    for row in profile["candidate_brand_inventory"]:
        print(
            f"  {row['candidate_brand']}: {row['matched_items']} items, "
            f"{row['matched_parent_asins']} parent ASINs"
        )
    print("\nCandidate brand review counts:")
    for row in profile["candidate_brand_review_counts"]:
        print(
            f"  {row['candidate_brand']}: {row['matched_reviews']} reviews, "
            f"avg_rating={row['avg_rating']}"
        )
    print("\nRecommended MVP brands:")
    print("  " + (", ".join(profile["recommended_mvp_brands"]) or "(none)"))
    if profile["warnings"]:
        print("\nWarnings:")
        for warning in profile["warnings"]:
            print(f"  - {warning}")
    print("\nTop 20 ASINs by review count within recommended brands:")
    for item in profile["top_20_asins_by_review_count_within_recommended_brands"]:
        print(f"  {item['asin']}: {item['review_count']}")


def _parse_candidates(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile real Amazon Reviews 2023 Electronics inputs.")
    parser.add_argument("--metadata", default=None, help="Path to metadata JSONL(.GZ)")
    parser.add_argument("--reviews", default=None, help="Path to reviews JSONL(.GZ)")
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate brands")
    parser.add_argument("--min-items", type=int, default=20, help="Minimum matched metadata items")
    parser.add_argument("--min-reviews", type=int, default=1000, help="Minimum matched reviews")
    parser.add_argument("--limit", type=int, default=None, help="Max review rows to scan")
    parser.add_argument("--metadata-limit", type=int, default=None, help="Max metadata rows to scan")
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan the full review file. Without this or --limit, scans the first 10000 rows.",
    )
    parser.add_argument(
        "--write",
        nargs="?",
        const="data/dataset_profile.json",
        default=None,
        help="Optionally write JSON profile, defaulting to data/dataset_profile.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        metadata_path = resolve_metadata_path(args.metadata)
        reviews_path = resolve_reviews_path(args.reviews)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    limit = args.limit
    metadata_limit = args.metadata_limit
    if limit is None and not args.full_scan:
        limit = 10_000
        print("No --full-scan or --limit provided; scanning first 10000 review rows.", file=sys.stderr)
    if metadata_limit is None and not args.full_scan:
        metadata_limit = max(100_000, limit or 0)
        print(
            f"No --full-scan or --metadata-limit provided; scanning first {metadata_limit} metadata rows.",
            file=sys.stderr,
        )
    profile = profile_dataset(
        metadata_path,
        reviews_path,
        candidates=_parse_candidates(args.candidates),
        min_items=args.min_items,
        min_reviews=args.min_reviews,
        limit=limit,
        metadata_limit=metadata_limit,
    )
    _print_profile(profile)
    if args.write:
        output_path = Path(args.write)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\nWrote dataset profile: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
