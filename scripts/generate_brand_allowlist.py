from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_metric_csv(path: Path, metric: str) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    values: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand = (row.get("candidate_brand") or "").strip()
            raw_value = row.get(metric) or "0"
            if not brand:
                continue
            try:
                values[brand] = int(float(raw_value))
            except ValueError:
                values[brand] = 0
    return values


def generate_allowlist(
    inventory_path: Path,
    review_counts_path: Path,
    *,
    min_items: int = 20,
    min_reviews: int = 1000,
) -> dict[str, object]:
    item_counts = _read_metric_csv(inventory_path, "matched_items")
    review_counts = _read_metric_csv(review_counts_path, "matched_reviews")
    brands = sorted(item_counts.keys(), key=lambda b: (-review_counts.get(b, 0), b))

    included: list[str] = []
    dropped: dict[str, str] = {}
    for brand in brands:
        items = item_counts.get(brand, 0)
        reviews = review_counts.get(brand, 0)
        if items >= min_items and reviews >= min_reviews:
            included.append(brand)
            continue
        if items <= 0:
            reason = "not found in metadata"
        elif reviews <= 0:
            reason = "no matched reviews after metadata join"
        elif items < min_items and reviews < min_reviews:
            reason = f"below thresholds: {items} items < {min_items} and {reviews} reviews < {min_reviews}"
        elif items < min_items:
            reason = f"too few matched metadata items: {items} < {min_items}"
        else:
            reason = f"too few matched reviews: {reviews} < {min_reviews}"
        if brand == "Soundcore" and "Anker" in included and reviews < min_reviews:
            reason = "too few matched reviews or merged under Anker"
        dropped[brand] = reason

    return {"brands": included, "dropped": dropped}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resolved runtime BRAND_ALLOWLIST from scan outputs.")
    parser.add_argument("--inventory", default="data/brand_inventory.csv", help="Input brand inventory CSV")
    parser.add_argument("--review-counts", default="data/brand_review_counts.csv", help="Input review counts CSV")
    parser.add_argument("--output", default="data/resolved_brand_allowlist.json", help="Output JSON path")
    parser.add_argument("--min-items", type=int, default=20, help="Minimum matched metadata items")
    parser.add_argument("--min-reviews", type=int, default=1000, help="Minimum matched reviews")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = generate_allowlist(
            Path(args.inventory),
            Path(args.review_counts),
            min_items=args.min_items,
            min_reviews=args.min_reviews,
        )
    except FileNotFoundError as exc:
        print(
            f"ERROR: {exc}. Run make scan-amazon-brands and make scan-amazon-brand-reviews first.",
            file=sys.stderr,
        )
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote resolved brand allowlist: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
