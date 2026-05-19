"""Streaming review counts by metadata-derived candidate brand."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voicelens.config import AMAZON_REVIEWS_PATH, BRAND_CANDIDATES, EXTERNAL_DATA_DIR, PROJECT_ROOT
from voicelens.ingest.brand_scan import (
    _compile_aliases,
    _match_candidate_brands,
    choose_best_brand,
    resolve_metadata_path,
    stream_jsonl,
)

DEFAULT_REVIEW_NAMES = (
    "Electronics.jsonl.gz",
    "Electronics.jsonl",
    "reviews_Electronics.jsonl.gz",
    "reviews_Electronics.jsonl",
)


@dataclass
class BrandReviewStats:
    candidate_brand: str
    matched_reviews: int = 0
    parent_asins: set[str] = field(default_factory=set)
    asins: set[str] = field(default_factory=set)
    rating_counts: Counter[int] = field(default_factory=Counter)
    rating_sum: float = 0.0
    rating_n: int = 0
    first_review_date: str | None = None
    last_review_date: str | None = None

    def add_review(self, row: dict[str, Any]) -> None:
        self.matched_reviews += 1
        parent_asin = _clean_str(row.get("parent_asin"))
        asin = _clean_str(row.get("asin"))
        if parent_asin:
            self.parent_asins.add(parent_asin)
        if asin:
            self.asins.add(asin)

        rating = _coerce_rating(row.get("rating"))
        if rating is not None:
            self.rating_counts[rating] += 1
            self.rating_sum += rating
            self.rating_n += 1

        date = _coerce_review_date(row.get("timestamp") or row.get("posted_at"))
        if date:
            if self.first_review_date is None or date < self.first_review_date:
                self.first_review_date = date
            if self.last_review_date is None or date > self.last_review_date:
                self.last_review_date = date

    @property
    def avg_rating(self) -> float | None:
        if self.rating_n == 0:
            return None
        return round(self.rating_sum / self.rating_n, 4)


@dataclass
class ReviewScanResult:
    metadata_rows_scanned: int
    review_rows_scanned: int
    stats: dict[str, BrandReviewStats]
    asin_review_counts: Counter[str] = field(default_factory=Counter)
    brand_asin_review_counts: dict[str, Counter[str]] = field(default_factory=dict)


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_rating(value: Any) -> int | None:
    try:
        rating = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if rating < 1 or rating > 5:
        return None
    return rating


def _coerce_review_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=UTC).date().isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).date().isoformat()
        except ValueError:
            return None
    return None


def build_brand_maps(
    metadata_path: Path,
    *,
    candidates: tuple[str, ...] | list[str] | None = None,
    metadata_limit: int | None = None,
) -> tuple[int, dict[str, str], dict[str, str]]:
    candidate_tuple = tuple(candidates or BRAND_CANDIDATES)
    parent_to_brand: dict[str, str] = {}
    asin_to_brand: dict[str, str] = {}
    compiled = _compile_aliases(candidate_tuple)
    scanned = 0
    for row in stream_jsonl(metadata_path):
        scanned += 1
        matches = _match_candidate_brands(row, compiled)
        brand = choose_best_brand(matches, candidate_tuple)
        if brand is not None:
            parent_asin = _clean_str(row.get("parent_asin"))
            asin = _clean_str(row.get("asin"))
            if parent_asin:
                parent_to_brand[parent_asin] = brand
            if asin:
                asin_to_brand[asin] = brand
        if metadata_limit is not None and scanned >= metadata_limit:
            break
    return scanned, parent_to_brand, asin_to_brand


def resolve_reviews_path(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if AMAZON_REVIEWS_PATH.exists():
        return AMAZON_REVIEWS_PATH
    for name in DEFAULT_REVIEW_NAMES:
        candidate = EXTERNAL_DATA_DIR / name
        if candidate.exists():
            return candidate
    searched = ", ".join(f"data/{name}" for name in DEFAULT_REVIEW_NAMES)
    raise FileNotFoundError(
        "Amazon reviews file not found. Set AMAZON_REVIEWS_PATH, pass --reviews, "
        f"or put one of these files under data/: {searched}."
    )


def scan_brand_reviews(
    metadata_path: Path,
    reviews_path: Path,
    *,
    candidates: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
    metadata_limit: int | None = None,
) -> ReviewScanResult:
    candidate_tuple = tuple(candidates or BRAND_CANDIDATES)
    metadata_rows_scanned, parent_to_brand, asin_to_brand = build_brand_maps(
        metadata_path,
        candidates=candidate_tuple,
        metadata_limit=metadata_limit,
    )
    stats = {brand: BrandReviewStats(candidate_brand=brand) for brand in candidate_tuple}
    asin_review_counts: Counter[str] = Counter()
    brand_asin_review_counts: dict[str, Counter[str]] = {brand: Counter() for brand in candidate_tuple}
    review_rows_scanned = 0
    for row in _stream_jsonl(reviews_path):
        review_rows_scanned += 1
        parent_asin = _clean_str(row.get("parent_asin"))
        asin = _clean_str(row.get("asin"))
        brand = None
        if parent_asin:
            brand = parent_to_brand.get(parent_asin)
        if brand is None and asin:
            brand = asin_to_brand.get(asin)
        if brand:
            stats[brand].add_review(row)
            asin_key = parent_asin or asin
            if asin_key:
                asin_review_counts[asin_key] += 1
                brand_asin_review_counts[brand][asin_key] += 1
        if limit is not None and review_rows_scanned >= limit:
            break
    return ReviewScanResult(
        metadata_rows_scanned=metadata_rows_scanned,
        review_rows_scanned=review_rows_scanned,
        stats=stats,
        asin_review_counts=asin_review_counts,
        brand_asin_review_counts=brand_asin_review_counts,
    )


def review_count_csv_rows(stats: dict[str, BrandReviewStats]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for brand, item in stats.items():
        rows.append(
            {
                "candidate_brand": brand,
                "matched_reviews": item.matched_reviews,
                "matched_parent_asins": len(item.parent_asins),
                "matched_asins": len(item.asins),
                "avg_rating": "" if item.avg_rating is None else item.avg_rating,
                "rating_1_count": item.rating_counts.get(1, 0),
                "rating_2_count": item.rating_counts.get(2, 0),
                "rating_3_count": item.rating_counts.get(3, 0),
                "rating_4_count": item.rating_counts.get(4, 0),
                "rating_5_count": item.rating_counts.get(5, 0),
                "first_review_date": item.first_review_date or "",
                "last_review_date": item.last_review_date or "",
            }
        )
    return rows


def write_review_counts_csv(path: Path, stats: dict[str, BrandReviewStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = review_count_csv_rows(stats)
    fieldnames = [
        "candidate_brand",
        "matched_reviews",
        "matched_parent_asins",
        "matched_asins",
        "avg_rating",
        "rating_1_count",
        "rating_2_count",
        "rating_3_count",
        "rating_4_count",
        "rating_5_count",
        "first_review_date",
        "last_review_date",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_candidates(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate candidate-brand review counts by joining reviews to metadata."
    )
    parser.add_argument("--metadata", default=None, help="Path to metadata JSONL(.GZ)")
    parser.add_argument("--reviews", default=None, help="Path to reviews JSONL(.GZ)")
    parser.add_argument("--output", default="data/brand_review_counts.csv", help="CSV output path")
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate brands")
    parser.add_argument("--limit", type=int, default=None, help="Max review rows to scan")
    parser.add_argument("--metadata-limit", type=int, default=None, help="Max metadata rows to scan")
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan the full review file. Without this or --limit, scans the first 10000 rows.",
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
    candidates = _parse_candidates(args.candidates)
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
    result = scan_brand_reviews(
        metadata_path,
        reviews_path,
        candidates=candidates,
        limit=limit,
        metadata_limit=metadata_limit,
    )
    output_path = Path(args.output)
    write_review_counts_csv(output_path, result.stats)
    print(f"Scanned metadata rows: {result.metadata_rows_scanned}")
    print(f"Scanned review rows: {result.review_rows_scanned}")
    print(f"Wrote brand review counts: {output_path}")
    for row in review_count_csv_rows(result.stats):
        print(f"{row['candidate_brand']}: {row['matched_reviews']} matched reviews")
    return 0


if __name__ == "__main__":
    sys.exit(main())
