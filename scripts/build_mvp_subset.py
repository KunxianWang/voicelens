from __future__ import annotations

import argparse
import gzip
import heapq
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path
from typing import Any

from voicelens.ingest.brand_review_scan import (
    _clean_str,
    _coerce_rating,
    _open_text,
    build_brand_maps,
    resolve_reviews_path,
)
from voicelens.ingest.brand_scan import resolve_metadata_path

DEFAULT_TARGET_COUNTS: dict[str, int] = {
    "Anker": 80_000,
    "Bose": 60_000,
    "JBL": 50_000,
    "UGREEN": 30_000,
    "Soundcore": 30_000,
    "RAVPower": 2_000,
    "Aukey": 1_500,
}


def load_resolved_brands(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(
            f"Resolved brand allowlist not found: {path}. "
            "Run python scripts/generate_brand_allowlist.py first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    brands = payload.get("brands")
    if not isinstance(brands, list) or not all(isinstance(b, str) for b in brands):
        raise ValueError(f"Resolved allowlist must contain a string list at key 'brands': {path}")
    return tuple(brands)


def _stream_review_rows(path: Path):
    with _open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield line_no, row


def _coerce_year(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000.0
        try:
            return str(datetime.fromtimestamp(ts, tz=UTC).year)
        except (OSError, OverflowError, ValueError):
            return "unknown"
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            return str(datetime.fromisoformat(text).year)
        except ValueError:
            return "unknown"
    return "unknown"


def _rating_key(value: Any) -> str:
    rating = _coerce_rating(value)
    return str(rating) if rating is not None else "unknown"


def _row_brand(row: dict[str, Any], parent_to_brand: dict[str, str], asin_to_brand: dict[str, str]) -> str | None:
    parent_asin = _clean_str(row.get("parent_asin"))
    asin = _clean_str(row.get("asin"))
    if parent_asin and parent_asin in parent_to_brand:
        return parent_to_brand[parent_asin]
    if asin and asin in asin_to_brand:
        return asin_to_brand[asin]
    return None


def _stratum(row: dict[str, Any], brand: str) -> tuple[str, str, str]:
    return brand, _rating_key(row.get("rating")), _coerce_year(row.get("timestamp") or row.get("posted_at"))


def _largest_remainder_quotas(
    stratum_counts: Counter[tuple[str, str, str]],
    brand_counts: Counter[str],
    target_counts: dict[str, int],
) -> dict[tuple[str, str, str], int]:
    quotas: dict[tuple[str, str, str], int] = {}
    by_brand: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in stratum_counts:
        by_brand[key[0]].append(key)

    for brand, keys in by_brand.items():
        available = brand_counts[brand]
        target = min(target_counts.get(brand, 0), available)
        if target <= 0:
            continue
        if target >= available:
            for key in keys:
                quotas[key] = stratum_counts[key]
            continue

        assigned = 0
        remainders: list[tuple[float, int, tuple[str, str, str]]] = []
        for key in keys:
            raw = stratum_counts[key] * target / available
            base = int(raw)
            quotas[key] = base
            assigned += base
            remainders.append((raw - base, stratum_counts[key], key))

        remaining = target - assigned
        for _, _, key in sorted(remainders, key=lambda item: (-item[0], -item[1], item[2]))[:remaining]:
            quotas[key] += 1
    return quotas


def _selection_score(seed: int, line_no: int, brand: str, row: dict[str, Any]) -> int:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    key = f"{seed}|{line_no}|{brand}|{payload}".encode("utf-8", errors="replace")
    return int.from_bytes(blake2b(key, digest_size=8).digest(), "big")


def _push_selected(
    heap: list[tuple[int, int, dict[str, Any]]],
    quota: int,
    score: int,
    line_no: int,
    row: dict[str, Any],
) -> None:
    item = (-score, -line_no, row)
    if len(heap) < quota:
        heapq.heappush(heap, item)
        return
    worst_score = -heap[0][0]
    worst_line = -heap[0][1]
    if (score, line_no) < (worst_score, worst_line):
        heapq.heapreplace(heap, item)


def build_mvp_subset(
    *,
    metadata_path: Path,
    reviews_path: Path,
    allowlist_path: Path,
    output_path: Path,
    profile_path: Path,
    target_counts: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    brands = load_resolved_brands(allowlist_path)
    targets = dict(target_counts or DEFAULT_TARGET_COUNTS)
    targets = {brand: targets.get(brand, 0) for brand in brands}

    metadata_rows, parent_to_brand, asin_to_brand = build_brand_maps(
        metadata_path,
        candidates=brands,
    )

    brand_counts: Counter[str] = Counter()
    stratum_counts: Counter[tuple[str, str, str]] = Counter()
    input_rows = 0
    matched_rows = 0
    for _, row in _stream_review_rows(reviews_path):
        input_rows += 1
        brand = _row_brand(row, parent_to_brand, asin_to_brand)
        if brand is None or brand not in targets:
            continue
        matched_rows += 1
        brand_counts[brand] += 1
        stratum_counts[_stratum(row, brand)] += 1

    quotas = _largest_remainder_quotas(stratum_counts, brand_counts, targets)
    selected: dict[tuple[str, str, str], list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for line_no, row in _stream_review_rows(reviews_path):
        brand = _row_brand(row, parent_to_brand, asin_to_brand)
        if brand is None or brand not in targets:
            continue
        key = _stratum(row, brand)
        quota = quotas.get(key, 0)
        if quota <= 0:
            continue
        score = _selection_score(seed, line_no, brand, row)
        _push_selected(selected[key], quota, score, line_no, row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual_counts: Counter[str] = Counter()
    rating_distribution: dict[str, Counter[str]] = {brand: Counter() for brand in brands}
    year_distribution: dict[str, Counter[str]] = {brand: Counter() for brand in brands}
    selected_items: list[tuple[str, str, str, int, dict[str, Any]]] = []
    for key, heap in selected.items():
        brand, rating, year = key
        for _neg_score, neg_line_no, row in heap:
            selected_items.append((brand, rating, year, -neg_line_no, row))

    selected_items.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    with open(output_path, "wb") as raw_f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, mtime=0) as gz_f:
            with io.TextIOWrapper(gz_f, encoding="utf-8") as f:
                for brand, rating, year, _, row in selected_items:
                    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    actual_counts[brand] += 1
                    rating_distribution[brand][rating] += 1
                    year_distribution[brand][year] += 1

    profile = {
        "target_counts": targets,
        "actual_counts_by_brand": {brand: actual_counts.get(brand, 0) for brand in brands},
        "rating_distribution_by_brand": {
            brand: dict(sorted(rating_distribution[brand].items())) for brand in brands
        },
        "year_distribution_by_brand": {
            brand: dict(sorted(year_distribution[brand].items())) for brand in brands
        },
        "total_output_reviews": sum(actual_counts.values()),
        "input_paths": {
            "metadata": str(metadata_path),
            "reviews": str(reviews_path),
            "resolved_brand_allowlist": str(allowlist_path),
            "output_reviews": str(output_path),
        },
        "seed": seed,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_scan": {
            "metadata_rows": metadata_rows,
            "review_rows": input_rows,
            "matched_review_rows": matched_rows,
            "available_counts_by_brand": {brand: brand_counts.get(brand, 0) for brand in brands},
        },
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return profile


def _parse_targets(raw: str | None) -> dict[str, int] | None:
    if not raw:
        return None
    targets: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        brand, _, count = part.partition(":")
        if not brand or not count:
            raise ValueError("Targets must be comma-separated BRAND:COUNT pairs.")
        targets[brand.strip()] = int(count)
    return targets


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Amazon Reviews MVP subset.")
    parser.add_argument("--metadata", default=None, help="Path to metadata JSONL(.GZ)")
    parser.add_argument("--reviews", default=None, help="Path to reviews JSONL(.GZ)")
    parser.add_argument("--allowlist", default="data/resolved_brand_allowlist.json")
    parser.add_argument("--output", default="data/amazon_mvp_reviews.jsonl.gz")
    parser.add_argument("--profile-output", default="data/amazon_mvp_subset_profile.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", default=None, help="Optional comma-separated BRAND:COUNT overrides")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        metadata_path = resolve_metadata_path(args.metadata)
        reviews_path = resolve_reviews_path(args.reviews)
        profile = build_mvp_subset(
            metadata_path=metadata_path,
            reviews_path=reviews_path,
            allowlist_path=Path(args.allowlist),
            output_path=Path(args.output),
            profile_path=Path(args.profile_output),
            target_counts=_parse_targets(args.targets),
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote MVP subset: {profile['input_paths']['output_reviews']}")
    print(f"Wrote profile: {args.profile_output}")
    print(f"Total output reviews: {profile['total_output_reviews']}")
    for brand, count in profile["actual_counts_by_brand"].items():
        print(f"{brand}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
