"""Streaming brand inventory scan for Amazon Reviews 2023 metadata."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicelens.config import AMAZON_METADATA_PATH, BRAND_CANDIDATES, EXTERNAL_DATA_DIR, PROJECT_ROOT

BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "Anker": ("anker", "anker innovations", "ankerdirect", "anker direct"),
    "Soundcore": ("soundcore", "soundcore by anker"),
    "Bose": ("bose",),
    "JBL": ("jbl",),
    "UGREEN": ("ugreen",),
    "RAVPower": ("ravpower", "rav power"),
    "Aukey": ("aukey",),
}

FIELD_WEIGHTS: dict[str, int] = {
    "details.Brand": 100,
    "store": 90,
    "details_json": 50,
    "title": 30,
    "categories": 20,
}

DEFAULT_METADATA_NAMES = (
    "raw_meta_Electronics.jsonl.gz",
    "raw_meta_Electronics.jsonl",
    "meta_Electronics.jsonl.gz",
    "meta_Electronics.jsonl",
)


@dataclass
class BrandMatch:
    fields: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)


@dataclass
class BrandInventory:
    candidate_brand: str
    matched_items: int = 0
    parent_asins: set[str] = field(default_factory=set)
    asins: set[str] = field(default_factory=set)
    example_parent_asins: list[str] = field(default_factory=list)
    example_titles: list[str] = field(default_factory=list)
    matched_fields: set[str] = field(default_factory=set)
    confidence_notes: set[str] = field(default_factory=set)

    def add(self, row: dict[str, Any], match: BrandMatch) -> None:
        self.matched_items += 1
        parent_asin = _clean_str(row.get("parent_asin"))
        asin = _clean_str(row.get("asin"))
        title = _clean_str(row.get("title"))
        if parent_asin:
            self.parent_asins.add(parent_asin)
            if len(self.example_parent_asins) < 5 and parent_asin not in self.example_parent_asins:
                self.example_parent_asins.append(parent_asin)
        if asin:
            self.asins.add(asin)
        if title and len(self.example_titles) < 5 and title not in self.example_titles:
            self.example_titles.append(title[:180])
        self.matched_fields.update(match.fields)
        self.confidence_notes.add(confidence_note(match.fields))


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _candidate_list(candidates: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    return tuple(candidates or BRAND_CANDIDATES)


def _compile_aliases(candidates: tuple[str, ...]) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for brand in candidates:
        aliases = BRAND_ALIASES.get(brand, (brand,))
        compiled[brand] = []
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
            compiled[brand].append((alias, re.compile(pattern, flags=re.IGNORECASE)))
    return compiled


def _details_brand_values(details: Any) -> Iterator[str]:
    parsed = details
    if isinstance(details, str):
        try:
            parsed = json.loads(details)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if str(key).lower() == "brand":
                text = _clean_str(value)
                if text:
                    yield text


def _field_values(row: dict[str, Any]) -> Iterator[tuple[str, str]]:
    for key in ("store", "title"):
        value = _clean_str(row.get(key))
        if value:
            yield key, value

    details = row.get("details")
    for value in _details_brand_values(details):
        yield "details.Brand", value
    if details:
        if isinstance(details, str):
            details_text = details
        else:
            details_text = json.dumps(details, ensure_ascii=False, sort_keys=True)
        if details_text.strip():
            yield "details_json", details_text

    categories = row.get("categories")
    if categories:
        if isinstance(categories, str):
            categories_text = categories
        else:
            categories_text = json.dumps(categories, ensure_ascii=False)
        if categories_text.strip():
            yield "categories", categories_text


def match_candidate_brands(
    row: dict[str, Any],
    candidates: tuple[str, ...] | list[str] | None = None,
) -> dict[str, BrandMatch]:
    return _match_candidate_brands(row, _compile_aliases(_candidate_list(candidates)))


def _match_candidate_brands(
    row: dict[str, Any],
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]],
) -> dict[str, BrandMatch]:
    matches: dict[str, BrandMatch] = {}
    for field_name, value in _field_values(row):
        for brand, alias_patterns in compiled.items():
            for alias, pattern in alias_patterns:
                if pattern.search(value):
                    match = matches.setdefault(brand, BrandMatch())
                    match.fields.add(field_name)
                    match.aliases.add(alias)
    return matches


def choose_best_brand(matches: dict[str, BrandMatch], candidates: tuple[str, ...] | None = None) -> str | None:
    if not matches:
        return None
    order = {brand: i for i, brand in enumerate(_candidate_list(candidates))}

    def score(item: tuple[str, BrandMatch]) -> tuple[int, int, int]:
        brand, match = item
        field_score = max((FIELD_WEIGHTS.get(f, 0) for f in match.fields), default=0)
        alias_score = max((len(a) for a in match.aliases), default=0)
        return field_score, alias_score, -order.get(brand, 999)

    return max(matches.items(), key=score)[0]


def confidence_note(fields: set[str]) -> str:
    if "details.Brand" in fields or "store" in fields:
        return "high: matched structured brand/store field"
    if "details_json" in fields:
        return "medium: matched serialized details"
    return "medium: matched title/categories alias"


def scan_metadata(
    metadata_path: Path,
    *,
    candidates: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> tuple[int, dict[str, BrandInventory]]:
    candidate_tuple = _candidate_list(candidates)
    compiled = _compile_aliases(candidate_tuple)
    inventory = {brand: BrandInventory(candidate_brand=brand) for brand in candidate_tuple}
    scanned = 0
    for row in stream_jsonl(metadata_path):
        scanned += 1
        matches = _match_candidate_brands(row, compiled)
        for brand, match in matches.items():
            inventory[brand].add(row, match)
        if limit is not None and scanned >= limit:
            break
    return scanned, inventory


def inventory_csv_rows(inventory: dict[str, BrandInventory]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for brand, item in inventory.items():
        rows.append(
            {
                "candidate_brand": brand,
                "matched_items": item.matched_items,
                "matched_parent_asins": len(item.parent_asins),
                "matched_asins": len(item.asins),
                "example_parent_asins": "; ".join(item.example_parent_asins),
                "example_titles": " | ".join(item.example_titles),
                "matched_fields": "; ".join(sorted(item.matched_fields)),
                "confidence_notes": "; ".join(sorted(item.confidence_notes)),
            }
        )
    return rows


def write_inventory_csv(path: Path, inventory: dict[str, BrandInventory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = inventory_csv_rows(inventory)
    fieldnames = [
        "candidate_brand",
        "matched_items",
        "matched_parent_asins",
        "matched_asins",
        "example_parent_asins",
        "example_titles",
        "matched_fields",
        "confidence_notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_metadata_path(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if AMAZON_METADATA_PATH is not None and AMAZON_METADATA_PATH.exists():
        return AMAZON_METADATA_PATH
    for name in DEFAULT_METADATA_NAMES:
        candidate = EXTERNAL_DATA_DIR / name
        if candidate.exists():
            return candidate
    searched = ", ".join(f"data/{name}" for name in DEFAULT_METADATA_NAMES)
    raise FileNotFoundError(
        "Amazon metadata file not found. Set AMAZON_METADATA_PATH, pass --metadata, "
        f"or put one of these files under data/: {searched}."
    )


def _parse_candidates(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Amazon Reviews 2023 metadata for candidate brands.")
    parser.add_argument("--metadata", default=None, help="Path to raw_meta_Electronics/meta_Electronics JSONL(.GZ)")
    parser.add_argument("--output", default="data/brand_inventory.csv", help="CSV output path")
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate brands")
    parser.add_argument("--limit", type=int, default=None, help="Max metadata rows to scan")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        metadata_path = resolve_metadata_path(args.metadata)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    candidates = _parse_candidates(args.candidates)
    scanned, inventory = scan_metadata(metadata_path, candidates=candidates, limit=args.limit)
    output_path = Path(args.output)
    write_inventory_csv(output_path, inventory)
    print(f"Scanned metadata rows: {scanned}")
    print(f"Wrote brand inventory: {output_path}")
    for row in inventory_csv_rows(inventory):
        print(f"{row['candidate_brand']}: {row['matched_items']} matched metadata items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
