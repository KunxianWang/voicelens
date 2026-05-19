from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generate_brand_allowlist import generate_allowlist
from voicelens.ingest.brand_review_scan import resolve_reviews_path, scan_brand_reviews
from voicelens.ingest.brand_scan import (
    choose_best_brand,
    match_candidate_brands,
    resolve_metadata_path,
    scan_metadata,
    write_inventory_csv,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_brand_scanner_detects_store_title_and_details(tmp_path: Path):
    path = tmp_path / "meta.jsonl"
    _write_jsonl(
        path,
        [
            {"parent_asin": "P1", "asin": "A1", "store": "AnkerDirect", "title": "USB C hub"},
            {"parent_asin": "P2", "asin": "A2", "title": "JBL Flip 5 waterproof speaker"},
            {"parent_asin": "P3", "asin": "A3", "details": {"Brand": "UGREEN"}},
            {"parent_asin": "P4", "asin": "A4", "details": "{\"Brand\": \"Bose\"}"},
        ],
    )

    scanned, inventory = scan_metadata(path, candidates=("Anker", "JBL", "UGREEN", "Bose"))

    assert scanned == 4
    assert inventory["Anker"].matched_items == 1
    assert inventory["JBL"].matched_items == 1
    assert inventory["UGREEN"].matched_items == 1
    assert inventory["Bose"].matched_items == 1
    assert "store" in inventory["Anker"].matched_fields
    assert "title" in inventory["JBL"].matched_fields
    assert "details.Brand" in inventory["UGREEN"].matched_fields
    assert "details.Brand" in inventory["Bose"].matched_fields


def test_aliases_map_to_canonical_brands():
    matches = match_candidate_brands(
        {"store": "Soundcore by Anker", "title": "RAV power portable charger"},
        candidates=("Anker", "Soundcore", "RAVPower"),
    )

    assert set(matches) == {"Anker", "Soundcore", "RAVPower"}
    assert choose_best_brand(matches, ("Anker", "Soundcore", "RAVPower")) == "Soundcore"


def test_unknown_brands_are_ignored(tmp_path: Path):
    path = tmp_path / "meta.jsonl"
    _write_jsonl(path, [{"parent_asin": "P1", "asin": "A1", "store": "Sony", "title": "Camera"}])

    _, inventory = scan_metadata(path, candidates=("Anker", "Bose"))

    assert inventory["Anker"].matched_items == 0
    assert inventory["Bose"].matched_items == 0


def test_review_count_scanner_joins_reviews_to_metadata_brand_map(tmp_path: Path):
    meta_path = tmp_path / "meta.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    _write_jsonl(
        meta_path,
        [
            {"parent_asin": "P1", "asin": "A1", "store": "Anker Direct"},
            {"parent_asin": "P2", "asin": "A2", "details": {"Brand": "JBL"}},
            {"parent_asin": "P3", "asin": "A3", "store": "Sony"},
        ],
    )
    _write_jsonl(
        review_path,
        [
            {"parent_asin": "P1", "asin": "A1", "rating": 5, "timestamp": 1714521600000},
            {"parent_asin": "P1", "asin": "A1", "rating": 4, "timestamp": 1714608000000},
            {"parent_asin": "P2", "asin": "A2", "rating": 3, "timestamp": "2024-05-03T00:00:00Z"},
            {"parent_asin": "P3", "asin": "A3", "rating": 1, "timestamp": 1714521600000},
        ],
    )

    result = scan_brand_reviews(meta_path, review_path, candidates=("Anker", "JBL"))

    assert result.review_rows_scanned == 4
    assert result.stats["Anker"].matched_reviews == 2
    assert result.stats["Anker"].rating_counts[5] == 1
    assert result.stats["Anker"].rating_counts[4] == 1
    assert result.stats["Anker"].first_review_date == "2024-05-01"
    assert result.stats["JBL"].matched_reviews == 1
    assert result.stats["JBL"].rating_counts[3] == 1


def test_resolved_allowlist_generator_drops_low_count_brands(tmp_path: Path):
    inventory_path = tmp_path / "brand_inventory.csv"
    review_counts_path = tmp_path / "brand_review_counts.csv"
    _write_csv(
        inventory_path,
        [
            {"candidate_brand": "Anker", "matched_items": 30},
            {"candidate_brand": "Aukey", "matched_items": 5},
            {"candidate_brand": "Bose", "matched_items": 25},
        ],
    )
    _write_csv(
        review_counts_path,
        [
            {"candidate_brand": "Anker", "matched_reviews": 1200},
            {"candidate_brand": "Aukey", "matched_reviews": 2000},
            {"candidate_brand": "Bose", "matched_reviews": 100},
        ],
    )

    payload = generate_allowlist(
        inventory_path,
        review_counts_path,
        min_items=20,
        min_reviews=1000,
    )

    assert payload["brands"] == ["Anker"]
    assert "too few matched metadata items" in payload["dropped"]["Aukey"]
    assert "too few matched reviews" in payload["dropped"]["Bose"]


def test_inventory_csv_writer_has_required_columns(tmp_path: Path):
    meta_path = tmp_path / "meta.jsonl"
    out_path = tmp_path / "brand_inventory.csv"
    _write_jsonl(meta_path, [{"parent_asin": "P1", "asin": "A1", "store": "Anker"}])
    _, inventory = scan_metadata(meta_path, candidates=("Anker",))

    write_inventory_csv(out_path, inventory)
    with open(out_path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["candidate_brand"] == "Anker"
    assert row["matched_items"] == "1"
    assert row["matched_parent_asins"] == "1"
    assert row["matched_asins"] == "1"
    assert row["matched_fields"] == "store"


def test_missing_file_errors_are_clear(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("voicelens.ingest.brand_scan.EXTERNAL_DATA_DIR", tmp_path)
    monkeypatch.setattr("voicelens.ingest.brand_review_scan.EXTERNAL_DATA_DIR", tmp_path)
    monkeypatch.setattr("voicelens.ingest.brand_review_scan.AMAZON_REVIEWS_PATH", tmp_path / "missing.jsonl")

    try:
        resolve_metadata_path(None)
    except FileNotFoundError as exc:
        assert "Set AMAZON_METADATA_PATH" in str(exc)
        assert "put one of these files under data/" in str(exc)
    else:
        raise AssertionError("expected metadata FileNotFoundError")

    try:
        resolve_reviews_path(None)
    except FileNotFoundError as exc:
        assert "Set AMAZON_REVIEWS_PATH" in str(exc)
        assert "put one of these files under data/" in str(exc)
    else:
        raise AssertionError("expected reviews FileNotFoundError")
