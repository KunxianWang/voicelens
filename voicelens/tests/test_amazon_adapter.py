from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from voicelens.config import AMAZON_FIXTURE_PATH, BRAND_ALLOWLIST
from voicelens.ingest.amazon_reviews_2023 import (
    canonicalize_brand,
    iter_amazon_reviews,
    load_asin_brand_map,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_canonicalize_brand_handles_variants():
    al = ("Anker", "Soundcore", "Bose")
    assert canonicalize_brand("Anker", al) == "Anker"
    assert canonicalize_brand("ANKER", al) == "Anker"
    assert canonicalize_brand("Anker Innovations", al) == "Anker"
    assert canonicalize_brand("anker direct", al) == "Anker"
    assert canonicalize_brand("soundcore", al) == "Soundcore"
    assert canonicalize_brand("Sony", al) is None
    assert canonicalize_brand("", al) is None
    assert canonicalize_brand(None, al) is None


def test_jsonl_reader(tmp_path: Path):
    rows = [
        {"rating": 5.0, "title": "great", "text": "love it for the price, works perfectly.",
         "asin": "B0ANK10001", "user_id": "U1", "timestamp": 1714521600000, "brand": "Anker"},
        {"rating": 4.0, "title": "good", "text": "works as advertised, no complaints at all.",
         "asin": "B0SND20001", "user_id": "U2", "timestamp": 1715126400000, "brand": "Soundcore"},
    ]
    path = tmp_path / "reviews.jsonl"
    _write_jsonl(path, rows)

    out = list(iter_amazon_reviews(path, brand_allowlist=("Anker", "Soundcore")))
    assert len(out) == 2
    assert {r["brand"] for r in out} == {"Anker", "Soundcore"}
    assert all(r["source"] == "amazon_reviews_2023" for r in out)


def test_jsonl_gz_reader(tmp_path: Path):
    rows = [
        {"rating": 5.0, "title": "t", "text": "this is a sufficiently long review body for the test.",
         "asin": "B0ANK10001", "user_id": "U1", "timestamp": 1714521600000, "brand": "Anker"},
    ]
    path = tmp_path / "reviews.jsonl.gz"
    _write_jsonl_gz(path, rows)

    out = list(iter_amazon_reviews(path, brand_allowlist=("Anker",)))
    assert len(out) == 1
    assert out[0]["asin"] == "B0ANK10001"


def test_malformed_lines_are_skipped(tmp_path: Path):
    path = tmp_path / "reviews.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json\n")
        f.write(json.dumps({"rating": 5.0, "title": "ok", "text": "this review text is fine for the test.",
                            "asin": "B0ANK10001", "user_id": "U1", "timestamp": 1714521600000,
                            "brand": "Anker"}) + "\n")
        f.write("\n")
    out = list(iter_amazon_reviews(path, brand_allowlist=("Anker",)))
    assert len(out) == 1


def test_brand_allowlist_filters_unknown(tmp_path: Path):
    rows = [
        {"rating": 5.0, "title": "a", "text": "anker review with enough text body content.",
         "asin": "B0ANK10001", "user_id": "U1", "timestamp": 1714521600000, "brand": "Anker"},
        {"rating": 5.0, "title": "s", "text": "sony review should be filtered out by the allowlist.",
         "asin": "B0SNY99001", "user_id": "U2", "timestamp": 1714521600000, "brand": "Sony"},
        {"rating": 5.0, "title": "x", "text": "no brand field at all, should also be dropped.",
         "asin": "B0XXX99001", "user_id": "U3", "timestamp": 1714521600000},
    ]
    path = tmp_path / "reviews.jsonl"
    _write_jsonl(path, rows)

    out = list(iter_amazon_reviews(path, brand_allowlist=("Anker", "Soundcore")))
    assert len(out) == 1
    assert out[0]["brand"] == "Anker"


def test_limit_caps_yield(tmp_path: Path):
    rows = [
        {"rating": 5.0, "title": "t", "text": "this is a long enough review body for testing limits.",
         "asin": f"B0ANK{i:05d}", "user_id": f"U{i}", "timestamp": 1714521600000, "brand": "Anker"}
        for i in range(10)
    ]
    path = tmp_path / "reviews.jsonl"
    _write_jsonl(path, rows)

    out = list(iter_amazon_reviews(path, brand_allowlist=("Anker",), limit=3))
    assert len(out) == 3


def test_metadata_asin_lookup(tmp_path: Path):
    metadata_rows = [
        {"parent_asin": "B0XYZ00001", "asin": "B0XYZ00001", "store": "Anker Direct",
         "details": {"Brand": "Anker"}},
        {"parent_asin": "B0XYZ00002", "asin": "B0XYZ00002", "store": "Sony Inc",
         "details": {"Manufacturer": "Sony"}},
    ]
    review_rows = [
        {"rating": 5.0, "title": "great", "text": "no inline brand on the review row.",
         "asin": "B0XYZ00001", "user_id": "U1", "timestamp": 1714521600000},
        {"rating": 4.0, "title": "ok", "text": "this row should be dropped via metadata lookup.",
         "asin": "B0XYZ00002", "user_id": "U2", "timestamp": 1714521600000},
    ]
    meta_path = tmp_path / "meta.jsonl"
    rev_path = tmp_path / "reviews.jsonl"
    _write_jsonl(meta_path, metadata_rows)
    _write_jsonl(rev_path, review_rows)

    al = ("Anker", "Soundcore")
    mapping = load_asin_brand_map(meta_path, allowlist=al)
    assert mapping == {"B0XYZ00001": "Anker"}

    out = list(iter_amazon_reviews(rev_path, asin_to_brand=mapping, brand_allowlist=al))
    assert len(out) == 1
    assert out[0]["asin"] == "B0XYZ00001"
    assert out[0]["brand"] == "Anker"


def test_metadata_asin_lookup_uses_candidate_brand_scanner(tmp_path: Path):
    metadata_rows = [
        {"parent_asin": "B0TITLE001", "title": "JBL Flip portable speaker", "details": {}},
        {"parent_asin": "B0DETAIL01", "title": "USB cable", "details": {"Compatible": "UGREEN hub"}},
    ]
    meta_path = tmp_path / "meta.jsonl"
    _write_jsonl(meta_path, metadata_rows)

    mapping = load_asin_brand_map(meta_path, allowlist=("JBL", "UGREEN"))

    assert mapping == {"B0TITLE001": "JBL", "B0DETAIL01": "UGREEN"}


def test_review_brand_lookup_prefers_parent_asin_when_child_asin_present(tmp_path: Path):
    metadata_rows = [
        {"parent_asin": "PARENT1", "title": "Anker USB C charger"},
    ]
    review_rows = [
        {
            "parent_asin": "PARENT1",
            "asin": "CHILD1",
            "rating": 5.0,
            "title": "great",
            "text": "this review has a child asin and should still resolve brand.",
            "user_id": "U1",
            "timestamp": 1714521600000,
        }
    ]
    meta_path = tmp_path / "meta.jsonl"
    rev_path = tmp_path / "reviews.jsonl"
    _write_jsonl(meta_path, metadata_rows)
    _write_jsonl(rev_path, review_rows)

    mapping = load_asin_brand_map(meta_path, allowlist=("Anker",))
    out = list(iter_amazon_reviews(rev_path, asin_to_brand=mapping, brand_allowlist=("Anker",)))

    assert len(out) == 1
    assert out[0]["brand"] == "Anker"
    assert out[0]["asin"] == "CHILD1"


def test_defensive_field_mapping(tmp_path: Path):
    rows = [
        {"rating": "5", "title": None, "text": "rating-as-string, title-null, still loadable.",
         "asin": "B0ANK10001", "user_id": "U1", "timestamp": 1714521600000, "brand": "Anker"},
        {"rating": 4.0, "title": "", "text": "boolean-as-string verified field works.",
         "asin": "B0ANK10002", "user_id": "U2", "timestamp": 1714521600000,
         "verified_purchase": "true", "brand": "Anker"},
        {"rating": 5.0, "title": "iso ts", "text": "timestamp as ISO string should still parse.",
         "asin": "B0ANK10003", "user_id": "U3", "timestamp": "2025-03-15T10:00:00Z",
         "brand": "Anker"},
    ]
    path = tmp_path / "reviews.jsonl"
    _write_jsonl(path, rows)

    out = list(iter_amazon_reviews(path, brand_allowlist=("Anker",)))
    assert len(out) == 3
    assert out[0]["rating"] == 5
    assert out[1]["verified"] is True
    assert out[2]["posted_at"].startswith("2025-03-15T10:00:00")


def test_missing_asin_passes_through_to_dq(tmp_path: Path):
    """Adapter should not pre-filter missing ASIN — DQ owns that check."""
    rows = [
        {"rating": 5.0, "title": "x", "text": "this row has neither asin nor parent_asin set.",
         "user_id": "U1", "timestamp": 1714521600000, "brand": "Anker"},
    ]
    path = tmp_path / "reviews.jsonl"
    _write_jsonl(path, rows)
    out = list(iter_amazon_reviews(path, brand_allowlist=("Anker",)))
    assert len(out) == 1
    assert out[0]["asin"] is None


def test_fixture_file_exists_and_parses():
    """The committed fixture must be valid JSONL and yield matched rows after
    brand-allowlist filter against the default BRAND_ALLOWLIST."""
    assert AMAZON_FIXTURE_PATH.exists(), AMAZON_FIXTURE_PATH
    out = list(iter_amazon_reviews(AMAZON_FIXTURE_PATH, brand_allowlist=BRAND_ALLOWLIST))
    assert len(out) >= 15  # Sony + Apple dropped; the rest yielded


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        list(iter_amazon_reviews(Path("/nonexistent/path/reviews.jsonl")))
