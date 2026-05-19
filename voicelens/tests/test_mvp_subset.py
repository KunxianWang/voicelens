from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.build_mvp_subset import build_mvp_subset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_allowlist(path: Path, brands: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"brands": brands}, f)


def _read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fixture_paths(tmp_path: Path):
    metadata_path = tmp_path / "meta.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    allowlist_path = tmp_path / "resolved_brand_allowlist.json"
    _write_jsonl(
        metadata_path,
        [
            {"parent_asin": "PANK", "store": "Anker Direct"},
            {"parent_asin": "PBOSE", "details": {"Brand": "Bose"}},
            {"parent_asin": "PSONY", "store": "Sony"},
        ],
    )
    reviews: list[dict] = []
    for i in range(10):
        reviews.append(
            {
                "parent_asin": "PANK",
                "asin": f"AANK{i}",
                "rating": (i % 5) + 1,
                "timestamp": f"202{i % 3}-01-01T00:00:00Z",
                "user_id": f"UA{i}",
                "text": f"Anker review {i} has enough words for deterministic sampling.",
            }
        )
    for i in range(3):
        reviews.append(
            {
                "parent_asin": "PBOSE",
                "asin": f"ABOSE{i}",
                "rating": 5,
                "timestamp": "2021-06-01T00:00:00Z",
                "user_id": f"UB{i}",
                "text": f"Bose review {i} should be fully retained.",
            }
        )
    for i in range(2):
        reviews.append(
            {
                "parent_asin": "PSONY",
                "asin": f"ASONY{i}",
                "rating": 1,
                "timestamp": "2021-06-01T00:00:00Z",
                "user_id": f"US{i}",
                "text": "Unknown brand should be ignored.",
            }
        )
    _write_jsonl(reviews_path, reviews)
    _write_allowlist(allowlist_path, ["Anker", "Bose"])
    return metadata_path, reviews_path, allowlist_path


def test_mvp_subset_sampling_is_deterministic_for_same_seed(tmp_path: Path):
    metadata_path, reviews_path, allowlist_path = _fixture_paths(tmp_path)
    output_a = tmp_path / "subset_a.jsonl.gz"
    output_b = tmp_path / "subset_b.jsonl.gz"

    build_mvp_subset(
        metadata_path=metadata_path,
        reviews_path=reviews_path,
        allowlist_path=allowlist_path,
        output_path=output_a,
        profile_path=tmp_path / "profile_a.json",
        target_counts={"Anker": 4, "Bose": 2},
        seed=42,
    )
    build_mvp_subset(
        metadata_path=metadata_path,
        reviews_path=reviews_path,
        allowlist_path=allowlist_path,
        output_path=output_b,
        profile_path=tmp_path / "profile_b.json",
        target_counts={"Anker": 4, "Bose": 2},
        seed=42,
    )

    assert _read_gzip_jsonl(output_a) == _read_gzip_jsonl(output_b)
    assert output_a.read_bytes() == output_b.read_bytes()


def test_mvp_subset_retains_brand_below_target_and_ignores_unknown(tmp_path: Path):
    metadata_path, reviews_path, allowlist_path = _fixture_paths(tmp_path)
    output_path = tmp_path / "subset.jsonl.gz"

    profile = build_mvp_subset(
        metadata_path=metadata_path,
        reviews_path=reviews_path,
        allowlist_path=allowlist_path,
        output_path=output_path,
        profile_path=tmp_path / "profile.json",
        target_counts={"Anker": 4, "Bose": 10},
        seed=42,
    )

    rows = _read_gzip_jsonl(output_path)
    assert profile["actual_counts_by_brand"] == {"Anker": 4, "Bose": 3}
    assert len(rows) == 7
    assert {row["parent_asin"] for row in rows} == {"PANK", "PBOSE"}


def test_mvp_subset_profile_contains_required_fields(tmp_path: Path):
    metadata_path, reviews_path, allowlist_path = _fixture_paths(tmp_path)
    profile_path = tmp_path / "profile.json"

    build_mvp_subset(
        metadata_path=metadata_path,
        reviews_path=reviews_path,
        allowlist_path=allowlist_path,
        output_path=tmp_path / "subset.jsonl.gz",
        profile_path=profile_path,
        target_counts={"Anker": 4, "Bose": 10},
        seed=123,
    )
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)

    for key in (
        "target_counts",
        "actual_counts_by_brand",
        "rating_distribution_by_brand",
        "year_distribution_by_brand",
        "total_output_reviews",
        "input_paths",
        "seed",
        "generated_at",
    ):
        assert key in profile
    assert profile["seed"] == 123
    assert profile["rating_distribution_by_brand"]["Bose"] == {"5": 3}
    assert profile["year_distribution_by_brand"]["Bose"] == {"2021": 3}
