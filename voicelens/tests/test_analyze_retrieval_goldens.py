"""Tests for the retrieval-golden diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analyze_retrieval_goldens import (
    diagnose_golden,
    load_no_gold_hit_counts,
)

# A fully indexed corpus for the happy-path cases.
INDEXED = set(range(1, 21))


def _diagnose(golden, indexed=INDEXED, counts=None, n_modes=0):
    return diagnose_golden(
        golden, indexed,
        no_gold_hit_counts=counts or {},
        n_modes_seen=n_modes,
    )


def test_diagnose_flags_zero_gold_query():
    row = _diagnose({"query_id": "q1", "query": "x", "gold_review_ids": []})
    assert row["num_gold"] == 0
    assert row["needs_manual_review"] is True
    assert "no_gold" in row["notes"]


def test_diagnose_flags_high_gold_query():
    row = _diagnose(
        {"query_id": "q2", "query": "x", "gold_review_ids": list(range(1, 13))}
    )
    assert row["num_gold"] == 12
    assert row["needs_manual_review"] is True
    assert "too_many_gold" in row["notes"]


def test_diagnose_flags_low_max_recall_at_5():
    # 8 gold, all indexed -> max Recall@5 = 5/8 = 0.625 < 0.75.
    row = _diagnose(
        {"query_id": "q3", "query": "x", "gold_review_ids": list(range(1, 9))}
    )
    assert row["max_possible_recall_at_5"] == 0.625
    assert "low_max_recall_at_5" in row["notes"]
    assert row["needs_manual_review"] is True


def test_diagnose_flags_gold_missing_from_index():
    row = _diagnose(
        {"query_id": "q4", "query": "x", "gold_review_ids": [1, 2, 999]}
    )
    assert json.loads(row["gold_review_ids_missing"]) == [999]
    assert "gold_missing_from_index" in row["notes"]
    assert row["needs_manual_review"] is True


def test_diagnose_clean_golden_is_not_flagged():
    row = _diagnose(
        {"query_id": "q5", "query": "x", "gold_review_ids": [1, 2, 3]}
    )
    assert row["needs_manual_review"] is False
    assert row["notes"] == "ok"
    assert row["max_possible_recall_at_5"] == 1.0


def test_diagnose_flags_no_gold_hit_in_all_modes():
    row = _diagnose(
        {"query_id": "q6", "query": "x", "gold_review_ids": [1, 2, 3]},
        counts={"q6": 3},
        n_modes=3,
    )
    assert "no_gold_hit_all_modes" in row["notes"]
    assert row["needs_manual_review"] is True


def test_diagnose_no_gold_hit_in_some_modes_not_flagged():
    # Missed in 1 of 3 modes -> not "all modes", clean golden stays ok.
    row = _diagnose(
        {"query_id": "q7", "query": "x", "gold_review_ids": [1, 2, 3]},
        counts={"q7": 1},
        n_modes=3,
    )
    assert "no_gold_hit_all_modes" not in row["notes"]
    assert row["needs_manual_review"] is False


def test_load_no_gold_hit_counts_aggregates_across_mode_files(tmp_path: Path):
    for mode in ("dense", "lexical"):
        path = tmp_path / f"retrieval_errors_{mode}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "error_type"])
            writer.writeheader()
            writer.writerow({"query_id": "qa", "error_type": "no_gold_hit"})
            writer.writerow({"query_id": "qb", "error_type": "good"})
    counts, n_modes = load_no_gold_hit_counts(str(tmp_path / "retrieval_errors_*.csv"))
    assert n_modes == 2
    assert counts["qa"] == 2
    assert "qb" not in counts
