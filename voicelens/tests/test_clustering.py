"""Pure clustering + labelling tests (no DB, no heavy models)."""
from __future__ import annotations

from voicelens.analytics.clustering import (
    ClusterRecord,
    ClusterResult,
    cluster_records,
    severity_weight,
)
from voicelens.analytics.labels import build_cluster_label


def _rec(aspect: str, quote: str, *, severity: str | None = "medium", rid: int) -> ClusterRecord:
    return ClusterRecord(
        review_id=rid,
        brand="Anker",
        asin="B0X",
        rating=1,
        posted_at="2024-01-01T00:00:00",
        aspect_code=aspect,
        severity=severity,
        evidence_quote=quote,
        text_raw=quote,
    )


# ---- severity weighting -------------------------------------------------


def test_severity_weight_mapping():
    assert severity_weight("high") == 3.0
    assert severity_weight("medium") == 2.0
    assert severity_weight("low") == 1.0


def test_severity_weight_defaults_for_unknown_or_missing():
    assert severity_weight(None) == 1.0
    assert severity_weight("") == 1.0
    assert severity_weight("catastrophic") == 1.0


# ---- cluster_records ----------------------------------------------------


def test_cluster_records_skips_aspect_below_min_size():
    """An aspect with fewer mentions than min_cluster_size is dropped."""
    records = [
        # battery: only 2 mentions -> below the floor of 5.
        _rec("battery", "battery drains way too fast", rid=1),
        _rec("battery", "battery dies in an hour", rid=2),
        # charging: 8 mentions -> enough to cluster.
        *[
            _rec("charging", f"charging port stopped working unit {i}", rid=100 + i)
            for i in range(8)
        ],
    ]
    results = cluster_records(records, min_cluster_size=5)
    aspects = {r.aspect_code for r in results}
    assert "battery" not in aspects
    assert "charging" in aspects


def test_cluster_records_every_cluster_meets_min_size():
    records = [
        *[_rec("charging", f"charging port is loose and wobbly {i}", rid=200 + i) for i in range(6)],
        *[_rec("charging", f"charging cable frayed and broke {i}", rid=300 + i) for i in range(6)],
    ]
    results = cluster_records(records, min_cluster_size=5)
    assert results
    for result in results:
        assert result.aspect_code == "charging"
        assert result.size >= 5
        assert result.algorithm == "tfidf_kmeans"
        assert result.label  # non-empty label


def test_cluster_records_is_deterministic():
    records = [
        _rec("bluetooth", f"bluetooth keeps disconnecting randomly {i}", rid=400 + i)
        for i in range(10)
    ]
    first = cluster_records(records, min_cluster_size=5, random_state=7)
    second = cluster_records(records, min_cluster_size=5, random_state=7)
    assert [c.label for c in first] == [c.label for c in second]
    assert [c.member_review_ids for c in first] == [c.member_review_ids for c in second]


def test_cluster_records_empty_input_returns_empty():
    assert cluster_records([], min_cluster_size=5) == []


# ---- ClusterResult properties ------------------------------------------


def test_cluster_result_severity_weighted_size():
    members = [
        _rec("price", "too expensive", severity="high", rid=1),     # 3.0
        _rec("price", "not worth it", severity="low", rid=2),       # 1.0
        _rec("price", "overpriced", severity=None, rid=3),          # 1.0
    ]
    result = ClusterResult(
        aspect_code="price", label="x", algorithm="tfidf_kmeans",
        keywords=[], members=members,
    )
    assert result.size == 3
    assert result.severity_weighted_size == 5.0
    assert result.member_review_ids == [1, 2, 3]


# ---- labels -------------------------------------------------------------


def test_build_cluster_label_from_keywords():
    label = build_cluster_label(
        ["stopped working", "working", "stopped", "week", "the"],
        ["it stopped working after a week"],
        aspect_code="reliability",
    )
    # de-duped (working/stopped repeat), stop-words dropped, order kept.
    assert label == "stopped working week"


def test_build_cluster_label_respects_max_words():
    label = build_cluster_label(
        ["alpha beta", "gamma", "delta", "epsilon", "zeta", "eta"],
        [],
        aspect_code="charging",
        max_words=3,
    )
    assert len(label.split()) == 3


def test_build_cluster_label_falls_back_to_quote():
    label = build_cluster_label(
        [], ["The charging port completely failed"], aspect_code="charging"
    )
    assert "charging" in label and "port" in label


def test_build_cluster_label_falls_back_to_aspect_when_empty():
    label = build_cluster_label([], [], aspect_code="sound_quality")
    assert label == "sound quality issues"
