from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from voicelens.db.models import ABSAReviewStatus, AspectMention, IngestRun, Review
from voicelens.pipeline.flows.absa_flow import (
    _read_review_ids_file,
    _resolve_review_ids,
    absa_flow,
)
from voicelens.tests._absa_fixture import REVIEW_TEXTS  # noqa: F401


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_read_review_ids_file_prefers_review_id(tmp_path: Path):
    path = tmp_path / "ids.jsonl"
    _write_jsonl(
        path,
        [
            {"review_id": 7, "source_id": "X1"},
            {"review_id": 9},
            {"source_id": "Y2"},
            {"review_id": 7},  # duplicate
        ],
    )
    review_ids, source_ids = _read_review_ids_file(path)
    assert review_ids == [7, 9]
    assert source_ids == ["Y2"]


def test_read_review_ids_file_rejects_empty_file(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no review_id or source_id entries"):
        _read_review_ids_file(path)


def test_read_review_ids_file_rejects_bad_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        _read_review_ids_file(path)


def test_resolve_review_ids_uses_source_id_fallback(seeded_engine):
    engine, ids = seeded_engine
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        found, missing = _resolve_review_ids(
            s,
            review_ids=[ids["Anker-1"]],
            source_ids=["Soundcore-1", "DOES_NOT_EXIST"],
        )
    assert ids["Anker-1"] in found
    assert ids["Soundcore-1"] in found
    assert "source_id='DOES_NOT_EXIST'" in missing


def test_review_ids_file_processes_only_listed_reviews(seeded_engine, tmp_path: Path):
    engine, ids = seeded_engine
    target_rids = [ids["Anker-1"], ids["Bose-1"], ids["JBL-1"]]
    path = tmp_path / "holdout.jsonl"
    _write_jsonl(path, [{"review_id": rid} for rid in target_rids])

    summary = absa_flow(provider="mock", review_ids_file=str(path))

    assert summary["input_reviews"] == len(target_rids)
    assert summary["processed_reviews"] == len(target_rids)
    assert summary["review_ids_file_resolved"] == len(target_rids)
    assert summary["review_ids_file_unresolved"] == 0

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        touched = set(
            s.execute(
                select(ABSAReviewStatus.review_id).where(
                    ABSAReviewStatus.provider == "mock",
                    ABSAReviewStatus.aspect_version == "v1",
                )
            ).scalars()
        )
    assert touched == set(target_rids), "only the listed reviews should be processed"


def test_review_ids_file_source_id_fallback_resolves(seeded_engine, tmp_path: Path):
    engine, ids = seeded_engine
    path = tmp_path / "by_source_id.jsonl"
    _write_jsonl(
        path,
        [
            {"source_id": "Bose-1"},
            {"source_id": "UGREEN-1"},
        ],
    )

    summary = absa_flow(provider="mock", review_ids_file=str(path))

    assert summary["processed_reviews"] == 2
    assert summary["review_ids_file_resolved"] == 2

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        touched = set(
            s.execute(
                select(ABSAReviewStatus.review_id).where(
                    ABSAReviewStatus.provider == "mock",
                    ABSAReviewStatus.aspect_version == "v1",
                )
            ).scalars()
        )
    assert touched == {ids["Bose-1"], ids["UGREEN-1"]}


def test_review_ids_file_ignores_source_filter(seeded_engine, tmp_path: Path):
    """When --review-ids-file is set, --source should be ignored."""
    engine, ids = seeded_engine
    path = tmp_path / "ids.jsonl"
    target = ids["Anker-1"]
    _write_jsonl(path, [{"review_id": target}])

    summary = absa_flow(
        provider="mock",
        review_ids_file=str(path),
        source="this_source_does_not_match_anything",
    )
    assert summary["processed_reviews"] == 1
    assert summary["review_ids_file_resolved"] == 1


def test_review_ids_file_preserves_idempotency(seeded_engine, tmp_path: Path):
    engine, ids = seeded_engine
    path = tmp_path / "ids.jsonl"
    target = [ids["Anker-1"], ids["Soundcore-1"]]
    _write_jsonl(path, [{"review_id": rid} for rid in target])

    first = absa_flow(provider="mock", review_ids_file=str(path))
    second = absa_flow(provider="mock", review_ids_file=str(path))

    assert first["processed_reviews"] == 2
    assert second["input_reviews"] == 0, "already-processed reviews must be skipped"
    assert second["processed_reviews"] == 0

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_status = s.scalar(
            select(func.count()).select_from(ABSAReviewStatus).where(
                ABSAReviewStatus.review_id.in_(target),
                ABSAReviewStatus.provider == "mock",
            )
        )
    assert n_status == 2, "no duplicate status rows"


def test_review_ids_file_force_reprocesses(seeded_engine, tmp_path: Path):
    engine, ids = seeded_engine
    path = tmp_path / "ids.jsonl"
    target = [ids["Anker-1"], ids["Bose-1"]]
    _write_jsonl(path, [{"review_id": rid} for rid in target])

    first = absa_flow(provider="mock", review_ids_file=str(path))
    forced = absa_flow(provider="mock", review_ids_file=str(path), force=True)

    assert forced["input_reviews"] == 2
    assert forced["processed_reviews"] == 2
    assert forced["inserted_mentions"] == first["inserted_mentions"]

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_status = s.scalar(
            select(func.count()).select_from(ABSAReviewStatus).where(
                ABSAReviewStatus.review_id.in_(target),
                ABSAReviewStatus.provider == "mock",
            )
        )
        n_mentions = s.scalar(
            select(func.count()).select_from(AspectMention).where(
                AspectMention.review_id.in_(target),
                AspectMention.model_name == "mock",
            )
        )
    assert n_status == 2, "force must replace, not duplicate"
    assert n_mentions == first["inserted_mentions"], "mention counts must stay stable"


def test_review_ids_file_with_unresolved_entries_logs_count(seeded_engine, tmp_path: Path):
    engine, ids = seeded_engine
    path = tmp_path / "ids.jsonl"
    _write_jsonl(
        path,
        [
            {"review_id": ids["Anker-1"]},
            {"review_id": 999_999_999},
            {"source_id": "DOES_NOT_EXIST"},
        ],
    )
    summary = absa_flow(provider="mock", review_ids_file=str(path))

    assert summary["review_ids_file_requested"] == 3
    assert summary["review_ids_file_resolved"] == 1
    assert summary["review_ids_file_unresolved"] == 2
    assert summary["processed_reviews"] == 1


def test_source_limit_mode_unchanged_when_review_ids_file_not_set(seeded_engine):
    """Regression guard: legacy --source/--limit path must keep its behaviour."""
    engine, _ids = seeded_engine
    summary = absa_flow(
        provider="mock",
        source="amazon_reviews_2023_mvp_subset",
        limit=3,
    )
    assert summary["input_reviews"] == 3
    assert summary["processed_reviews"] == 3
    # review_ids_file fields should be absent in summary
    assert "review_ids_file" not in summary
    assert "review_ids_file_resolved" not in summary


def test_review_ids_file_handles_other_run_ingest_source(seeded_engine, tmp_path: Path):
    """A review on a non-MVP ingest_run must still be processable by review-id selection."""
    engine, ids = seeded_engine
    path = tmp_path / "ids.jsonl"
    _write_jsonl(path, [{"review_id": ids["OTHER-1"]}])

    summary = absa_flow(provider="mock", review_ids_file=str(path))
    assert summary["processed_reviews"] == 1

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        run_source = s.scalar(
            select(IngestRun.source)
            .join(Review, Review.ingest_run_id == IngestRun.id)
            .where(Review.id == ids["OTHER-1"])
        )
    assert run_source == "other_source"
