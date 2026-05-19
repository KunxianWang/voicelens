from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    IngestRun,
    Review,
)
from voicelens.pipeline.flows.absa_flow import absa_flow
from voicelens.tests._absa_fixture import REVIEW_TEXTS
from voicelens.tests._absa_fixture import seed_reviews as _seed_reviews


def test_absa_flow_inserts_valid_mentions(seeded_engine):
    engine, _ids = seeded_engine
    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider="mock",
    )

    assert summary["input_reviews"] == len(REVIEW_TEXTS)
    assert summary["processed_reviews"] == len(REVIEW_TEXTS)
    assert summary["inserted_mentions"] > 0
    assert summary["reviews_with_mentions"] >= 5
    assert summary["reviews_no_mentions"] >= 1, "fixture has one keyword-free review"
    assert summary["evidence_verbatim_rate"] == 1.0
    assert summary["provider"] == "mock"
    assert summary["model_name"] == "mock"
    assert summary["aspect_version"] == "v1"

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n = s.scalar(select(func.count()).select_from(AspectMention))
        rows = list(s.execute(select(AspectMention)).scalars())
    assert n == summary["inserted_mentions"]
    assert all(r.model_name == "mock" for r in rows)
    assert all(r.aspect_version == "v1" for r in rows)
    for r in rows:
        if r.sentiment == "negative":
            assert r.severity in {"low", "medium", "high"}
        else:
            assert r.severity is None


def test_absa_flow_is_idempotent_without_force(seeded_engine):
    engine, _ids = seeded_engine
    first = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")
    second = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    assert first["inserted_mentions"] > 0
    assert second["input_reviews"] == 0
    assert second["processed_reviews"] == 0
    assert second["inserted_mentions"] == 0

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n = s.scalar(select(func.count()).select_from(AspectMention))
    assert n == first["inserted_mentions"]


def test_absa_flow_force_reprocesses(seeded_engine):
    engine, _ids = seeded_engine
    first = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")
    forced = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock", force=True)

    assert forced["input_reviews"] == len(REVIEW_TEXTS)
    assert forced["inserted_mentions"] == first["inserted_mentions"]

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n = s.scalar(select(func.count()).select_from(AspectMention))
    assert n == first["inserted_mentions"], "force=True should replace, not duplicate"


def test_absa_flow_respects_source_filter(seeded_engine):
    engine, _ids = seeded_engine
    absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        other_review_ids = list(
            s.execute(
                select(Review.id)
                .join(IngestRun, IngestRun.id == Review.ingest_run_id)
                .where(IngestRun.source == "other_source")
            ).scalars()
        )
        mention_review_ids = set(
            s.execute(
                select(AspectMention.review_id).where(
                    AspectMention.review_id.in_(other_review_ids)
                )
            ).scalars()
        )
    assert mention_review_ids == set(), "source filter must exclude other-run reviews"


def test_absa_flow_limit_caps_processing(seeded_engine):
    engine, _ids = seeded_engine
    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider="mock",
        limit=2,
    )

    assert summary["input_reviews"] == 2
    assert summary["processed_reviews"] == 2


def test_absa_flow_requires_seeded_ontology(fresh_engine):
    SessionLocal = sessionmaker(bind=fresh_engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        _seed_reviews(s)
        s.commit()

    with pytest.raises(RuntimeError, match="aspect_ontology has no rows"):
        absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")


def test_absa_flow_runs_on_real_postgres_like_store(seeded_engine):
    engine, _ids = seeded_engine
    summary = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")
    assert "aspect_counts" in summary
    assert isinstance(summary["aspect_counts"], dict)
    assert "battery" in summary["aspect_counts"] or "charging" in summary["aspect_counts"]

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_ontology = s.scalar(select(func.count()).select_from(AspectOntology))
    assert n_ontology == 7


# ---------------------------------------------------------------------------
# Milestone 2A.1: review-level idempotency via absa_review_status
# ---------------------------------------------------------------------------


def test_no_mentions_review_is_recorded_as_no_mentions(seeded_engine):
    """A review whose provider output is ``{"aspects": []}`` must still
    create an ``absa_review_status`` row with status='no_mentions'."""
    engine, ids = seeded_engine
    absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        no_aspects_review_id = ids["NoAspects-1"]
        status_row = s.scalar(
            select(ABSAReviewStatus).where(
                ABSAReviewStatus.review_id == no_aspects_review_id,
                ABSAReviewStatus.provider == "mock",
                ABSAReviewStatus.aspect_version == "v1",
            )
        )
        n_mentions_for_review = s.scalar(
            select(func.count())
            .select_from(AspectMention)
            .where(AspectMention.review_id == no_aspects_review_id)
        )

    assert status_row is not None, "no_mentions review must still have a status row"
    assert status_row.status == "no_mentions"
    assert status_row.n_mentions == 0
    assert status_row.n_errors == 0
    assert n_mentions_for_review == 0


def test_rerun_does_not_reprocess_no_mentions_reviews(seeded_engine):
    """The bug from M2A: reviews with zero aspects were retried every run.
    With absa_review_status, they must be skipped."""
    engine, _ids = seeded_engine
    first = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")
    assert first["reviews_no_mentions"] >= 1

    second = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")
    assert second["input_reviews"] == 0
    assert second["processed_reviews"] == 0
    assert second["reviews_no_mentions"] == 0


def test_rerun_does_not_reprocess_success_reviews(seeded_engine):
    """The original idempotency guarantee: successful reviews are not
    reprocessed."""
    engine, _ids = seeded_engine
    absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")
    second = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    assert second["input_reviews"] == 0
    assert second["reviews_with_mentions"] == 0

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_status = s.scalar(select(func.count()).select_from(ABSAReviewStatus))
    assert n_status == len(REVIEW_TEXTS), "one status row per processed review"


def test_force_reprocesses_both_success_and_no_mentions(seeded_engine):
    """``--force`` must clear status rows for success AND no_mentions
    reviews and reprocess them; final counts equal first run."""
    engine, _ids = seeded_engine
    first = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")
    forced = absa_flow(
        source="amazon_reviews_2023_mvp_subset", provider="mock", force=True
    )

    assert forced["input_reviews"] == len(REVIEW_TEXTS)
    assert forced["processed_reviews"] == len(REVIEW_TEXTS)
    assert forced["reviews_with_mentions"] == first["reviews_with_mentions"]
    assert forced["reviews_no_mentions"] == first["reviews_no_mentions"]
    assert forced["inserted_mentions"] == first["inserted_mentions"]

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_status = s.scalar(select(func.count()).select_from(ABSAReviewStatus))
        n_mentions = s.scalar(select(func.count()).select_from(AspectMention))
    assert n_status == len(REVIEW_TEXTS), "force must replace, not duplicate"
    assert n_mentions == first["inserted_mentions"]


def test_invalid_output_is_recorded_as_invalid(seeded_engine, monkeypatch):
    """Provider emitting aspects that all fail validation -> status=invalid,
    no AspectMention rows for that review, but a status row exists."""
    from voicelens.nlp.absa.providers import MockABSAProvider
    from voicelens.nlp.absa.schema import ABSAOutput, AspectMentionOut

    engine, _ids = seeded_engine

    class AlwaysInvalidProvider(MockABSAProvider):
        provider = "mock"
        model_name = "always-invalid"

        def extract(self, review_text: str) -> ABSAOutput:
            # Quote that is never a verbatim substring of any review.
            return ABSAOutput(
                aspects=[
                    AspectMentionOut(
                        aspect_code="battery",
                        sentiment="negative",
                        severity="low",
                        evidence_quote="zzzNEVER-IN-ANY-REVIEWzzz",
                    )
                ]
            )

    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider=AlwaysInvalidProvider(),
        max_invalid_rate=None,
    )

    assert summary["invalid_reviews"] == len(REVIEW_TEXTS)
    assert summary["reviews_with_mentions"] == 0
    assert summary["inserted_mentions"] == 0

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        statuses = list(
            s.execute(
                select(ABSAReviewStatus).where(
                    ABSAReviewStatus.model_name == "always-invalid"
                )
            ).scalars()
        )
        n_mentions = s.scalar(
            select(func.count())
            .select_from(AspectMention)
            .where(AspectMention.model_name == "always-invalid")
        )
    assert len(statuses) == len(REVIEW_TEXTS)
    assert all(row.status == "invalid" for row in statuses)
    assert all(row.n_mentions == 0 for row in statuses)
    assert all(row.n_errors >= 1 for row in statuses)
    assert n_mentions == 0


def test_failed_provider_is_recorded_as_failed(seeded_engine):
    """Provider raising an exception -> status=failed, no rows of mentions."""
    from voicelens.nlp.absa.providers import ABSAProvider
    from voicelens.nlp.absa.schema import ABSAOutput

    engine, _ids = seeded_engine

    class ExplodingProvider(ABSAProvider):
        provider = "exploder"
        model_name = "boom"

        def extract(self, review_text: str) -> ABSAOutput:
            raise RuntimeError("provider blew up")

    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset", provider=ExplodingProvider(), max_fail_rate=None
    )

    assert summary["failed_reviews"] == len(REVIEW_TEXTS)
    assert summary["inserted_mentions"] == 0

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        statuses = list(
            s.execute(
                select(ABSAReviewStatus).where(
                    ABSAReviewStatus.model_name == "boom"
                )
            ).scalars()
        )
    assert len(statuses) == len(REVIEW_TEXTS)
    assert all(row.status == "failed" for row in statuses)
    assert all(row.error_codes_json is not None for row in statuses)


def test_status_row_has_unique_provider_model_combo(seeded_engine):
    """The same review can have multiple status rows for different
    providers — mock and (later) LLM must not collide."""
    from voicelens.nlp.absa.providers import MockABSAProvider

    engine, _ids = seeded_engine
    absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    class OtherProvider(MockABSAProvider):
        provider = "openai"
        model_name = "gpt-4o-mini"

    absa_flow(source="amazon_reviews_2023_mvp_subset", provider=OtherProvider())

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        n_mock = s.scalar(
            select(func.count())
            .select_from(ABSAReviewStatus)
            .where(ABSAReviewStatus.provider == "mock")
        )
        n_llm = s.scalar(
            select(func.count())
            .select_from(ABSAReviewStatus)
            .where(ABSAReviewStatus.provider == "openai")
        )
    assert n_mock == len(REVIEW_TEXTS)
    assert n_llm == len(REVIEW_TEXTS)


def test_processed_coverage_rate_visible_in_summary(seeded_engine):
    engine, _ids = seeded_engine
    summary = absa_flow(source="amazon_reviews_2023_mvp_subset", provider="mock")

    assert summary["processed_reviews"] == len(REVIEW_TEXTS)
    assert summary["reviews_with_mentions"] + summary["reviews_no_mentions"] \
        + summary["invalid_reviews"] + summary["failed_reviews"] \
        == summary["processed_reviews"]


def test_cost_guardrail_stops_processing(seeded_engine):
    from voicelens.nlp.absa.providers import MockABSAProvider

    class CostlyProvider(MockABSAProvider):
        provider = "openai"
        model_name = "costly-test"

        def extract(self, review_text: str):
            out = super().extract(review_text)
            self.usage.estimated_cost_usd = 0.02
            return out

    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider=CostlyProvider(),
        max_cost_usd=0.01,
    )

    assert summary["processed_reviews"] == 1
    assert summary["guardrail_triggered"] is True
    assert "estimated_cost_usd" in summary["guardrail_reason"]


def test_invalid_rate_guardrail_stops_processing(seeded_engine):
    from voicelens.nlp.absa.providers import MockABSAProvider
    from voicelens.nlp.absa.schema import ABSAOutput, AspectMentionOut

    class InvalidProvider(MockABSAProvider):
        provider = "openai"
        model_name = "invalid-rate-test"

        def extract(self, review_text: str) -> ABSAOutput:
            return ABSAOutput(
                aspects=[
                    AspectMentionOut(
                        aspect_code="battery",
                        sentiment="negative",
                        severity="low",
                        evidence_quote="not a verbatim quote",
                    )
                ]
            )

    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider=InvalidProvider(),
        max_invalid_rate=0.10,
        min_processed_for_rate_guardrail=1,
    )

    assert summary["processed_reviews"] == 1
    assert summary["invalid_reviews"] == 1
    assert summary["guardrail_triggered"] is True
    assert "invalid_rate" in summary["guardrail_reason"]


def test_fail_rate_guardrail_stops_processing(seeded_engine):
    from voicelens.nlp.absa.providers import ABSAProvider
    from voicelens.nlp.absa.schema import ABSAOutput

    class FailingProvider(ABSAProvider):
        provider = "anthropic"
        model_name = "fail-rate-test"

        def extract(self, review_text: str) -> ABSAOutput:
            raise RuntimeError("boom")

    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider=FailingProvider(),
        max_fail_rate=0.05,
        min_processed_for_rate_guardrail=1,
    )

    assert summary["processed_reviews"] == 1
    assert summary["failed_reviews"] == 1
    assert summary["guardrail_triggered"] is True
    assert "fail_rate" in summary["guardrail_reason"]


def test_fail_rate_guardrail_waits_for_min_processed(seeded_engine):
    from voicelens.nlp.absa.providers import ABSAProvider
    from voicelens.nlp.absa.schema import ABSAOutput

    class FailingProvider(ABSAProvider):
        provider = "anthropic"
        model_name = "fail-before-min-test"

        def extract(self, review_text: str) -> ABSAOutput:
            raise RuntimeError("boom")

    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider=FailingProvider(),
        max_fail_rate=0.05,
        min_processed_for_rate_guardrail=50,
    )

    assert summary["processed_reviews"] == len(REVIEW_TEXTS)
    assert summary["failed_reviews"] == len(REVIEW_TEXTS)
    assert summary["guardrail_triggered"] is False
    assert summary["min_processed_for_rate_guardrail"] == 50


def test_fail_rate_guardrail_triggers_after_min_processed(seeded_engine):
    from voicelens.nlp.absa.providers import ABSAProvider
    from voicelens.nlp.absa.schema import ABSAOutput

    class FailingProvider(ABSAProvider):
        provider = "anthropic"
        model_name = "fail-after-min-test"

        def extract(self, review_text: str) -> ABSAOutput:
            raise RuntimeError("boom")

    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider=FailingProvider(),
        max_fail_rate=0.05,
        min_processed_for_rate_guardrail=3,
    )

    assert summary["processed_reviews"] == 3
    assert summary["failed_reviews"] == 3
    assert summary["guardrail_triggered"] is True
    assert "fail_rate" in summary["guardrail_reason"]


def test_guardrail_summary_defaults_to_not_triggered(seeded_engine):
    summary = absa_flow(
        source="amazon_reviews_2023_mvp_subset",
        provider="mock",
        limit=1,
    )

    assert summary["guardrail_triggered"] is False
    assert summary["guardrail_reason"] is None
