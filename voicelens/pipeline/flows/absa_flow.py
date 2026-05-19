"""ABSA v1 extraction flow.

Selects reviews from Postgres that have not yet been processed by the
configured provider for the active ontology version, runs the provider,
validates the output, and writes:

- one row in :class:`AspectMention` per surviving aspect, and
- one row in :class:`ABSAReviewStatus` per *processed review* (so reviews
  with ``{"aspects": []}`` are still recorded as "we paid to process this
  and got nothing").

The flow is **idempotent on the review level**: re-running with the same
``(provider, model_name, aspect_version)`` is a no-op for any review that
already has a status row, regardless of whether the prior run produced
zero, one, or many mentions. ``--force`` clears the status row + any
existing mentions for those reviews and reprocesses.

Out of scope for Milestone 2A:
- embeddings / Qdrant indexing
- BERTopic clustering
- anomaly detection
- RAG / LangGraph
- Streamlit
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from typing import Any

from prefect import flow, get_run_logger
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from voicelens.db.engine import get_session
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    IngestRun,
    Review,
)
from voicelens.nlp.absa import (
    ONTOLOGY_CODES_V1,
    ABSAProvider,
    get_provider,
)
from voicelens.nlp.absa.extractor import extract_for_review
from voicelens.nlp.absa.validators import (
    ERR_DUPLICATE_ASPECT,
    ERR_NON_VERBATIM,
    ERR_NOT_IN_ONTOLOGY,
    ERR_SCHEMA,
)

DEFAULT_ONTOLOGY_VERSION = "v1"
DEFAULT_BATCH_SIZE = 200


def _load_ontology(session: Session, version: str) -> dict[str, int]:
    """Return ``{aspect_code: aspect_ontology.id}`` for the active version."""
    rows = session.execute(
        select(AspectOntology).where(AspectOntology.version == version)
    ).scalars().all()
    if not rows:
        raise RuntimeError(
            f"aspect_ontology has no rows for version={version!r}. "
            f"Run `make seed-aspects` (or `python scripts/seed_aspect_ontology.py`) first."
        )
    return {row.code: row.id for row in rows}


def _select_review_ids(
    session: Session,
    *,
    source: str | None,
    aspect_version: str,
    provider_name: str,
    model_name: str,
    limit: int | None,
    force: bool,
) -> list[int]:
    """Resolve which Review rows still need ABSA processing.

    Without ``--force``: skip reviews that already have an
    ``absa_review_status`` row for the same
    ``(aspect_version, provider, model_name)`` — regardless of whether the
    prior run produced mentions or zero. With ``--force``: process
    everything matching the source filter.
    """
    stmt = select(Review.id).order_by(Review.id)
    if source is not None:
        stmt = stmt.where(
            Review.ingest_run_id.in_(
                select(IngestRun.id).where(IngestRun.source == source)
            )
        )
    if not force:
        already = (
            select(ABSAReviewStatus.review_id)
            .where(
                and_(
                    ABSAReviewStatus.aspect_version == aspect_version,
                    ABSAReviewStatus.provider == provider_name,
                    ABSAReviewStatus.model_name == model_name,
                )
            )
            .distinct()
        )
        stmt = stmt.where(Review.id.notin_(already))
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def _delete_existing(
    session: Session,
    review_ids: Iterable[int],
    aspect_version: str,
    provider_name: str,
    model_name: str,
) -> tuple[int, int]:
    """Used by --force: remove prior status + mentions for these reviews.

    Returns ``(n_status_deleted, n_mentions_deleted)``.
    """
    ids = list(review_ids)
    if not ids:
        return 0, 0

    status_rows = session.execute(
        select(ABSAReviewStatus).where(
            ABSAReviewStatus.review_id.in_(ids),
            ABSAReviewStatus.aspect_version == aspect_version,
            ABSAReviewStatus.provider == provider_name,
            ABSAReviewStatus.model_name == model_name,
        )
    ).scalars().all()
    for row in status_rows:
        session.delete(row)

    mention_rows = session.execute(
        select(AspectMention).where(
            AspectMention.review_id.in_(ids),
            AspectMention.aspect_version == aspect_version,
            AspectMention.model_name == model_name,
        )
    ).scalars().all()
    for row in mention_rows:
        session.delete(row)

    if status_rows or mention_rows:
        session.flush()
    return len(status_rows), len(mention_rows)


def _summary_skeleton() -> dict[str, Any]:
    return {
        "input_reviews": 0,
        "processed_reviews": 0,
        "reviews_with_mentions": 0,
        "reviews_no_mentions": 0,
        "invalid_reviews": 0,
        "failed_reviews": 0,
        "inserted_mentions": 0,
        "invalid_outputs": 0,
        "aspect_counts": {},
        "sentiment_counts": {},
        "severity_counts": {},
        "evidence_verbatim_rate": 0.0,
        "validation_errors": {
            ERR_SCHEMA: 0,
            ERR_NOT_IN_ONTOLOGY: 0,
            ERR_NON_VERBATIM: 0,
            ERR_DUPLICATE_ASPECT: 0,
        },
    }


def _record_status(
    session: Session,
    *,
    review_id: int,
    aspect_version: str,
    provider_name: str,
    model_name: str,
    status: str,
    n_mentions: int,
    n_errors: int,
    error_codes: dict[str, int] | None,
) -> None:
    session.add(
        ABSAReviewStatus(
            review_id=review_id,
            aspect_version=aspect_version,
            provider=provider_name,
            model_name=model_name,
            status=status,
            n_mentions=n_mentions,
            n_errors=n_errors,
            error_codes_json=error_codes,
        )
    )


def _classify_review_status(
    *, n_valid: int, n_errors: int, raw_aspect_count: int
) -> str:
    """Map per-review extraction outcome to an ``ABSAReviewStatus.status``.

    - ``success``: at least one valid mention survived validation
    - ``invalid``: provider emitted aspects but none survived validation
    - ``no_mentions``: provider emitted no aspects and no errors
    """
    if n_valid > 0:
        return ABSAReviewStatus.STATUS_SUCCESS
    if raw_aspect_count > 0 or n_errors > 0:
        return ABSAReviewStatus.STATUS_INVALID
    return ABSAReviewStatus.STATUS_NO_MENTIONS


@flow(name="absa_flow")
def absa_flow(
    *,
    source: str | None = None,
    limit: int | None = None,
    provider: str | ABSAProvider = "mock",
    force: bool = False,
    aspect_version: str = DEFAULT_ONTOLOGY_VERSION,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    try:
        logger = get_run_logger()
    except Exception:  # pragma: no cover - prefect context missing
        logger = logging.getLogger("absa_flow")
    absa: ABSAProvider = provider if isinstance(provider, ABSAProvider) else get_provider(provider, model=model)
    provider_name = absa.provider
    model_name = absa.model_name

    aspect_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    summary = _summary_skeleton()

    raw_quote_total = 0

    with get_session() as session:
        ontology = _load_ontology(session, aspect_version)
        ontology_codes = tuple(ontology.keys()) or ONTOLOGY_CODES_V1

        review_ids = _select_review_ids(
            session,
            source=source,
            aspect_version=aspect_version,
            provider_name=provider_name,
            model_name=model_name,
            limit=limit,
            force=force,
        )
        summary["input_reviews"] = len(review_ids)
        if not review_ids:
            logger.info(
                "absa_flow: nothing to do (source=%s, force=%s, provider=%s, model_name=%s)",
                source, force, provider_name, model_name,
            )
            summary["aspect_counts"] = dict(aspect_counts)
            summary["sentiment_counts"] = dict(sentiment_counts)
            summary["severity_counts"] = dict(severity_counts)
            summary["provider"] = provider_name
            summary["model_name"] = model_name
            summary["aspect_version"] = aspect_version
            return summary

        if force:
            _delete_existing(session, review_ids, aspect_version, provider_name, model_name)

        logger.info(
            "absa_flow: processing %d reviews | provider=%s | model=%s | force=%s | version=%s",
            len(review_ids), provider_name, model_name, force, aspect_version,
        )

        for start in range(0, len(review_ids), batch_size):
            batch_ids = review_ids[start : start + batch_size]
            reviews = list(
                session.execute(
                    select(Review).where(Review.id.in_(batch_ids))
                ).scalars()
            )
            by_id = {r.id: r for r in reviews}
            for review_id in batch_ids:
                review = by_id.get(review_id)
                if review is None:
                    continue
                summary["processed_reviews"] += 1

                try:
                    outcome = extract_for_review(absa, review.id, review.text_raw, ontology_codes)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "absa_flow: provider raised on review_id=%s: %s", review.id, exc
                    )
                    summary["failed_reviews"] += 1
                    _record_status(
                        session,
                        review_id=review.id,
                        aspect_version=aspect_version,
                        provider_name=provider_name,
                        model_name=model_name,
                        status=ABSAReviewStatus.STATUS_FAILED,
                        n_mentions=0,
                        n_errors=0,
                        error_codes={"exception": type(exc).__name__},
                    )
                    continue

                raw_quote_total += outcome.raw_aspect_count
                error_codes: dict[str, int] = {}
                for err in outcome.result.errors:
                    summary["validation_errors"][err.code] = (
                        summary["validation_errors"].get(err.code, 0) + 1
                    )
                    summary["invalid_outputs"] += 1
                    error_codes[err.code] = error_codes.get(err.code, 0) + 1

                n_valid = len(outcome.result.valid_mentions)
                if n_valid:
                    summary["reviews_with_mentions"] += 1
                    for mention in outcome.result.valid_mentions:
                        aspect_id = ontology.get(mention.aspect_code)
                        if aspect_id is None:
                            continue
                        session.add(
                            AspectMention(
                                review_id=review.id,
                                aspect_id=aspect_id,
                                sentiment=mention.sentiment,
                                severity=mention.severity,
                                evidence_quote=mention.evidence_quote,
                                model_name=model_name,
                                aspect_version=aspect_version,
                            )
                        )
                        summary["inserted_mentions"] += 1
                        aspect_counts[mention.aspect_code] += 1
                        sentiment_counts[mention.sentiment] += 1
                        if mention.severity is not None:
                            severity_counts[mention.severity] += 1

                status = _classify_review_status(
                    n_valid=n_valid,
                    n_errors=len(outcome.result.errors),
                    raw_aspect_count=outcome.raw_aspect_count,
                )
                if status == ABSAReviewStatus.STATUS_NO_MENTIONS:
                    summary["reviews_no_mentions"] += 1
                elif status == ABSAReviewStatus.STATUS_INVALID:
                    summary["invalid_reviews"] += 1

                _record_status(
                    session,
                    review_id=review.id,
                    aspect_version=aspect_version,
                    provider_name=provider_name,
                    model_name=model_name,
                    status=status,
                    n_mentions=n_valid,
                    n_errors=len(outcome.result.errors),
                    error_codes=error_codes or None,
                )
            session.flush()

    non_verbatim = summary["validation_errors"].get(ERR_NON_VERBATIM, 0)
    summary["evidence_verbatim_rate"] = (
        round((raw_quote_total - non_verbatim) / raw_quote_total, 4) if raw_quote_total else 1.0
    )
    summary["aspect_counts"] = dict(aspect_counts.most_common())
    summary["sentiment_counts"] = dict(sentiment_counts.most_common())
    summary["severity_counts"] = dict(severity_counts.most_common())
    summary["raw_aspect_candidates"] = raw_quote_total
    summary["provider"] = provider_name
    summary["model_name"] = model_name
    summary["aspect_version"] = aspect_version
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABSA v1 on stored reviews.")
    parser.add_argument("--source", default=None, help="ingest_run.source filter (e.g. amazon_reviews_2023_mvp_subset)")
    parser.add_argument("--limit", type=int, default=None, help="Max reviews to process this run")
    parser.add_argument("--provider", default="mock", help="ABSA provider name: mock | openai | anthropic")
    parser.add_argument("--model", default=None, help="Optional model identifier passed to provider")
    parser.add_argument("--aspect-version", default=DEFAULT_ONTOLOGY_VERSION, help="aspect_ontology.version to target")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Reviews per DB fetch batch")
    parser.add_argument("--force", action="store_true", help="Reprocess reviews even if they already have a status row for this provider+version")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = absa_flow(
        source=args.source,
        limit=args.limit,
        provider=args.provider,
        force=args.force,
        aspect_version=args.aspect_version,
        model=args.model,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
