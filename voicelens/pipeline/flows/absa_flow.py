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
from pathlib import Path
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


def _read_review_ids_file(path: str | Path) -> tuple[list[int], list[str]]:
    """Read a JSONL file and return ``(review_ids, source_ids)`` lists.

    Each line is a JSON object. ``review_id`` is preferred; if missing, the
    row's ``source_id`` is collected as a fallback resolver in the DB. Lines
    without either field are skipped. Order is preserved and duplicates
    are de-duplicated while keeping first occurrence.
    """
    review_ids: list[int] = []
    source_ids: list[str] = []
    seen_rids: set[int] = set()
    seen_sids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"--review-ids-file {path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                continue
            rid = row.get("review_id")
            if rid is not None:
                try:
                    rid_int = int(rid)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"--review-ids-file {path}:{line_no} has non-int review_id={rid!r}"
                    ) from exc
                if rid_int not in seen_rids:
                    review_ids.append(rid_int)
                    seen_rids.add(rid_int)
                continue
            sid = row.get("source_id")
            if isinstance(sid, str) and sid and sid not in seen_sids:
                source_ids.append(sid)
                seen_sids.add(sid)
    if not review_ids and not source_ids:
        raise ValueError(
            f"--review-ids-file {path} contained no review_id or source_id entries"
        )
    return review_ids, source_ids


def _resolve_review_ids(
    session: Session, review_ids: list[int], source_ids: list[str]
) -> tuple[list[int], list[str]]:
    """Resolve a ``(review_ids, source_ids)`` selection against ``Review``.

    Returns ``(found_ids, missing_descriptions)``. ``found_ids`` is the set
    of ``Review.id`` values that exist in the database; ``missing_descriptions``
    lists the raw selectors that could not be resolved (e.g. a source_id
    that doesn't match any review).
    """
    found: list[int] = []
    missing: list[str] = []
    if review_ids:
        existing = set(
            session.execute(
                select(Review.id).where(Review.id.in_(review_ids))
            ).scalars()
        )
        for rid in review_ids:
            if rid in existing:
                found.append(rid)
            else:
                missing.append(f"review_id={rid}")
    if source_ids:
        rows = session.execute(
            select(Review.id, Review.source_id).where(Review.source_id.in_(source_ids))
        ).all()
        by_sid = {row.source_id: int(row.id) for row in rows}
        for sid in source_ids:
            rid = by_sid.get(sid)
            if rid is None:
                missing.append(f"source_id={sid!r}")
                continue
            if rid not in found:
                found.append(rid)
    return found, missing


def _select_review_ids(
    session: Session,
    *,
    source: str | None,
    aspect_version: str,
    provider_name: str,
    model_name: str,
    limit: int | None,
    force: bool,
    explicit_review_ids: list[int] | None = None,
) -> list[int]:
    """Resolve which Review rows still need ABSA processing.

    Without ``--force``: skip reviews that already have an
    ``absa_review_status`` row for the same
    ``(aspect_version, provider, model_name)`` — regardless of whether the
    prior run produced mentions or zero. With ``--force``: process
    everything matching the source filter (or ``explicit_review_ids``).

    When ``explicit_review_ids`` is provided (from ``--review-ids-file``),
    the source filter is ignored and the input list is used as the
    candidate set. Order from the input list is preserved (not by
    ``Review.id``) so callers control the processing order. The
    idempotency and ``--force`` semantics are unchanged.
    """
    if explicit_review_ids is not None:
        if not explicit_review_ids:
            return []
        if force:
            return list(explicit_review_ids[:limit] if limit and limit > 0 else explicit_review_ids)
        already = set(
            session.execute(
                select(ABSAReviewStatus.review_id)
                .where(
                    and_(
                        ABSAReviewStatus.aspect_version == aspect_version,
                        ABSAReviewStatus.provider == provider_name,
                        ABSAReviewStatus.model_name == model_name,
                        ABSAReviewStatus.review_id.in_(explicit_review_ids),
                    )
                )
                .distinct()
            ).scalars()
        )
        out = [rid for rid in explicit_review_ids if rid not in already]
        if limit is not None and limit > 0:
            out = out[:limit]
        return out

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
        "guardrail_triggered": False,
        "guardrail_reason": None,
    }


def _attach_provider_usage(summary: dict[str, Any], provider: ABSAProvider) -> None:
    usage = provider.usage_summary()
    summary["provider_usage"] = usage
    summary["total_input_tokens"] = usage.get("total_input_tokens")
    summary["total_output_tokens"] = usage.get("total_output_tokens")
    summary["estimated_cost_usd"] = usage.get("estimated_cost_usd")
    summary["avg_latency_ms"] = usage.get("avg_latency_ms")
    summary["p95_latency_ms"] = usage.get("p95_latency_ms")


def _guardrail_reason(
    summary: dict[str, Any],
    provider: ABSAProvider,
    *,
    max_cost_usd: float | None,
    max_invalid_rate: float | None,
    max_fail_rate: float | None,
    min_processed_for_rate_guardrail: int,
) -> str | None:
    processed = int(summary.get("processed_reviews") or 0)
    usage = provider.usage_summary()
    estimated_cost = usage.get("estimated_cost_usd")
    if max_cost_usd is not None and estimated_cost is not None and estimated_cost > max_cost_usd:
        return f"estimated_cost_usd {estimated_cost:.4f} exceeded max_cost_usd {max_cost_usd:.4f}"
    if processed >= min_processed_for_rate_guardrail and max_invalid_rate is not None:
        invalid_rate = float(summary.get("invalid_reviews") or 0) / processed
        if invalid_rate > max_invalid_rate:
            return f"invalid_rate {invalid_rate:.4f} exceeded max_invalid_rate {max_invalid_rate:.4f}"
    if processed >= min_processed_for_rate_guardrail and max_fail_rate is not None:
        fail_rate = float(summary.get("failed_reviews") or 0) / processed
        if fail_rate > max_fail_rate:
            return f"fail_rate {fail_rate:.4f} exceeded max_fail_rate {max_fail_rate:.4f}"
    return None


def _check_guardrails(
    summary: dict[str, Any],
    provider: ABSAProvider,
    *,
    max_cost_usd: float | None,
    max_invalid_rate: float | None,
    max_fail_rate: float | None,
    min_processed_for_rate_guardrail: int,
    stop_on_guardrail: bool,
) -> bool:
    reason = _guardrail_reason(
        summary,
        provider,
        max_cost_usd=max_cost_usd,
        max_invalid_rate=max_invalid_rate,
        max_fail_rate=max_fail_rate,
        min_processed_for_rate_guardrail=min_processed_for_rate_guardrail,
    )
    if reason is None:
        return False
    summary["guardrail_triggered"] = True
    summary["guardrail_reason"] = reason
    return stop_on_guardrail


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
    max_cost_usd: float | None = None,
    max_invalid_rate: float | None = 0.10,
    max_fail_rate: float | None = 0.05,
    stop_on_guardrail: bool = True,
    llm_max_retries: int = 2,
    llm_retry_base_seconds: float = 1.0,
    min_processed_for_rate_guardrail: int = 50,
    review_ids_file: str | Path | None = None,
) -> dict[str, Any]:
    try:
        logger = get_run_logger()
    except Exception:  # pragma: no cover - prefect context missing
        logger = logging.getLogger("absa_flow")
    absa: ABSAProvider = provider if isinstance(provider, ABSAProvider) else get_provider(provider, model=model)
    absa.configure_retries(max_retries=llm_max_retries, base_seconds=llm_retry_base_seconds)
    provider_name = absa.provider
    model_name = absa.model_name

    aspect_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    summary = _summary_skeleton()
    summary["min_processed_for_rate_guardrail"] = min_processed_for_rate_guardrail

    raw_quote_total = 0

    with get_session() as session:
        ontology = _load_ontology(session, aspect_version)
        ontology_codes = tuple(ontology.keys()) or ONTOLOGY_CODES_V1

        explicit_ids: list[int] | None = None
        if review_ids_file is not None:
            raw_review_ids, raw_source_ids = _read_review_ids_file(review_ids_file)
            resolved_ids, missing = _resolve_review_ids(
                session, raw_review_ids, raw_source_ids
            )
            if missing:
                logger.warning(
                    "absa_flow: %d entries in --review-ids-file could not be resolved "
                    "(first 5: %s)",
                    len(missing), missing[:5],
                )
            summary["review_ids_file"] = str(review_ids_file)
            summary["review_ids_file_requested"] = len(raw_review_ids) + len(raw_source_ids)
            summary["review_ids_file_resolved"] = len(resolved_ids)
            summary["review_ids_file_unresolved"] = len(missing)
            explicit_ids = resolved_ids

        review_ids = _select_review_ids(
            session,
            source=source,
            aspect_version=aspect_version,
            provider_name=provider_name,
            model_name=model_name,
            limit=limit,
            force=force,
            explicit_review_ids=explicit_ids,
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
            summary["retry_attempts_total"] = 0
            summary["recovered_after_retry"] = 0
            _attach_provider_usage(summary, absa)
            return summary

        if force:
            _delete_existing(session, review_ids, aspect_version, provider_name, model_name)

        logger.info(
            "absa_flow: processing %d reviews | provider=%s | model=%s | force=%s | version=%s",
            len(review_ids), provider_name, model_name, force, aspect_version,
        )

        stop_processing = False
        for start in range(0, len(review_ids), batch_size):
            if stop_processing:
                break
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
                    stop_processing = _check_guardrails(
                        summary,
                        absa,
                        max_cost_usd=max_cost_usd,
                        max_invalid_rate=max_invalid_rate,
                        max_fail_rate=max_fail_rate,
                        min_processed_for_rate_guardrail=min_processed_for_rate_guardrail,
                        stop_on_guardrail=stop_on_guardrail,
                    )
                    if stop_processing:
                        break
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
                stop_processing = _check_guardrails(
                    summary,
                    absa,
                    max_cost_usd=max_cost_usd,
                    max_invalid_rate=max_invalid_rate,
                    max_fail_rate=max_fail_rate,
                    min_processed_for_rate_guardrail=min_processed_for_rate_guardrail,
                    stop_on_guardrail=stop_on_guardrail,
                )
                if stop_processing:
                    break
            session.flush()

    non_verbatim = summary["validation_errors"].get(ERR_NON_VERBATIM, 0)
    summary["evidence_verbatim_rate"] = (
        round((raw_quote_total - non_verbatim) / raw_quote_total, 4) if raw_quote_total else 1.0
    )
    summary["aspect_counts"] = dict(aspect_counts.most_common())
    summary["sentiment_counts"] = dict(sentiment_counts.most_common())
    summary["severity_counts"] = dict(severity_counts.most_common())
    summary["raw_aspect_candidates"] = raw_quote_total
    usage = absa.usage_summary()
    summary["retry_attempts_total"] = usage.get("retry_attempts_total")
    summary["recovered_after_retry"] = usage.get("recovered_after_retry")
    summary["provider"] = provider_name
    summary["model_name"] = model_name
    summary["aspect_version"] = aspect_version
    _attach_provider_usage(summary, absa)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABSA v1 on stored reviews.")
    parser.add_argument("--source", default=None, help="ingest_run.source filter (e.g. amazon_reviews_2023_mvp_subset)")
    parser.add_argument("--limit", type=int, default=None, help="Max reviews to process this run")
    parser.add_argument("--provider", default="mock", help="ABSA provider name: mock | openai | anthropic")
    parser.add_argument("--model", default=None, help="Optional model identifier passed to provider")
    parser.add_argument("--aspect-version", default=DEFAULT_ONTOLOGY_VERSION, help="aspect_ontology.version to target")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Reviews per DB fetch batch")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="Stop when estimated provider cost exceeds this amount, if cost is available")
    parser.add_argument("--max-invalid-rate", type=float, default=0.10, help="Stop when invalid review rate exceeds this threshold")
    parser.add_argument("--max-fail-rate", type=float, default=0.05, help="Stop when provider failure rate exceeds this threshold")
    parser.add_argument("--llm-max-retries", type=int, default=2, help="Max retries for transient LLM gateway/network errors")
    parser.add_argument("--llm-retry-base-seconds", type=float, default=1.0, help="Base seconds for exponential retry backoff")
    parser.add_argument("--min-processed-for-rate-guardrail", type=int, default=50, help="Do not evaluate invalid/fail-rate guardrails until this many reviews are processed")
    parser.add_argument("--stop-on-guardrail", dest="stop_on_guardrail", action="store_true", default=True, help="Stop processing when a guardrail is exceeded")
    parser.add_argument("--no-stop-on-guardrail", dest="stop_on_guardrail", action="store_false", help="Record guardrail breach but continue processing")
    parser.add_argument("--force", action="store_true", help="Reprocess reviews even if they already have a status row for this provider+version")
    parser.add_argument(
        "--review-ids-file",
        default=None,
        help=(
            "Path to a JSONL file whose rows carry review_id (preferred) or source_id. "
            "When set, --source is ignored and only the listed reviews are processed."
        ),
    )
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
        max_cost_usd=args.max_cost_usd,
        max_invalid_rate=args.max_invalid_rate,
        max_fail_rate=args.max_fail_rate,
        stop_on_guardrail=args.stop_on_guardrail,
        llm_max_retries=args.llm_max_retries,
        llm_retry_base_seconds=args.llm_retry_base_seconds,
        min_processed_for_rate_guardrail=args.min_processed_for_rate_guardrail,
        review_ids_file=args.review_ids_file,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
