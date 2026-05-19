"""Print a quick summary of what's in the ``aspect_mention`` and
``absa_review_status`` tables.

Run after ``make absa-smoke`` to confirm the flow inserted reasonable
rows and to verify processed coverage is rising over re-runs.
"""
from __future__ import annotations

import sys

from sqlalchemy import func, select

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import get_session
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Review,
    Sku,
)


def collect_absa_stats() -> dict:
    out: dict = {}
    with get_session() as s:
        total_mentions = s.scalar(select(func.count()).select_from(AspectMention)) or 0
        total_reviews = s.scalar(select(func.count()).select_from(Review)) or 0

        out["total_aspect_mentions"] = int(total_mentions)
        out["total_reviews"] = int(total_reviews)

        # processed-coverage view from absa_review_status
        status_rows = s.execute(
            select(ABSAReviewStatus.status, func.count())
            .group_by(ABSAReviewStatus.status)
        ).all()
        status_counts = {status: int(n) for status, n in status_rows}
        processed = int(
            s.scalar(
                select(func.count(func.distinct(ABSAReviewStatus.review_id)))
            )
            or 0
        )
        out["processed_reviews"] = processed
        out["reviews_with_mentions"] = status_counts.get("success", 0)
        out["reviews_no_mentions"] = status_counts.get("no_mentions", 0)
        out["invalid_reviews"] = status_counts.get("invalid", 0)
        out["failed_reviews"] = status_counts.get("failed", 0)
        out["status_counts"] = status_counts

        out["processed_coverage_rate"] = (
            round(processed / total_reviews, 4) if total_reviews else 0.0
        )
        out["mention_coverage_rate"] = (
            round(out["reviews_with_mentions"] / total_reviews, 4) if total_reviews else 0.0
        )

        verbatim_hits = 0
        if total_mentions:
            for quote, text in s.execute(
                select(AspectMention.evidence_quote, Review.text_raw)
                .join(Review, Review.id == AspectMention.review_id)
            ):
                if quote and text and quote in text:
                    verbatim_hits += 1
        # ``valid_mention_evidence_verbatim_rate`` is computed over rows
        # that already survived validation (i.e. inserted ``aspect_mention``
        # rows). The validator drops non-verbatim quotes before insert, so
        # this should be 1.0 in practice — the rate is reported anyway
        # because it doubles as a self-consistency check on the validator.
        # For the "raw-output" view of verbatim rate (i.e. what fraction of
        # the LLM's raw aspects passed the verbatim check), see the
        # ``raw_output_evidence_verbatim_rate`` field on absa_flow's
        # per-run JSON summary.
        valid_rate = round(verbatim_hits / total_mentions, 4) if total_mentions else 1.0
        out["valid_mention_evidence_verbatim_rate"] = valid_rate
        # Legacy alias for older dashboards / notebooks that read the
        # original key. Will be removed in M4 once nothing reads it.
        out["evidence_verbatim_rate"] = valid_rate

        aspect_rows = s.execute(
            select(AspectOntology.code, func.count(AspectMention.id))
            .join(AspectMention, AspectMention.aspect_id == AspectOntology.id)
            .group_by(AspectOntology.code)
            .order_by(func.count(AspectMention.id).desc())
        ).all()
        out["aspect_distribution"] = {code: int(n) for code, n in aspect_rows}

        sentiment_rows = s.execute(
            select(AspectMention.sentiment, func.count())
            .group_by(AspectMention.sentiment)
            .order_by(func.count().desc())
        ).all()
        out["sentiment_distribution"] = {sent: int(n) for sent, n in sentiment_rows}

        severity_rows = s.execute(
            select(AspectMention.severity, func.count())
            .where(AspectMention.severity.isnot(None))
            .group_by(AspectMention.severity)
            .order_by(func.count().desc())
        ).all()
        out["severity_distribution"] = {sev: int(n) for sev, n in severity_rows}

        top_neg = s.execute(
            select(Brand.name, func.count(AspectMention.id))
            .join(Sku, Sku.brand_id == Brand.id)
            .join(Review, Review.sku_id == Sku.id)
            .join(AspectMention, AspectMention.review_id == Review.id)
            .where(AspectMention.sentiment == "negative")
            .group_by(Brand.name)
            .order_by(func.count(AspectMention.id).desc())
            .limit(10)
        ).all()
        out["top_brands_by_negative_mentions"] = {
            name: int(n) for name, n in top_neg
        }
    return out


def _print(stats: dict) -> None:
    print("== VoiceLens ABSA stats ==")
    print(f"  total_aspect_mentions    : {stats['total_aspect_mentions']}")
    print(f"  total_reviews            : {stats['total_reviews']}")
    print(f"  processed_reviews        : {stats['processed_reviews']}")
    print(f"  reviews_with_mentions    : {stats['reviews_with_mentions']}")
    print(f"  reviews_no_mentions      : {stats['reviews_no_mentions']}")
    print(f"  invalid_reviews          : {stats['invalid_reviews']}")
    print(f"  failed_reviews           : {stats['failed_reviews']}")
    print(f"  processed_coverage_rate  : {stats['processed_coverage_rate']}")
    print(f"  mention_coverage_rate    : {stats['mention_coverage_rate']}")
    print(f"  valid_mention_evidence_verbatim_rate : {stats['valid_mention_evidence_verbatim_rate']}")

    print("\n-- aspect distribution --")
    if stats["aspect_distribution"]:
        for code, n in stats["aspect_distribution"].items():
            print(f"  {code:<16} {n}")
    else:
        print("  (no aspect_mention rows)")

    print("\n-- sentiment distribution --")
    if stats["sentiment_distribution"]:
        for sent, n in stats["sentiment_distribution"].items():
            print(f"  {sent:<10} {n}")
    else:
        print("  (no rows)")

    print("\n-- severity distribution --")
    if stats["severity_distribution"]:
        for sev, n in stats["severity_distribution"].items():
            print(f"  {sev:<10} {n}")
    else:
        print("  (no negative-sentiment rows)")

    print("\n-- top brands by negative mentions --")
    if stats["top_brands_by_negative_mentions"]:
        for name, n in stats["top_brands_by_negative_mentions"].items():
            print(f"  {name:<16} {n}")
    else:
        print("  (no negative mentions)")


def main() -> int:
    stats = collect_absa_stats()
    _print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
