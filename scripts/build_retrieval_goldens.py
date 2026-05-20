"""Build the weakly supervised retrieval golden set.

For each natural-language query template the script:

1. Filters ``aspect_mention`` rows by the template's
   ``expected_aspect`` / ``expected_sentiment`` (when set).
2. Restricts further to mentions whose ``evidence_quote`` contains any
   of a small phrase list closely aligned with the query.
3. Joins back to ``review`` (under the configured aspect_version /
   provider / model_name) and takes up to ``--per-query`` review_ids
   per template.

The output is intentionally labelled **weakly supervised**: the gold
ids are seeded from the ABSA pipeline itself, so the eval primarily
measures retrieval consistency relative to the LLM's labels, not
ground truth in the strictest sense. A human-refinement pass can edit
``data/eval/retrieval_goldens.jsonl`` afterwards without re-running
this script.

The file is gitignored by default (under ``data/eval/``); rerun the
script when the underlying index changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

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


@dataclass
class QueryTemplate:
    query_id: str
    query: str
    expected_aspect: str | None = None
    expected_sentiment: str | None = None
    expected_brand: str | None = None
    quote_phrases: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


# ~50 templates spanning the v2 ontology + a few facet/brand variants.
# Phrases use case-insensitive ``LIKE %phrase%`` against evidence_quote
# so the join is grounded in what the LLM actually surfaced as
# evidence, not just the aspect label.
QUERY_TEMPLATES: tuple[QueryTemplate, ...] = (
    # reliability
    QueryTemplate(
        "rel-stopped-working", "reviews where product stopped working after a week",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("stopped working", "stopped after", "stopped charging", "no longer works"),
    ),
    QueryTemplate(
        "rel-dead-on-arrival", "dead on arrival",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("dead on arrival", "doa", "didn't work out of the box", "did not work out of the box"),
    ),
    QueryTemplate(
        "rel-broke-quickly", "broke after a few days",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("broke after", "broken after", "fell apart", "snapped"),
    ),
    QueryTemplate(
        "rel-defective", "defective unit",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("defective", "faulty", "malfunction"),
    ),
    QueryTemplate(
        "rel-lasted-only", "only lasted a few weeks",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("lasted only", "lasted just", "lasted a few", "only lasted"),
    ),
    QueryTemplate(
        "rel-died-quickly", "device died quickly",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("died after", "died on me", "just died"),
    ),
    # bluetooth
    QueryTemplate(
        "bt-disconnects", "bluetooth keeps disconnecting",
        expected_aspect="bluetooth", expected_sentiment="negative",
        quote_phrases=("disconnect", "disconnects", "drops connection", "loses connection", "won't stay connected", "wont stay connected"),
    ),
    QueryTemplate(
        "bt-pairing-issues", "pairing problems",
        expected_aspect="bluetooth", expected_sentiment="negative",
        quote_phrases=("won't pair", "wont pair", "pairing", "won't connect", "wont connect"),
    ),
    QueryTemplate(
        "bt-range", "bluetooth range is short",
        expected_aspect="bluetooth", expected_sentiment="negative",
        quote_phrases=("range", "short range", "limited range"),
    ),
    QueryTemplate(
        "bt-latency", "bluetooth audio lag or latency",
        expected_aspect="bluetooth", expected_sentiment="negative",
        quote_phrases=("latency", "lag", "delay"),
    ),
    # charging
    QueryTemplate(
        "chg-stopped", "charging stopped working",
        expected_aspect="charging", expected_sentiment="negative",
        quote_phrases=("won't charge", "wont charge", "stopped charging", "no longer charges", "no longer charge"),
    ),
    QueryTemplate(
        "chg-usbc", "USB-C charging port issue",
        expected_aspect="charging", expected_sentiment="negative",
        quote_phrases=("usb-c", "usb c", "type c", "type-c", "port"),
    ),
    QueryTemplate(
        "chg-slow", "slow charging",
        expected_aspect="charging", expected_sentiment="negative",
        quote_phrases=("slow charge", "slow charging", "takes forever to charge"),
    ),
    QueryTemplate(
        "chg-cable-broken", "charging cable broke",
        expected_aspect="charging", expected_sentiment="negative",
        quote_phrases=("cable broke", "cable broken", "frayed", "fraying", "cord broke"),
    ),
    QueryTemplate(
        "chg-fast", "fast charging works well",
        expected_aspect="charging", expected_sentiment="positive",
        quote_phrases=("fast charge", "fast charging", "charges quickly", "charges fast"),
    ),
    # price
    QueryTemplate(
        "price-expensive", "expensive not worth the price",
        expected_aspect="price", expected_sentiment="negative",
        quote_phrases=("not worth", "too expensive", "overpriced", "for the price"),
    ),
    QueryTemplate(
        "price-bad-value", "poor value for money",
        expected_aspect="price", expected_sentiment="negative",
        quote_phrases=("waste of money", "you get what you pay for", "don't waste", "dont waste"),
    ),
    QueryTemplate(
        "price-good-value", "good value for the price",
        expected_aspect="price", expected_sentiment="positive",
        quote_phrases=("worth the money", "great price", "good price", "great value", "good value"),
    ),
    QueryTemplate(
        "price-cheap-feel", "feels cheap",
        expected_aspect="price", expected_sentiment="negative",
        quote_phrases=("feels cheap", "cheap quality", "cheap plastic"),
    ),
    # delivery
    QueryTemplate(
        "del-damaged-pkg", "package arrived damaged",
        expected_aspect="delivery", expected_sentiment="negative",
        quote_phrases=("damaged", "package was", "packaging was", "arrived damaged"),
    ),
    QueryTemplate(
        "del-missing-item", "missing item in package",
        expected_aspect="delivery", expected_sentiment="negative",
        quote_phrases=("missing", "wasn't included", "wasnt included", "didn't come with", "didnt come with"),
    ),
    QueryTemplate(
        "del-slow-shipping", "slow shipping",
        expected_aspect="delivery", expected_sentiment="negative",
        quote_phrases=("slow ship", "shipping was slow", "took forever to arrive", "long time to arrive"),
    ),
    QueryTemplate(
        "del-fast-shipping", "shipping was fast",
        expected_aspect="delivery", expected_sentiment="positive",
        quote_phrases=("fast ship", "shipped quickly", "arrived quickly", "fast delivery"),
    ),
    # sound_quality
    QueryTemplate(
        "snd-bad-quality", "bad sound quality",
        expected_aspect="sound_quality", expected_sentiment="negative",
        quote_phrases=("tinny", "muffled", "distorted", "sounds bad"),
    ),
    QueryTemplate(
        "snd-low-volume", "low volume",
        expected_aspect="sound_quality", expected_sentiment="negative",
        quote_phrases=("low volume", "not loud", "too quiet", "very quiet"),
    ),
    QueryTemplate(
        "snd-distorted", "distorted audio",
        expected_aspect="sound_quality", expected_sentiment="negative",
        quote_phrases=("distort", "crackle", "static"),
    ),
    QueryTemplate(
        "snd-great-sound", "great sound quality",
        expected_aspect="sound_quality", expected_sentiment="positive",
        quote_phrases=("sounds great", "great sound", "amazing sound", "fantastic sound"),
    ),
    QueryTemplate(
        "snd-mic-bad", "microphone quality is bad",
        expected_aspect="sound_quality", expected_sentiment="negative",
        quote_phrases=("microphone", "mic ", "mic.", "mic,", "mic-", "mic;"),
    ),
    # overheating
    QueryTemplate(
        "heat-gets-hot", "device gets very hot",
        expected_aspect="overheating", expected_sentiment="negative",
        quote_phrases=("too hot", "gets hot", "very warm", "hot to touch", "overheat"),
    ),
    QueryTemplate(
        "heat-smoke", "smoke or burning smell",
        expected_aspect="overheating", expected_sentiment="negative",
        quote_phrases=("smoke", "burning", "smells like", "burnt", "burned"),
    ),
    QueryTemplate(
        "heat-melted", "device melted or warped",
        expected_aspect="overheating", expected_sentiment="negative",
        quote_phrases=("melted", "melt", "warped", "deformed"),
    ),
    # battery
    QueryTemplate(
        "bat-dies-fast", "battery dies quickly",
        expected_aspect="battery", expected_sentiment="negative",
        quote_phrases=("dies", "drains", "battery life is", "battery only lasts", "won't hold a charge", "wont hold a charge"),
    ),
    QueryTemplate(
        "bat-wont-hold", "battery won't hold a charge",
        expected_aspect="battery", expected_sentiment="negative",
        quote_phrases=("hold a charge", "hold charge", "stopped holding"),
    ),
    QueryTemplate(
        "bat-great-life", "great battery life",
        expected_aspect="battery", expected_sentiment="positive",
        quote_phrases=("great battery", "battery lasts", "long battery", "lasts forever"),
    ),
    # cross-cutting / brand-flavored queries
    QueryTemplate(
        "rel-anker-quality", "Anker product stopped working",
        expected_aspect="reliability", expected_sentiment="negative", expected_brand="Anker",
        quote_phrases=("stopped working", "broke", "broken", "defective", "doa"),
    ),
    QueryTemplate(
        "bt-anker-bluetooth", "Anker bluetooth disconnects",
        expected_aspect="bluetooth", expected_sentiment="negative", expected_brand="Anker",
        quote_phrases=("disconnect", "drops", "won't stay", "wont stay"),
    ),
    QueryTemplate(
        "chg-anker-cable", "Anker charging cable broke",
        expected_aspect="charging", expected_sentiment="negative", expected_brand="Anker",
        quote_phrases=("cable", "cord", "broke", "frayed"),
    ),
    QueryTemplate(
        "rel-unusable", "unusable after a few weeks",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("unusable", "useless", "junk", "garbage"),
    ),
    QueryTemplate(
        "rel-shocking", "got electric shock from device",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("shock", "shocked", "spark", "sparks"),
    ),
    QueryTemplate(
        "chg-port-loose", "charging port is loose",
        expected_aspect="charging", expected_sentiment="negative",
        quote_phrases=("loose", "wiggle", "wobble", "falls out"),
    ),
    QueryTemplate(
        "bat-overheats-during-charge", "battery overheats during charging",
        expected_aspect="overheating", expected_sentiment="negative",
        quote_phrases=("hot when charging", "hot while charging", "overheats while", "overheats when"),
    ),
    QueryTemplate(
        "price-refund-requested", "asked for a refund because product failed",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("refund", "return", "money back"),
    ),
    QueryTemplate(
        "snd-bass-weak", "bass is weak",
        expected_aspect="sound_quality", expected_sentiment="negative",
        quote_phrases=("bass", "weak bass", "no bass"),
    ),
    QueryTemplate(
        "snd-anc-bad", "noise cancelling does not work well",
        expected_aspect="sound_quality", expected_sentiment="negative",
        quote_phrases=("noise cancel", "noise-cancel", "anc"),
    ),
    QueryTemplate(
        "chg-fire-safety", "charger caught fire",
        expected_aspect="overheating", expected_sentiment="negative",
        quote_phrases=("fire", "caught fire", "flames", "burned up"),
    ),
    QueryTemplate(
        "del-wrong-item", "received the wrong item",
        expected_aspect="delivery", expected_sentiment="negative",
        quote_phrases=("wrong item", "wrong product", "different product"),
    ),
    QueryTemplate(
        "rel-build-quality", "poor build quality",
        expected_aspect="reliability", expected_sentiment="negative",
        quote_phrases=("build quality", "cheaply made", "flimsy", "poor quality"),
    ),
    QueryTemplate(
        "chg-power-bank", "power bank stopped charging",
        expected_aspect="charging", expected_sentiment="negative",
        quote_phrases=("power bank", "powerbank", "won't power", "wont power"),
    ),
    QueryTemplate(
        "bt-stereo", "only one earbud works",
        expected_aspect="bluetooth", expected_sentiment="negative",
        quote_phrases=("one side", "left side", "right side", "one ear", "left ear", "right ear"),
    ),
)


def _quote_clause(phrases: Iterable[str]):
    phrases = list(phrases)
    if not phrases:
        return None
    clauses = [AspectMention.evidence_quote.ilike(f"%{p}%") for p in phrases]
    return or_(*clauses) if clauses else None


def _collect_gold_ids(
    session: Session,
    template: QueryTemplate,
    *,
    aspect_version: str,
    provider: str,
    model_name: str,
    per_query: int,
) -> list[int]:
    stmt = (
        select(Review.id)
        .join(AspectMention, AspectMention.review_id == Review.id)
        .join(AspectOntology, AspectOntology.id == AspectMention.aspect_id)
        .join(
            ABSAReviewStatus,
            (ABSAReviewStatus.review_id == Review.id)
            & (ABSAReviewStatus.aspect_version == aspect_version)
            & (ABSAReviewStatus.provider == provider)
            & (ABSAReviewStatus.model_name == model_name),
        )
        .where(
            AspectMention.aspect_version == aspect_version,
            AspectMention.model_name == model_name,
        )
        .distinct()
    )
    if template.expected_aspect:
        stmt = stmt.where(AspectOntology.code == template.expected_aspect)
    if template.expected_sentiment:
        stmt = stmt.where(AspectMention.sentiment == template.expected_sentiment)
    if template.expected_brand:
        stmt = stmt.join(Sku, Sku.id == Review.sku_id).join(
            Brand, Brand.id == Sku.brand_id
        ).where(Brand.name == template.expected_brand)
    quote_clause = _quote_clause(template.quote_phrases)
    if quote_clause is not None:
        stmt = stmt.where(quote_clause)
    rows = list(session.execute(stmt.order_by(Review.id)).scalars())
    return rows[:per_query]


def build_goldens(
    *,
    aspect_version: str,
    provider: str,
    model_name: str,
    per_query: int,
    templates: Iterable[QueryTemplate] = QUERY_TEMPLATES,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with get_session() as session:
        for template in templates:
            gold_ids = _collect_gold_ids(
                session, template,
                aspect_version=aspect_version,
                provider=provider,
                model_name=model_name,
                per_query=per_query,
            )
            out.append(
                {
                    "query_id": template.query_id,
                    "query": template.query,
                    "expected_aspect": template.expected_aspect,
                    "expected_sentiment": template.expected_sentiment,
                    "expected_brand": template.expected_brand,
                    "gold_review_ids": gold_ids,
                    "notes": template.notes
                    or "weakly supervised from ABSA evidence_quote LIKE match",
                }
            )
    return out


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weakly supervised retrieval goldens.")
    parser.add_argument("--output", default="data/eval/retrieval_goldens.jsonl")
    parser.add_argument("--aspect-version", default="v2")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-opus-4.6")
    parser.add_argument("--per-query", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = build_goldens(
        aspect_version=args.aspect_version,
        provider=args.provider,
        model_name=args.model,
        per_query=args.per_query,
    )
    output = Path(args.output)
    write_jsonl(rows, output)
    n_with_gold = sum(1 for row in rows if row["gold_review_ids"])
    avg_gold = (
        round(sum(len(row["gold_review_ids"]) for row in rows) / len(rows), 2)
        if rows
        else 0.0
    )
    print(f"Wrote {len(rows)} retrieval golden rows to {output}")
    print(f"  templates_with_gold : {n_with_gold}")
    print(f"  avg_gold_per_query  : {avg_gold}")
    if n_with_gold < len(rows):
        empty = [row["query_id"] for row in rows if not row["gold_review_ids"]]
        print(f"  templates_with_no_gold ({len(empty)}): {', '.join(empty[:15])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
