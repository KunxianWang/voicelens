"""Deterministic keyword router + filter extraction for the M6B agent.

No LLM planner: routing is a transparent set of keyword rules so the
decision is reproducible and testable. ``classify_route`` picks one of
four routes; ``extract_filters`` pulls aspect / sentiment / brand hints
out of the question for the retrieval route.
"""
from __future__ import annotations

import re
from typing import Any

from voicelens.config import BRAND_ALLOWLIST

ROUTE_RETRIEVAL = "retrieval_answer"
ROUTE_INCIDENT = "incident_summary"
ROUTE_ANALYTICS = "analytics_summary"
ROUTE_INSUFFICIENT = "insufficient_scope"

ALL_ROUTES: tuple[str, ...] = (
    ROUTE_RETRIEVAL,
    ROUTE_INCIDENT,
    ROUTE_ANALYTICS,
    ROUTE_INSUFFICIENT,
)

# Keyword triggers, checked in priority order: incident first (most
# specific vocabulary), then analytics, then retrieval.
_INCIDENT_KEYWORDS: tuple[str, ...] = (
    "incident", "incidents", "spike", "spiked", "spiking", "anomaly",
    "anomalies", "emerging", "emerge", "surge", "surging", "getting worse",
    "trending up", "ramping",
)
_ANALYTICS_KEYWORDS: tuple[str, ...] = (
    "how many", "how much", "count", "counts", "distribution", "distributions",
    "top aspect", "top aspects", "most common", "most negative",
    "most frequent", "breakdown", "percentage", "proportion", "share of",
    "statistics", "stats", "overall sentiment", "severity breakdown",
    "absa status",
)
_RETRIEVAL_KEYWORDS: tuple[str, ...] = (
    "complaint", "complaints", "complaining", "customers say", "customer say",
    "customers saying", "users say", "users saying", "people say", "reviews",
    "review", "quote", "quotes", "example", "examples", "saying about",
    "say about", "feedback", "evidence", "what do customers", "what are users",
)

# The 7+1 ABSA aspect codes; aliases map natural phrasing to the code.
_ASPECT_ALIASES: dict[str, str] = {
    "reliability": "reliability",
    "reliable": "reliability",
    "broke": "reliability",
    "broken": "reliability",
    "stopped working": "reliability",
    "bluetooth": "bluetooth",
    "pairing": "bluetooth",
    "disconnect": "bluetooth",
    "charging": "charging",
    "charge": "charging",
    "charger": "charging",
    "price": "price",
    "value": "price",
    "expensive": "price",
    "cost": "price",
    "delivery": "delivery",
    "shipping": "delivery",
    "packaging": "delivery",
    "sound quality": "sound_quality",
    "sound_quality": "sound_quality",
    "audio": "sound_quality",
    "overheating": "overheating",
    "overheat": "overheating",
    "too hot": "overheating",
    "battery": "battery",
}

_POSITIVE_HINTS: tuple[str, ...] = (
    "positive", "praise", "love", "like about", "good about", "happy", "best",
)
_NEGATIVE_HINTS: tuple[str, ...] = (
    "negative", "complaint", "complaints", "complaining", "problem", "problems",
    "issue", "issues", "fail", "failure", "failing", "bad", "worst", "broken",
    "disappoint",
)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def classify_route(question: str) -> str:
    """Classify a question into one of :data:`ALL_ROUTES`.

    Priority: incident > analytics > retrieval. A question that names an
    aspect but matches no other route still goes to retrieval (the user
    wants to see reviews). Everything else is insufficient scope.
    """
    q = (question or "").lower().strip()
    if not q:
        return ROUTE_INSUFFICIENT
    if _contains_any(q, _INCIDENT_KEYWORDS):
        return ROUTE_INCIDENT
    if _contains_any(q, _ANALYTICS_KEYWORDS):
        return ROUTE_ANALYTICS
    if _contains_any(q, _RETRIEVAL_KEYWORDS):
        return ROUTE_RETRIEVAL
    # No explicit route keyword — but if the question clearly names an
    # aspect, treat it as a request to see what customers said about it.
    if _detect_aspect(q) is not None:
        return ROUTE_RETRIEVAL
    return ROUTE_INSUFFICIENT


def _detect_aspect(text: str) -> str | None:
    """Return the first ABSA aspect code mentioned in ``text``."""
    for alias, code in _ASPECT_ALIASES.items():
        if alias in text:
            return code
    return None


def _detect_sentiment(text: str) -> str | None:
    """Infer a sentiment filter, or ``None`` when the question is neutral."""
    has_neg = _contains_any(text, _NEGATIVE_HINTS)
    has_pos = _contains_any(text, _POSITIVE_HINTS)
    if has_neg and not has_pos:
        return "negative"
    if has_pos and not has_neg:
        return "positive"
    return None


def _detect_brand(text: str) -> str | None:
    """Return a known brand named in the question, if any."""
    for brand in BRAND_ALLOWLIST:
        if re.search(rf"\b{re.escape(brand.lower())}\b", text):
            return brand
    return None


def extract_filters(question: str) -> dict[str, Any]:
    """Pull aspect / sentiment / brand hints out of a question.

    Only non-``None`` hints are included, so an empty dict means "no
    filters" — the retrieval tool then searches unfiltered.
    """
    q = (question or "").lower()
    filters: dict[str, Any] = {}
    aspect = _detect_aspect(q)
    if aspect is not None:
        filters["aspect"] = aspect
    sentiment = _detect_sentiment(q)
    if sentiment is not None:
        filters["sentiment"] = sentiment
    brand = _detect_brand(q)
    if brand is not None:
        filters["brand"] = brand
    return filters
