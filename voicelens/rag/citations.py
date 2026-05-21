"""Citation objects — the bridge from retrieval hits to grounded claims.

A :class:`Citation` is one retrieved review, given a stable 1-based
``citation_id`` so the answer text can reference it as ``[1]``, ``[2]``
… Every field a reader (or an evaluator) needs to verify a claim is
carried on the citation, so nothing downstream has to re-query.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from voicelens.retrieval.search import SearchHit


@dataclass
class Citation:
    """One retrieved review, numbered for reference in an answer."""

    citation_id: int
    review_id: int | None
    source_id: str
    brand: str
    asin: str
    rating: int | None
    aspect_codes: list[str] = field(default_factory=list)
    evidence_quote: str = ""
    text_snippet: str = ""
    retrieval_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON output and the dashboard table."""
        return {
            "citation_id": self.citation_id,
            "review_id": self.review_id,
            "source_id": self.source_id,
            "brand": self.brand,
            "asin": self.asin,
            "rating": self.rating,
            "aspect_codes": list(self.aspect_codes),
            "evidence_quote": self.evidence_quote,
            "text_snippet": self.text_snippet,
            "retrieval_score": round(self.retrieval_score, 4),
        }


def _first_quote(quotes: Sequence[str]) -> str:
    """First non-empty evidence quote, or ``""`` when there is none."""
    for q in quotes:
        if q and q.strip():
            return q.strip()
    return ""


def build_citations(
    hits: Sequence[SearchHit], *, snippet_chars: int = 240
) -> list[Citation]:
    """Map retrieval hits to numbered citations, 1-based, in rank order.

    The ``citation_id`` is the hit's rank, so the answer's ``[n]``
    markers line up with the order the reviews were retrieved in.
    """
    citations: list[Citation] = []
    for rank, hit in enumerate(hits, start=1):
        source_id = str((hit.raw_payload or {}).get("source_id") or "")
        citations.append(
            Citation(
                citation_id=rank,
                review_id=hit.review_id,
                source_id=source_id,
                brand=hit.brand,
                asin=hit.asin,
                rating=hit.rating,
                aspect_codes=list(hit.aspect_codes),
                evidence_quote=_first_quote(hit.evidence_quotes),
                text_snippet=hit.text_snippet(snippet_chars),
                retrieval_score=float(hit.score),
            )
        )
    return citations
