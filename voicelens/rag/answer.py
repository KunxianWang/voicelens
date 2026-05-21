"""Citation-grounded answer generation — the Milestone 6A orchestrator.

:func:`generate_answer` takes a question and the reviews returned by the
existing hybrid retrieval layer, asks a provider for an answer, and then
*verifies* that answer: citation markers that do not map to a retrieved
review are stripped and flagged, and the insufficient-evidence path is
made explicit. It never performs retrieval itself, so it is trivially
testable with hand-built hits and the mock provider.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from voicelens.rag.citations import Citation, build_citations
from voicelens.rag.prompts import INSUFFICIENT_EVIDENCE_TOKEN
from voicelens.rag.providers import AnswerProvider, get_answer_provider
from voicelens.retrieval.search import SearchHit

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

_NO_EVIDENCE_ANSWER = (
    "I don't have enough retrieved evidence to answer this question. "
    "No reviews were returned for the query and filters."
)


@dataclass
class AnswerResult:
    """A grounded answer plus everything needed to audit it."""

    question: str
    answer: str
    citations: list[Citation]
    retrieved_review_ids: list[int]
    provider: str
    model: str
    insufficient_evidence: bool = False
    guardrail_flags: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view for the CLI and the dashboard."""
        return {
            "question": self.question,
            "answer": self.answer,
            "insufficient_evidence": self.insufficient_evidence,
            "provider": self.provider,
            "model": self.model,
            "retrieved_review_ids": list(self.retrieved_review_ids),
            "citations": [c.as_dict() for c in self.citations],
            "guardrail_flags": self.guardrail_flags,
            "filters": self.filters,
        }


def _cited_ids(text: str) -> list[int]:
    """Every distinct integer used as a ``[n]`` marker, in first-seen order."""
    seen: list[int] = []
    for raw in _CITATION_MARKER_RE.findall(text):
        n = int(raw)
        if n not in seen:
            seen.append(n)
    return seen


def _strip_unsupported_markers(text: str, valid_ids: set[int]) -> str:
    """Drop ``[n]`` markers whose ``n`` is not a real citation id."""

    def _repl(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid_ids else ""

    cleaned = _CITATION_MARKER_RE.sub(_repl, text)
    # collapse any double spaces a removed marker left behind
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _resolve_provider(
    provider: str | AnswerProvider | None, model: str | None
) -> AnswerProvider:
    if isinstance(provider, AnswerProvider):
        return provider
    return get_answer_provider(provider, model=model)


def generate_answer(
    question: str,
    hits: Sequence[SearchHit],
    *,
    provider: str | AnswerProvider | None = None,
    model: str | None = None,
    filters: dict[str, Any] | None = None,
) -> AnswerResult:
    """Generate a citation-grounded answer from retrieved reviews.

    ``provider`` may be a provider name (``mock`` / ``openai`` /
    ``anthropic``), an :class:`AnswerProvider` instance, or ``None`` to
    resolve from ``RAG_PROVIDER``. ``hits`` are the reviews from the
    hybrid retriever — this function does no retrieval of its own.
    """
    question = (question or "").strip()
    filters = dict(filters or {})
    citations = build_citations(hits)
    retrieved_review_ids = [
        c.review_id for c in citations if c.review_id is not None
    ]
    answer_provider = _resolve_provider(provider, model)

    # No evidence at all — short-circuit before calling the provider.
    if not citations:
        return AnswerResult(
            question=question,
            answer=_NO_EVIDENCE_ANSWER,
            citations=[],
            retrieved_review_ids=[],
            provider=answer_provider.provider,
            model=answer_provider.model_name,
            insufficient_evidence=True,
            guardrail_flags={"no_evidence": True},
            filters=filters,
        )

    raw = (answer_provider.generate(question, citations) or "").strip()

    # The provider declared the evidence insufficient.
    if raw.startswith(INSUFFICIENT_EVIDENCE_TOKEN):
        explanation = raw[len(INSUFFICIENT_EVIDENCE_TOKEN):].strip(" :-\n") or (
            "The retrieved reviews do not contain enough information to "
            "answer this question."
        )
        return AnswerResult(
            question=question,
            answer=explanation,
            citations=citations,
            retrieved_review_ids=retrieved_review_ids,
            provider=answer_provider.provider,
            model=answer_provider.model_name,
            insufficient_evidence=True,
            guardrail_flags={"insufficient_evidence": True},
            filters=filters,
        )

    valid_ids = {c.citation_id for c in citations}
    used_ids = _cited_ids(raw)
    unsupported = sorted(n for n in used_ids if n not in valid_ids)
    supported = sorted(n for n in used_ids if n in valid_ids)
    clean_answer = _strip_unsupported_markers(raw, valid_ids)

    guardrail_flags: dict[str, Any] = {
        "cited_ids": supported,
        "unsupported_citations": unsupported,
        "uncited_answer": len(supported) == 0,
    }
    return AnswerResult(
        question=question,
        answer=clean_answer,
        citations=citations,
        retrieved_review_ids=retrieved_review_ids,
        provider=answer_provider.provider,
        model=answer_provider.model_name,
        insufficient_evidence=False,
        guardrail_flags=guardrail_flags,
        filters=filters,
    )
