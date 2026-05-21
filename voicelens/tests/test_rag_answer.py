"""Tests for the M6A citation-grounded answer generator (voicelens.rag).

All paths use the offline mock provider or hand-built fakes — no real
LLM is ever called.
"""
from __future__ import annotations

import pytest

from voicelens.rag.answer import generate_answer
from voicelens.rag.citations import build_citations
from voicelens.rag.providers import (
    AnswerProvider,
    LLMAnswerProvider,
    MockAnswerProvider,
    get_answer_provider,
)
from voicelens.retrieval.search import SearchHit


def _hit(
    *,
    review_id: int,
    score: float = 1.0,
    brand: str = "Anker",
    asin: str = "A1",
    rating: int = 1,
    aspects: list[str] | None = None,
    quote: str = "stopped working after a week",
    text: str = "The unit stopped working after a week of light use.",
    source_id: str = "SRC1",
) -> SearchHit:
    return SearchHit(
        score=score,
        review_id=review_id,
        brand=brand,
        asin=asin,
        rating=rating,
        aspect_codes=aspects or ["reliability"],
        sentiments=["negative"],
        evidence_quotes=[quote],
        text_raw=text,
        raw_payload={"source_id": source_id, "review_id": review_id},
    )


_HITS = [
    _hit(review_id=11, score=0.9, source_id="S11"),
    _hit(review_id=12, score=0.7, asin="A2", quote="broke on the second use",
         text="It broke on the second use.", source_id="S12"),
]


# ---- citation builder ---------------------------------------------------


def test_build_citations_maps_hits_in_rank_order():
    citations = build_citations(_HITS)
    assert [c.citation_id for c in citations] == [1, 2]
    first = citations[0]
    assert first.review_id == 11
    assert first.source_id == "S11"
    assert first.brand == "Anker"
    assert first.asin == "A1"
    assert first.rating == 1
    assert first.aspect_codes == ["reliability"]
    assert first.evidence_quote == "stopped working after a week"
    assert first.text_snippet  # snippet derived from text_raw
    assert first.retrieval_score == pytest.approx(0.9)


def test_build_citations_falls_back_when_no_evidence_quote():
    hit = _hit(review_id=5, quote="")
    hit.evidence_quotes = ["", "  "]
    citation = build_citations([hit])[0]
    assert citation.evidence_quote == ""  # no non-empty quote
    assert citation.text_snippet         # snippet still present


# ---- mock provider ------------------------------------------------------


def test_mock_provider_returns_cited_answer():
    result = generate_answer(
        "What are the main reliability complaints?", _HITS, provider="mock"
    )
    assert result.insufficient_evidence is False
    assert len(result.citations) == 2
    assert "[1]" in result.answer
    assert result.guardrail_flags["unsupported_citations"] == []
    assert result.guardrail_flags["cited_ids"]
    assert result.provider == "mock"


def test_mock_provider_insufficient_for_irrelevant_question():
    result = generate_answer(
        "zzz blorptastic frobnicate nonsense", _HITS, provider="mock"
    )
    assert result.insufficient_evidence is True
    assert result.guardrail_flags.get("insufficient_evidence") is True


# ---- generate_answer guardrails -----------------------------------------


def test_generate_answer_no_hits_is_insufficient():
    result = generate_answer("anything at all", [], provider="mock")
    assert result.insufficient_evidence is True
    assert result.citations == []
    assert result.retrieved_review_ids == []
    assert result.guardrail_flags == {"no_evidence": True}


class _UnsupportedCitationProvider(AnswerProvider):
    """A fake provider that cites a review number that does not exist."""

    provider = "fake"
    model_name = "fake"

    def generate(self, question, citations):  # noqa: ARG002
        return "Customers report repeated failures [1] and also [99]."


def test_generate_answer_rejects_unsupported_citations():
    result = generate_answer(
        "reliability complaints", _HITS, provider=_UnsupportedCitationProvider()
    )
    # the bogus [99] marker is stripped from the answer text...
    assert "[99]" not in result.answer
    assert "[1]" in result.answer
    # ...and recorded in the guardrail flags.
    assert result.guardrail_flags["unsupported_citations"] == [99]
    assert result.guardrail_flags["cited_ids"] == [1]


def test_generate_answer_retrieved_ids_match_citations():
    result = generate_answer("reliability", _HITS, provider="mock")
    assert result.retrieved_review_ids == [11, 12]
    assert {c.review_id for c in result.citations} == {11, 12}


def test_generate_answer_records_filters():
    result = generate_answer(
        "reliability", _HITS, provider="mock", filters={"aspect": "reliability"}
    )
    assert result.filters == {"aspect": "reliability"}
    assert result.as_dict()["filters"] == {"aspect": "reliability"}


# ---- provider resolution ------------------------------------------------


def test_get_answer_provider_resolves_mock():
    assert isinstance(get_answer_provider("mock"), MockAnswerProvider)


def test_get_answer_provider_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown RAG provider"):
        get_answer_provider("not-a-provider")


def test_llm_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMAnswerProvider("anthropic")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMAnswerProvider("openai")
