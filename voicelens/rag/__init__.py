"""Citation-grounded answer generation over the hybrid retrieval layer.

Milestone 6A: a lightweight RAG answer generator. Given a question and
the reviews returned by :mod:`voicelens.retrieval`, it produces a
concise analyst-oriented answer whose every claim is bound to a
numbered citation ``[1]``, ``[2]`` … into the retrieved reviews.

It is deliberately *not* an agent: there is no planner, no tool
routing, no memory, and no autonomous action. Those are later
milestones (see the main README §8 / §11).
"""
from voicelens.rag.answer import AnswerResult, generate_answer
from voicelens.rag.citations import Citation, build_citations
from voicelens.rag.providers import (
    AnswerProvider,
    MockAnswerProvider,
    get_answer_provider,
)

__all__ = [
    "AnswerProvider",
    "AnswerResult",
    "Citation",
    "MockAnswerProvider",
    "build_citations",
    "generate_answer",
    "get_answer_provider",
]
