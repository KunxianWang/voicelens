"""Answer-generation providers: a deterministic mock plus real LLMs.

Mirrors the ABSA provider design (:mod:`voicelens.nlp.absa.providers`):
a ``mock`` backend for tests and offline demos, and an ``openai`` /
``anthropic`` backend that calls the real API. Provider and model are
resolved from ``RAG_PROVIDER`` / ``RAG_MODEL`` (or explicit arguments).
API keys are read from the environment and never logged.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from voicelens.rag.citations import Citation
from voicelens.rag.prompts import (
    ANSWER_SYSTEM_PROMPT,
    INSUFFICIENT_EVIDENCE_TOKEN,
    build_answer_user_prompt,
)

PROVIDER_MOCK = "mock"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
KNOWN_PROVIDERS: tuple[str, ...] = (PROVIDER_MOCK, PROVIDER_OPENAI, PROVIDER_ANTHROPIC)

# Tiny stopword set for the mock provider's relevance heuristic — just
# enough that "the/are/about" don't count as question-evidence overlap.
_STOPWORDS = frozenset(
    """a an and are about as at be been but by can could did do does for from
    had has have how in into is it its of on or that the their them then there
    these they this to was were what when where which who why will with would
    you your""".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+")


class AnswerProvider(ABC):
    """Abstract answer provider. Turns a question + citations into text."""

    provider: str = "abstract"
    model_name: str = "abstract"

    @abstractmethod
    def generate(self, question: str, citations: Sequence[Citation]) -> str:
        """Return raw answer text.

        The text may contain ``[n]`` citation markers, or begin with
        :data:`INSUFFICIENT_EVIDENCE_TOKEN` when the evidence does not
        support an answer. Implementations must not raise on empty
        ``citations`` — that case is handled upstream.
        """


def _content_tokens(text: str) -> set[str]:
    """Lowercase content words of length >= 3, stopwords removed."""
    return {
        w for w in _WORD_RE.findall(text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


class MockAnswerProvider(AnswerProvider):
    """Deterministic, offline answer provider for tests and demos.

    It does not paraphrase — it surfaces the retrieved evidence with
    correct citation markers, and it honestly returns the
    insufficient-evidence sentinel when no retrieved review shares any
    content word with the question. That makes both the cited-answer
    path and the insufficient-evidence path testable without an LLM.
    """

    provider = PROVIDER_MOCK
    model_name = PROVIDER_MOCK

    def generate(self, question: str, citations: Sequence[Citation]) -> str:
        q_tokens = _content_tokens(question)
        relevant: list[Citation] = []
        for c in citations:
            # aspect codes count as evidence text too, so a question that
            # names an aspect ("reliability") matches reviews tagged with
            # it even when the review wording never uses that word.
            evidence = " ".join(
                [c.evidence_quote, c.text_snippet, *c.aspect_codes]
            )
            if q_tokens & _content_tokens(evidence):
                relevant.append(c)

        if not relevant:
            return (
                f"{INSUFFICIENT_EVIDENCE_TOKEN} None of the retrieved reviews "
                "mention anything relevant to this question."
            )

        markers = "".join(f"[{c.citation_id}]" for c in relevant)
        lead = relevant[0]
        lead_quote = lead.evidence_quote or lead.text_snippet
        return (
            f"Based on {len(relevant)} retrieved review(s), customers raise "
            f"points relevant to this question {markers}. For example, one "
            f'review states: "{lead_quote}" [{lead.citation_id}].'
        )


class LLMAnswerProvider(AnswerProvider):
    """Real OpenAI / Anthropic answer provider.

    Uses the same HTTP shape and ``*_BASE_URL`` overrides as the ABSA
    LLM provider. The API key is required at construction time so a
    missing key fails loudly before any retrieval cost is spent.
    """

    def __init__(self, backend: str, *, model: str | None = None) -> None:
        backend = backend.lower()
        if backend not in (PROVIDER_OPENAI, PROVIDER_ANTHROPIC):
            raise ValueError(
                f"LLMAnswerProvider backend must be {PROVIDER_OPENAI!r} or "
                f"{PROVIDER_ANTHROPIC!r}, got {backend!r}"
            )
        env_key = "OPENAI_API_KEY" if backend == PROVIDER_OPENAI else "ANTHROPIC_API_KEY"
        if not os.getenv(env_key):
            raise RuntimeError(
                f"RAG_PROVIDER={backend} requires {env_key} to be set in the "
                f"environment (.env or shell). Use --provider mock for local "
                f"runs and tests."
            )
        env_model = os.getenv("RAG_MODEL")
        default_model = (
            "gpt-4o-mini" if backend == PROVIDER_OPENAI else "claude-3-5-haiku-latest"
        )
        self.backend = backend
        self.provider = backend
        self.model_name = model or env_model or default_model
        self._api_key = os.environ[env_key]
        self._timeout = float(os.getenv("RAG_HTTP_TIMEOUT", "60"))

    def generate(self, question: str, citations: Sequence[Citation]) -> str:
        user_prompt = build_answer_user_prompt(question, list(citations))
        if self.backend == PROVIDER_OPENAI:
            return self._call_openai(user_prompt)
        return self._call_anthropic(user_prompt)

    def _call_openai(self, user_prompt: str) -> str:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "input": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = _post_json(
            f"{base_url}/v1/responses",
            payload,
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        text = data.get("output_text") or _openai_output_text(data)
        return (text or "").strip()

    def _call_anthropic(self, user_prompt: str) -> str:
        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        payload = {
            "model": self.model_name,
            "max_tokens": 700,
            "temperature": 0,
            "system": ANSWER_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = _post_json(
            f"{base_url}/v1/messages",
            payload,
            {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return text.strip()


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], *, timeout: float
) -> dict[str, Any]:
    """POST JSON and return the parsed response (raises on HTTP error)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RAG LLM API request failed (HTTP {exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"RAG LLM API request failed: {exc}") from exc
    return json.loads(raw)


def _openai_output_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "".join(chunks)


def get_answer_provider(
    name: str | None = None, *, model: str | None = None
) -> AnswerProvider:
    """Resolve a provider name to an instance.

    Order: explicit ``name`` > ``RAG_PROVIDER`` env var > ``mock``.
    Unknown names raise ``ValueError`` so a typo fails loudly.
    """
    resolved = (name or os.getenv("RAG_PROVIDER") or PROVIDER_MOCK).strip().lower()
    if resolved == PROVIDER_MOCK:
        return MockAnswerProvider()
    if resolved in (PROVIDER_OPENAI, PROVIDER_ANTHROPIC):
        return LLMAnswerProvider(resolved, model=model)
    raise ValueError(
        f"Unknown RAG provider: {resolved!r}. Expected one of {KNOWN_PROVIDERS}."
    )
