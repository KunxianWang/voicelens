"""ABSA providers.

Two concrete providers ship today:

- :class:`MockABSAProvider` — deterministic, rule-based keyword matcher.
  Used in tests, the local smoke flow, and CI. Outputs are guaranteed to
  pass schema + verbatim validation when the keyword is present in the
  review text. No network, no model, no cost.

- :class:`LLMABSAProvider` — a thin stub that wires real OpenAI / Anthropic
  clients to the prompt in ``prompts.py``. It refuses to construct unless
  the corresponding API key is present, so a misconfigured environment
  fails fast instead of silently hitting a stub. Tests must use the mock.

A factory :func:`get_provider` reads ``ABSA_PROVIDER`` from the environment
(or an explicit argument) and returns the right concrete provider. The
flow uses the factory; downstream consumers can pass their own instance.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from voicelens.nlp.absa.schema import ABSAOutput, AspectMentionOut

PROVIDER_MOCK = "mock"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"

KNOWN_PROVIDERS: tuple[str, ...] = (PROVIDER_MOCK, PROVIDER_OPENAI, PROVIDER_ANTHROPIC)


class ABSAProvider(ABC):
    """Abstract provider. Subclasses must extract ABSA from one review.

    Subclasses set two identifiers used for idempotency and observability:

    - ``provider``: the family ("mock", "openai", "anthropic") that selects
      the runtime code path. Stable per implementation.
    - ``model_name``: the specific model identifier. For mock the two are
      equal; for LLMs ``model_name`` is the concrete model id passed at
      construction time (e.g. ``"gpt-4o-mini"``).

    ``name`` is kept as a legacy alias for ``model_name``.
    """

    provider: str = "abstract"
    model_name: str = "abstract"

    @property
    def name(self) -> str:
        return self.model_name

    @abstractmethod
    def extract(self, review_text: str) -> ABSAOutput:
        """Return aspect mentions for one review. Must not raise on empty
        input — return ``ABSAOutput(aspects=[])`` instead.
        """


@dataclass(frozen=True)
class _KeywordRule:
    aspect_code: str
    keywords: tuple[str, ...]
    negative_markers: tuple[str, ...] = ()
    positive_markers: tuple[str, ...] = ()


_NEGATIVE_MARKERS = (
    "broke", "broken", "stopped working", "doesn't work", "does not work",
    "stopped", "dead", "useless", "garbage", "terrible", "awful", "bad",
    "disappointed", "refund", "return", "returned", "waste", "scam",
    "horrible", "poor", "fail", "failed", "failure", "junk", "issue",
    "issues", "problem", "problems", "won't", "wouldn't", "couldn't",
    "no longer", "stop", "stops", "slow", "weak", "cheap quality",
    "not worth", "overpriced", "not happy", "unhappy", "complaint",
    "complaints", "annoying", "lousy", "subpar",
)
_POSITIVE_MARKERS = (
    "great", "excellent", "amazing", "love", "perfect", "fantastic",
    "best", "awesome", "good", "solid", "happy", "satisfied",
    "recommend", "fast", "worth", "value", "decent",
)
_HIGH_SEVERITY_MARKERS = (
    "fire", "smoke", "burned", "burning", "burnt", "shock", "shocked",
    "hospital", "danger", "dangerous", "exploded", "explode", "explosion",
    "melted", "refused refund",
)
_MEDIUM_SEVERITY_MARKERS = (
    "unusable", "returned", "broken", "never worked", "dead on arrival",
    "doesn't work", "does not work", "broke after", "stopped working",
)

_RULES: tuple[_KeywordRule, ...] = (
    _KeywordRule("battery", ("battery", "batteries", "charge cycle", "holds a charge", "holds charge")),
    _KeywordRule("charging", ("charging", "charger", "fast charge", "usb-c", "usb c", "cable", "ports", "port")),
    _KeywordRule("overheating", ("overheat", "overheats", "overheating", "too hot", "gets hot", "hot to touch")),
    _KeywordRule("sound_quality", ("sound quality", "audio", "bass", "treble", "anc", "noise cancel", "noise canceling", "noise-cancelling", "noise cancelling", "mic", "microphone", "volume")),
    _KeywordRule("bluetooth", ("bluetooth", "pair", "pairing", "paired", "connect", "connecting", "connection", "latency", "range", "drops out", "drop out", "drops")),
    _KeywordRule("delivery", ("delivery", "shipping", "shipped", "package", "packaging", "arrived", "missing item", "missing items", "wrong item")),
    _KeywordRule("price", ("price", "value", "expensive", "cheap", "overpriced", "worth the money", "for the money")),
)


def _find_sentence_containing(text: str, needle: str) -> str | None:
    """Return the sentence (period/exclam/question delimited) that contains needle.

    Case-insensitive search; the returned slice is a verbatim substring of
    ``text``. Falls back to a 120-char window around the match if no
    sentence boundary is found, still preserving verbatim semantics.
    """
    pattern = re.escape(needle)
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    start, end = match.span()
    sentence_start = 0
    for i in range(start - 1, -1, -1):
        if text[i] in ".!?\n":
            sentence_start = i + 1
            break
    sentence_end = len(text)
    for i in range(end, len(text)):
        if text[i] in ".!?\n":
            sentence_end = i + 1
            break
    snippet = text[sentence_start:sentence_end].strip()
    if not snippet:
        snippet = text[max(0, start - 40): min(len(text), end + 40)].strip()
    if snippet and snippet in text:
        return snippet
    return text[start:end]


def _classify_sentiment(quote: str) -> tuple[str, str | None]:
    """Return (sentiment, severity)."""
    lowered = quote.lower()
    has_neg = any(m in lowered for m in _NEGATIVE_MARKERS)
    has_pos = any(m in lowered for m in _POSITIVE_MARKERS)
    if has_neg and not has_pos:
        severity = "low"
        if any(m in lowered for m in _HIGH_SEVERITY_MARKERS):
            severity = "high"
        elif any(m in lowered for m in _MEDIUM_SEVERITY_MARKERS):
            severity = "medium"
        return "negative", severity
    if has_pos and not has_neg:
        return "positive", None
    return "neutral", None


class MockABSAProvider(ABSAProvider):
    """Deterministic keyword-based provider for tests and local smoke runs.

    Per-review behaviour:
    - scan the review text with the keyword rules in :data:`_RULES`
    - for each matched aspect, lift the surrounding sentence as the
      ``evidence_quote`` (always a verbatim substring of the input)
    - classify sentiment + severity by scanning the same sentence for
      negative / positive markers
    - emit at most one mention per aspect_code per review

    The provider never raises; reviews with no keyword hit return
    ``ABSAOutput(aspects=[])``.
    """

    provider = PROVIDER_MOCK
    model_name = PROVIDER_MOCK

    def extract(self, review_text: str) -> ABSAOutput:
        if not review_text or not review_text.strip():
            return ABSAOutput(aspects=[])

        mentions: list[AspectMentionOut] = []
        seen: set[str] = set()
        lowered = review_text.lower()
        for rule in _RULES:
            if rule.aspect_code in seen:
                continue
            for keyword in rule.keywords:
                if keyword in lowered:
                    quote = _find_sentence_containing(review_text, keyword)
                    if quote is None or quote not in review_text:
                        continue
                    sentiment, severity = _classify_sentiment(quote)
                    mentions.append(
                        AspectMentionOut(
                            aspect_code=rule.aspect_code,
                            sentiment=sentiment,  # type: ignore[arg-type]
                            severity=severity,  # type: ignore[arg-type]
                            evidence_quote=quote,
                        )
                    )
                    seen.add(rule.aspect_code)
                    break
        return ABSAOutput(aspects=mentions)


class LLMABSAProvider(ABSAProvider):
    """Real-LLM provider stub.

    Construction requires the matching API key. ``extract`` is intentionally
    not implemented in this milestone — the responsibility of M2A is to wire
    up the contract, schema, validators, flow, and idempotency. Real-LLM
    calls land in M2B alongside the 200-row labeled holdout.
    """

    provider = "llm"

    def __init__(self, backend: str, *, model: str | None = None) -> None:
        backend = backend.lower()
        if backend not in (PROVIDER_OPENAI, PROVIDER_ANTHROPIC):
            raise ValueError(
                f"LLMABSAProvider backend must be one of "
                f"{{{PROVIDER_OPENAI!r}, {PROVIDER_ANTHROPIC!r}}}, got {backend!r}"
            )
        env_key = "OPENAI_API_KEY" if backend == PROVIDER_OPENAI else "ANTHROPIC_API_KEY"
        if not os.getenv(env_key):
            raise RuntimeError(
                f"ABSA_PROVIDER={backend} requires {env_key} to be set in the "
                f"environment. Configure it in .env or your shell, or use "
                f"--provider mock for local runs."
            )
        self.backend = backend
        self.model = model
        self.provider = backend
        self.model_name = model or f"llm:{backend}"

    def extract(self, review_text: str) -> ABSAOutput:  # pragma: no cover - M2B
        raise NotImplementedError(
            "LLMABSAProvider.extract is not implemented in Milestone 2A. "
            "Use --provider mock for the smoke run; real-LLM extraction is "
            "scoped to Milestone 2B."
        )


def get_provider(name: str | None = None, *, model: str | None = None) -> ABSAProvider:
    """Resolve a provider name to a concrete instance.

    Resolution order: explicit ``name`` argument > ``ABSA_PROVIDER`` env var
    > default ``mock``. Raises ``ValueError`` for unknown names so a typo
    in CLI / env fails loudly.
    """
    resolved = (name or os.getenv("ABSA_PROVIDER") or PROVIDER_MOCK).strip().lower()
    if resolved == PROVIDER_MOCK:
        return MockABSAProvider()
    if resolved in (PROVIDER_OPENAI, PROVIDER_ANTHROPIC):
        return LLMABSAProvider(resolved, model=model)
    raise ValueError(
        f"Unknown ABSA provider: {resolved!r}. Expected one of {KNOWN_PROVIDERS}."
    )
