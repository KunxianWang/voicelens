"""ABSA providers."""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from voicelens.nlp.absa.prompts import ABSA_SYSTEM_PROMPT, build_user_prompt
from voicelens.nlp.absa.schema import ABSAOutput, AspectMentionOut

PROVIDER_MOCK = "mock"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"

KNOWN_PROVIDERS: tuple[str, ...] = (PROVIDER_MOCK, PROVIDER_OPENAI, PROVIDER_ANTHROPIC)


@dataclass
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    calls: int = 0
    retries: int = 0
    transient_retry_attempts: int = 0
    recovered_after_retry: int = 0
    estimated_cost_usd: float | None = None
    cost_note: str | None = None

    def add_call(
        self,
        *,
        latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self.calls += 1
        self.latencies_ms.append(latency_ms)
        if input_tokens is not None:
            self.input_tokens += input_tokens
        if output_tokens is not None:
            self.output_tokens += output_tokens
        self._refresh_estimated_cost()

    def _refresh_estimated_cost(self) -> None:
        input_price = _float_env("ABSA_INPUT_COST_PER_1M_TOKENS")
        output_price = _float_env("ABSA_OUTPUT_COST_PER_1M_TOKENS")
        if input_price is None or output_price is None:
            return
        self.estimated_cost_usd = round(
            (self.input_tokens / 1_000_000) * input_price
            + (self.output_tokens / 1_000_000) * output_price,
            6,
        )
        self.cost_note = (
            "Estimated from ABSA_INPUT_COST_PER_1M_TOKENS and "
            "ABSA_OUTPUT_COST_PER_1M_TOKENS."
        )

    def as_dict(self, *, provider: str, model_name: str) -> dict[str, Any]:
        latencies = sorted(self.latencies_ms)
        p95 = None
        avg = None
        if latencies:
            avg = round(sum(latencies) / len(latencies), 2)
            idx = min(len(latencies) - 1, int((len(latencies) - 1) * 0.95))
            p95 = round(latencies[idx], 2)
        return {
            "provider": provider,
            "model_name": model_name,
            "calls": self.calls,
            "retries": self.retries,
            "retry_attempts_total": self.transient_retry_attempts,
            "recovered_after_retry": self.recovered_after_retry,
            "total_input_tokens": self.input_tokens or None,
            "total_output_tokens": self.output_tokens or None,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_note": self.cost_note
            or "No built-in price table; token counts are reported when the API returns them.",
            "avg_latency_ms": avg,
            "p95_latency_ms": p95,
        }


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

    def __init__(self) -> None:
        self.usage = ProviderUsage()

    @property
    def name(self) -> str:
        return self.model_name

    def usage_summary(self) -> dict[str, Any]:
        return self.usage.as_dict(provider=self.provider, model_name=self.model_name)

    def configure_retries(self, *, max_retries: int, base_seconds: float) -> None:
        return None

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

    def __init__(self) -> None:
        super().__init__()

    def extract(self, review_text: str) -> ABSAOutput:
        start = time.perf_counter()
        if not review_text or not review_text.strip():
            self.usage.add_call(latency_ms=(time.perf_counter() - start) * 1000)
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
        out = ABSAOutput(aspects=mentions)
        self.usage.add_call(latency_ms=(time.perf_counter() - start) * 1000)
        return out


class LLMABSAProvider(ABSAProvider):
    """Real OpenAI / Anthropic ABSA provider with one retry on bad JSON."""

    provider = "llm"

    def __init__(self, backend: str, *, model: str | None = None) -> None:
        super().__init__()
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
        env_model = os.getenv("ABSA_MODEL")
        default_model = "gpt-4o-mini" if backend == PROVIDER_OPENAI else "claude-3-5-haiku-latest"
        self.backend = backend
        self.model = model or env_model or default_model
        self.provider = backend
        self.model_name = self.model
        self.api_key = os.environ[env_key]
        self.llm_max_retries = 2
        self.llm_retry_base_seconds = 1.0

    def configure_retries(self, *, max_retries: int, base_seconds: float) -> None:
        self.llm_max_retries = max(0, int(max_retries))
        self.llm_retry_base_seconds = max(0.0, float(base_seconds))

    def extract(self, review_text: str) -> ABSAOutput:
        if not review_text or not review_text.strip():
            return ABSAOutput(aspects=[])

        last_error: Exception | None = None
        for attempt in range(2):
            if attempt:
                self.usage.retries += 1
            try:
                raw_text, usage, latency_ms = self._call_model_with_transient_retries(review_text)
                self.usage.add_call(
                    latency_ms=latency_ms,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )
                return self._parse_json_output(raw_text)
            except (json.JSONDecodeError, PydanticValidationError, ValueError) as exc:
                last_error = exc
                continue
        raise RuntimeError(f"{self.backend} ABSA output failed JSON/schema validation after retry: {last_error}")

    def _call_model_with_transient_retries(self, review_text: str) -> tuple[str, dict[str, int], float]:
        last_error: Exception | None = None
        for attempt in range(self.llm_max_retries + 1):
            try:
                result = self._call_model(review_text)
                if attempt > 0:
                    self.usage.recovered_after_retry += 1
                return result
            except TransientLLMError as exc:
                last_error = exc
                if attempt >= self.llm_max_retries:
                    break
                self.usage.transient_retry_attempts += 1
                _sleep_with_backoff(self.llm_retry_base_seconds, attempt)
        raise RuntimeError(f"LLM transient request failed after {self.llm_max_retries} retries: {last_error}")

    def _call_model(self, review_text: str) -> tuple[str, dict[str, int], float]:
        if self.backend == PROVIDER_OPENAI:
            return self._call_openai(review_text)
        return self._call_anthropic(review_text)

    def _call_openai(self, review_text: str) -> tuple[str, dict[str, int], float]:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": ABSA_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(review_text)},
            ],
            "temperature": 0,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "absa_output",
                    "strict": True,
                    "schema": _absa_json_schema(),
                }
            },
        }
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        data, latency_ms = _post_json(
            f"{base_url}/v1/responses",
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        text = data.get("output_text")
        if not text:
            text = _extract_openai_output_text(data)
        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": _coerce_int(
                usage_raw.get("input_tokens") or usage_raw.get("prompt_tokens")
            ),
            "output_tokens": _coerce_int(
                usage_raw.get("output_tokens") or usage_raw.get("completion_tokens")
            ),
        }
        return text, usage, latency_ms

    def _call_anthropic(self, review_text: str) -> tuple[str, dict[str, int], float]:
        payload = {
            "model": self.model,
            "max_tokens": 900,
            "temperature": 0,
            "system": ABSA_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": build_user_prompt(review_text)
                    + '\n\nReturn JSON only. The first character must be "{".',
                }
            ],
        }
        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        data, latency_ms = _post_json(
            f"{base_url}/v1/messages",
            payload,
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        parts = data.get("content") or []
        text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": _coerce_int(usage_raw.get("input_tokens")),
            "output_tokens": _coerce_int(usage_raw.get("output_tokens")),
        }
        return text, usage, latency_ms

    def _parse_json_output(self, text: str) -> ABSAOutput:
        payload = _extract_json_object(text)
        return ABSAOutput.model_validate(payload)


def _absa_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["aspects"],
        "properties": {
            "aspects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["aspect_code", "sentiment", "severity", "evidence_quote"],
                    "properties": {
                        "aspect_code": {"type": "string", "enum": list(_ONTOLOGY_CODES())},
                        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                        "severity": {
                            "anyOf": [
                                {"type": "string", "enum": ["low", "medium", "high"]},
                                {"type": "null"},
                            ]
                        },
                        "evidence_quote": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def _ONTOLOGY_CODES() -> tuple[str, ...]:
    from voicelens.nlp.absa.schema import ONTOLOGY_CODES_V1

    return ONTOLOGY_CODES_V1


class TransientLLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {408, 409, 425, 429, 500, 502, 503, 504}:
            raise TransientLLMError(f"HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"LLM API request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise TransientLLMError(str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000
    return json.loads(raw), latency_ms


def _sleep_with_backoff(base_seconds: float, attempt: int) -> None:
    if base_seconds <= 0:
        return
    delay = base_seconds * (2**attempt)
    jitter = random.uniform(0, base_seconds * 0.25)
    time.sleep(delay + jitter)


def _extract_openai_output_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "".join(chunks)


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("ABSA provider returned JSON that is not an object")
    return payload


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_env(key: str) -> float | None:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


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
