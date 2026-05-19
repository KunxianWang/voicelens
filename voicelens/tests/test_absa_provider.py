from __future__ import annotations

import pytest

from voicelens.nlp.absa import (
    ONTOLOGY_CODES_V1,
    MockABSAProvider,
    get_provider,
)
from voicelens.nlp.absa.providers import LLMABSAProvider, TransientLLMError
from voicelens.nlp.absa.schema import ABSAOutput


def test_mock_provider_detects_battery_negative_with_severity():
    text = "The battery is broken after one week and now it won't hold a charge at all."
    provider = MockABSAProvider()

    output = provider.extract(text)
    aspects = {m.aspect_code: m for m in output.aspects}

    assert "battery" in aspects
    mention = aspects["battery"]
    assert mention.sentiment == "negative"
    assert mention.severity in {"low", "medium", "high"}
    assert mention.evidence_quote in text


def test_mock_provider_returns_empty_for_empty_text():
    provider = MockABSAProvider()
    assert provider.extract("").aspects == []
    assert provider.extract("   ").aspects == []


def test_mock_provider_detects_multiple_aspects_unique_per_review():
    text = (
        "Charging is fast and the sound quality is amazing. "
        "But bluetooth pairing keeps dropping and the price is too expensive."
    )
    provider = MockABSAProvider()

    output = provider.extract(text)
    codes = [m.aspect_code for m in output.aspects]

    assert len(codes) == len(set(codes)), "aspect_code must be unique per review"
    assert set(codes).issubset(set(ONTOLOGY_CODES_V1))
    assert "charging" in codes
    assert "sound_quality" in codes
    assert "bluetooth" in codes
    assert "price" in codes


def test_mock_provider_evidence_is_verbatim_substring():
    text = "The bluetooth pairing is unreliable and drops out every few minutes."
    provider = MockABSAProvider()

    output = provider.extract(text)
    for mention in output.aspects:
        assert mention.evidence_quote in text


def test_mock_provider_neutral_when_no_polarity_markers():
    text = "I use the battery on my desk in the office."
    provider = MockABSAProvider()

    output = provider.extract(text)
    battery = next(m for m in output.aspects if m.aspect_code == "battery")
    assert battery.sentiment in {"neutral", "positive", "negative"}


def test_get_provider_returns_mock_by_default(monkeypatch):
    monkeypatch.delenv("ABSA_PROVIDER", raising=False)
    provider = get_provider()
    assert isinstance(provider, MockABSAProvider)


def test_get_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown ABSA provider"):
        get_provider("totally-not-a-provider")


def test_llm_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMABSAProvider("openai")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMABSAProvider("anthropic")


def test_llm_provider_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="backend must be one of"):
        LLMABSAProvider("not-a-backend")


def test_llm_provider_uses_model_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ABSA_MODEL", "gpt-test")
    provider = LLMABSAProvider("openai")
    assert isinstance(provider, LLMABSAProvider)
    assert provider.name == "gpt-test"


def test_get_provider_returns_llm_when_env_set(monkeypatch):
    monkeypatch.setenv("ABSA_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = get_provider()
    assert isinstance(provider, LLMABSAProvider)
    assert provider.backend == "anthropic"


def test_provider_factory_selects_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = get_provider("openai", model="gpt-test")
    assert isinstance(provider, LLMABSAProvider)
    assert provider.provider == "openai"
    assert provider.model_name == "gpt-test"


def test_llm_provider_recovers_after_one_malformed_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FlakyProvider(LLMABSAProvider):
        def __init__(self) -> None:
            super().__init__("openai", model="gpt-test")
            self.calls = 0

        def _call_model(self, review_text: str):
            self.calls += 1
            if self.calls == 1:
                return "not json", {}, 10.0
            return (
                {
                    "aspects": [
                        {
                            "aspect_code": "battery",
                            "sentiment": "negative",
                            "severity": "low",
                            "evidence_quote": "battery is bad",
                        }
                    ]
                },
                {},
                12.0,
            )

        def _parse_json_output(self, text):
            if isinstance(text, dict):
                return ABSAOutput.model_validate(text)
            return super()._parse_json_output(text)

    out = FlakyProvider().extract("battery is bad")
    assert out.aspects[0].aspect_code == "battery"


def test_malformed_llm_output_fails_after_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class BadProvider(LLMABSAProvider):
        def __init__(self) -> None:
            super().__init__("openai", model="gpt-test")

        def _call_model(self, review_text: str):
            return "not json", {}, 1.0

    with pytest.raises(RuntimeError, match="failed JSON/schema validation"):
        BadProvider().extract("battery is bad")


def test_transient_failure_succeeds_after_one_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class TransientThenOk(LLMABSAProvider):
        def __init__(self) -> None:
            super().__init__("openai", model="gpt-test")
            self.calls = 0
            self.configure_retries(max_retries=2, base_seconds=0)

        def _call_model(self, review_text: str):
            self.calls += 1
            if self.calls == 1:
                raise TransientLLMError("connection reset")
            return '{"aspects":[]}', {}, 1.0

    provider = TransientThenOk()
    out = provider.extract("no aspects here")

    assert out.aspects == []
    assert provider.calls == 2
    assert provider.usage.transient_retry_attempts == 1
    assert provider.usage.recovered_after_retry == 1


def test_persistent_transient_failure_raises_after_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class AlwaysTransient(LLMABSAProvider):
        def __init__(self) -> None:
            super().__init__("openai", model="gpt-test")
            self.calls = 0
            self.configure_retries(max_retries=2, base_seconds=0)

        def _call_model(self, review_text: str):
            self.calls += 1
            raise TransientLLMError("connection reset")

    provider = AlwaysTransient()
    with pytest.raises(RuntimeError, match="transient request failed"):
        provider.extract("battery")
    assert provider.calls == 3
    assert provider.usage.transient_retry_attempts == 2


def test_mock_provider_output_is_pydantic_abas_output():
    provider = MockABSAProvider()
    out = provider.extract("battery charging bluetooth")
    assert isinstance(out, ABSAOutput)
