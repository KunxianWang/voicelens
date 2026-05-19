from __future__ import annotations

import pytest

from voicelens.nlp.absa import (
    ONTOLOGY_CODES_V1,
    MockABSAProvider,
    get_provider,
)
from voicelens.nlp.absa.providers import LLMABSAProvider
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


def test_llm_provider_extract_not_implemented_m2a(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = LLMABSAProvider("openai")
    assert isinstance(provider, LLMABSAProvider)
    assert provider.name == "llm:openai"


def test_get_provider_returns_llm_when_env_set(monkeypatch):
    monkeypatch.setenv("ABSA_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = get_provider()
    assert isinstance(provider, LLMABSAProvider)
    assert provider.backend == "anthropic"


def test_mock_provider_output_is_pydantic_abas_output():
    provider = MockABSAProvider()
    out = provider.extract("battery charging bluetooth")
    assert isinstance(out, ABSAOutput)
