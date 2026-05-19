from __future__ import annotations

import pytest

from voicelens.retrieval.embeddings import (
    DEFAULT_MOCK_DIMENSION,
    MockEmbeddingProvider,
    chunked,
    get_embedding_provider,
)


def test_mock_provider_is_deterministic_for_same_input():
    p = MockEmbeddingProvider(dimension=16)
    a = p.embed_batch(["hello world"])[0]
    b = p.embed_batch(["hello world"])[0]
    assert a == b
    assert len(a) == 16


def test_mock_provider_different_inputs_produce_different_vectors():
    p = MockEmbeddingProvider(dimension=16)
    [a, b] = p.embed_batch(["alpha", "beta"])
    assert a != b


def test_mock_provider_dimension_property_matches_output():
    p = MockEmbeddingProvider(dimension=24)
    vectors = p.embed_batch(["x", "y", "z"])
    assert p.dimension == 24
    assert all(len(v) == 24 for v in vectors)


def test_mock_provider_vector_is_normalised():
    """Cosine distance in Qdrant only behaves on normalised vectors."""
    p = MockEmbeddingProvider(dimension=64)
    [vec] = p.embed_batch(["some review text"])
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_mock_provider_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="dimension"):
        MockEmbeddingProvider(dimension=0)
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dimension=-3)


def test_factory_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    provider = get_embedding_provider()
    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dimension == DEFAULT_MOCK_DIMENSION


def test_factory_respects_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    assert isinstance(get_embedding_provider(), MockEmbeddingProvider)


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_provider("totally-bogus")


def test_factory_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embedding_provider("openai")


def test_chunked_yields_full_and_partial_chunks():
    chunks = list(chunked(["a", "b", "c", "d", "e"], 2))
    assert chunks == [["a", "b"], ["c", "d"], ["e"]]


def test_chunked_rejects_zero_or_negative_size():
    with pytest.raises(ValueError):
        list(chunked(["a"], 0))


def test_chunked_handles_empty_iterable():
    assert list(chunked([], 4)) == []
