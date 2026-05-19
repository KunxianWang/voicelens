"""Embedding providers used by the indexing flow and retrieval scripts.

Three concrete providers ship today:

- :class:`MockEmbeddingProvider` — deterministic, dependency-free,
  fixed-dimension. Same input always returns the same vector. Used in
  tests, the local smoke flow, and CI.

- :class:`LocalEmbeddingProvider` — wraps a ``sentence-transformers``
  model. The model is lazy-loaded so importing this module does not
  pull in torch. Default model is ``BAAI/bge-small-en-v1.5`` (384 dims)
  — a deliberate "good enough" pick for M3A; ``bge-m3`` and similar are
  scoped to a later milestone.

- :class:`OpenAIEmbeddingProvider` — stub that uses the OpenAI v1
  ``/v1/embeddings`` endpoint. Construction fails fast unless
  ``OPENAI_API_KEY`` is set so a misconfigured env is loud rather than
  silently degrading to a different provider.

A factory :func:`get_embedding_provider` resolves a provider name
(``"mock" | "local" | "openai"``) to a concrete instance, mirroring the
ABSA provider factory in style.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

PROVIDER_MOCK = "mock"
PROVIDER_LOCAL = "local"
PROVIDER_OPENAI = "openai"

KNOWN_PROVIDERS: tuple[str, ...] = (PROVIDER_MOCK, PROVIDER_LOCAL, PROVIDER_OPENAI)

DEFAULT_MOCK_DIMENSION = 32
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class EmbeddingProvider(ABC):
    """Abstract embedding provider. Subclasses produce dense vectors.

    Two attributes are part of the public contract:

    - ``name`` — short identifier (``"mock"`` / ``"local:<model>"``);
      written into payloads and used for instance-level logging.
    - ``dimension`` — vector size. Must remain constant for the lifetime
      of the provider; the indexing flow reads this to size the Qdrant
      collection.
    """

    name: str = "abstract"
    dimension: int = 0

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense vector per input text. Must not reorder."""


def _hash_to_floats(seed_bytes: bytes, dimension: int) -> list[float]:
    """Deterministic hash-to-vector helper used by the mock provider.

    Produces a length-``dimension`` list of floats in roughly ``[-1, 1]``
    by chunking the SHA-256 of ``seed_bytes`` and extending with further
    hash rounds until we have enough entropy. The vector is L2-normalised
    so the cosine distance Qdrant uses behaves sensibly.
    """
    floats: list[float] = []
    counter = 0
    while len(floats) < dimension:
        digest = hashlib.sha256(seed_bytes + counter.to_bytes(4, "big")).digest()
        for offset in range(0, len(digest), 4):
            if len(floats) >= dimension:
                break
            raw = struct.unpack(">I", digest[offset : offset + 4])[0]
            floats.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
        counter += 1
    norm = math.sqrt(sum(f * f for f in floats)) or 1.0
    return [f / norm for f in floats]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic SHA-256-based embedding provider.

    Same input always returns the same vector (modulo ``dimension``).
    No external dependencies; safe to run in tests and CI.
    """

    name = PROVIDER_MOCK

    def __init__(self, dimension: int = DEFAULT_MOCK_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension!r}")
        self.dimension = int(dimension)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_to_floats(str(t).encode("utf-8"), self.dimension) for t in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """``sentence-transformers``-backed provider.

    The model is loaded on first ``embed_batch`` call so importing the
    retrieval package never imports torch. If the dependency is missing
    or the model can't be loaded, the constructor raises a clear
    :class:`RuntimeError` directing the user at how to install or
    configure it.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model_name_str = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_LOCAL_MODEL
        self.name = f"local:{self.model_name_str}"
        self._model = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:  # type: ignore[override]
        if self._dimension is None:
            self._load_model()
        return int(self._dimension or 0)

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only when dep missing
            raise RuntimeError(
                "LocalEmbeddingProvider requires sentence-transformers. "
                "Install it with `pip install sentence-transformers` "
                "or switch EMBEDDING_PROVIDER to mock/openai."
            ) from exc
        model = SentenceTransformer(self.model_name_str)
        self._model = model
        try:
            self._dimension = int(model.get_sentence_embedding_dimension())
        except Exception:  # pragma: no cover - depends on installed model
            sample = model.encode(["dimension-probe"], convert_to_numpy=True)
            self._dimension = int(getattr(sample, "shape", (1, 0))[-1])

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            self._load_model()
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vec)) for vec in vectors]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI ``/v1/embeddings``-backed provider.

    Fails fast unless ``OPENAI_API_KEY`` is set. Network errors propagate
    so a misconfigured deployment is loud, not silently degraded.
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAIEmbeddingProvider requires OPENAI_API_KEY in the environment. "
                "Configure it in .env or your shell, or use EMBEDDING_PROVIDER=local."
            )
        self.api_key = api_key
        self.model_id = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_OPENAI_MODEL
        self.name = f"openai:{self.model_id}"
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:  # type: ignore[override]
        if self._dimension is None:
            # Probe with a single short input so the OpenAI account is the
            # source of truth for dim. Adds one tiny request per process.
            self._dimension = len(self.embed_batch(["dimension-probe"])[0])
        return int(self._dimension)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - real API
        payload = {"model": self.model_id, "input": list(texts)}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"OpenAI /v1/embeddings HTTPError {exc.code}: {detail}") from exc
        return [item["embedding"] for item in data.get("data", [])]


def get_embedding_provider(
    name: str | None = None, *, model: str | None = None, dimension: int | None = None
) -> EmbeddingProvider:
    """Resolve a provider name to a concrete instance.

    Resolution order: explicit ``name`` arg > ``EMBEDDING_PROVIDER`` env
    var > ``mock``. ``model`` overrides ``EMBEDDING_MODEL`` for local /
    openai; ignored by mock. ``dimension`` is mock-only.
    """
    resolved = (name or os.getenv("EMBEDDING_PROVIDER") or PROVIDER_MOCK).strip().lower()
    if resolved == PROVIDER_MOCK:
        return MockEmbeddingProvider(dimension or DEFAULT_MOCK_DIMENSION)
    if resolved == PROVIDER_LOCAL:
        return LocalEmbeddingProvider(model=model)
    if resolved == PROVIDER_OPENAI:
        return OpenAIEmbeddingProvider(model=model)
    raise ValueError(f"Unknown embedding provider: {resolved!r}. Expected one of {KNOWN_PROVIDERS}.")


def chunked(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    """Yield consecutive ``size``-chunks from ``iterable``."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    bucket: list[str] = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
