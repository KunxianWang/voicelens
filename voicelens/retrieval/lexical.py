"""BM25 lexical retriever over the indexed review corpus.

The lexical baseline complements the dense (sentence-transformers)
retriever from M3A. Dense retrieval picks up paraphrased complaints;
BM25 catches exact-keyword queries (model numbers, brand-specific
phrases, ASINs).

The index is built in memory at retriever construction time by
scrolling Qdrant for ``(review_id, text_raw + aspect summary)``. That
keeps the lexical and dense retrievers operating on **the same
corpus** so eval metrics compare apples to apples.

If the corpus is empty the retriever still works — :func:`rank` just
returns ``[]`` for every query.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from voicelens.retrieval.qdrant_index import build_embedding_text

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer.

    Cheap and consistent — both the index build and the query path go
    through the exact same function so BM25 scores are reproducible.
    """
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class LexicalHit:
    review_id: int
    score: float
    payload: dict[str, Any]


def _scroll_corpus(
    client: QdrantClient,
    collection: str,
    *,
    scan_limit: int | None = None,
    chunk: int = 256,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        next_chunk_size = chunk
        if scan_limit is not None:
            remaining = scan_limit - len(payloads)
            if remaining <= 0:
                break
            next_chunk_size = min(next_chunk_size, remaining)
        rows, offset = client.scroll(
            collection_name=collection,
            limit=next_chunk_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not rows:
            break
        for point in rows:
            payload = dict(getattr(point, "payload", None) or {})
            payloads.append(payload)
        if offset is None:
            break
    return payloads


class BM25LexicalRetriever:
    """In-memory BM25 over Qdrant payloads.

    The corpus document for each indexed review is
    :func:`build_embedding_text` of ``(text_raw, mentions)``, matching
    what the dense embedder saw. That symmetry matters for the M3B
    hybrid retriever — dense and lexical should compete over the same
    text, not against each other's input distribution.
    """

    def __init__(self, payloads: Sequence[dict[str, Any]]) -> None:
        self._payloads = [dict(p) for p in payloads]
        # Reconstruct the structured mention list per payload so
        # ``build_embedding_text`` gets the same input shape the indexer
        # used at write time.
        self._docs: list[str] = []
        for payload in self._payloads:
            mentions = _reconstruct_mentions(payload)
            self._docs.append(
                build_embedding_text(str(payload.get("text_raw") or ""), mentions)
            )
        self._tokenized = [tokenize(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    @classmethod
    def from_qdrant(
        cls,
        client: QdrantClient,
        collection: str,
        *,
        scan_limit: int | None = None,
    ) -> BM25LexicalRetriever:
        payloads = _scroll_corpus(client, collection, scan_limit=scan_limit)
        return cls(payloads)

    @property
    def corpus_size(self) -> int:
        return len(self._docs)

    def rank(self, query: str, *, limit: int) -> list[LexicalHit]:
        if self._bm25 is None or not self._tokenized:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        scored = list(enumerate(scores))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        hits: list[LexicalHit] = []
        for idx, score in scored[: max(limit, 0)]:
            payload = self._payloads[idx]
            rid = payload.get("review_id")
            if rid is None:
                continue
            hits.append(
                LexicalHit(
                    review_id=int(rid),
                    score=float(score),
                    payload=payload,
                )
            )
        return hits


def _reconstruct_mentions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild a parallel-list-style mention sequence from the payload."""
    aspect_codes = payload.get("aspect_codes") or []
    sentiments = payload.get("sentiments") or []
    severities = payload.get("severities") or []
    mentions: list[dict[str, Any]] = []
    for i, code in enumerate(aspect_codes):
        if not isinstance(code, str):
            continue
        mentions.append(
            {
                "aspect_code": code,
                "sentiment": sentiments[i] if i < len(sentiments) else None,
                "severity": severities[i] if i < len(severities) else None,
            }
        )
    return mentions


def filter_payloads(
    payloads: Iterable[dict[str, Any]],
    *,
    brand: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
) -> list[dict[str, Any]]:
    """Apply the same predicates as the Qdrant Filter, in Python.

    Used by lexical and hybrid modes since BM25 scoring happens outside
    Qdrant; the filter has to be enforced post-hoc on the lexical hits.
    """
    out: list[dict[str, Any]] = []
    for payload in payloads:
        if brand and (payload.get("brand") or "") != brand:
            continue
        if aspect and aspect not in (payload.get("aspect_codes") or []):
            continue
        if sentiment and sentiment not in (payload.get("sentiments") or []):
            continue
        rating = payload.get("rating")
        if rating_min is not None and (rating is None or rating < rating_min):
            continue
        if rating_max is not None and (rating is None or rating > rating_max):
            continue
        out.append(payload)
    return out
