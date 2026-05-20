"""Export top retrieval candidates per query for manual golden refinement.

For every golden query this runs dense, lexical and hybrid retrieval
(filters from the golden applied) and dumps the top hits side by side
into ``data/eval/retrieval_review_candidates.csv``. A reviewer can then
scan each query's candidates, decide which review_ids really belong in
the gold set, and hand-edit ``retrieval_goldens_refined.jsonl``.

The ``currently_gold`` column shows what the weakly-supervised set
already has; ``suggested_action`` is left blank for the human to fill
(e.g. ``add`` / ``remove`` / ``keep``).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly puts only ``scripts/`` on sys.path; add the
# repo root so the ``scripts`` package (sibling modules) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from typing import Any

from qdrant_client import QdrantClient

from scripts.evaluate_retrieval import _dense, _golden_filter_payload, _hybrid, _lexical
from voicelens.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from voicelens.retrieval.embeddings import get_embedding_provider
from voicelens.retrieval.lexical import BM25LexicalRetriever
from voicelens.retrieval.search import SearchHit

DEFAULT_GOLDENS = Path("data/eval/retrieval_goldens.jsonl")
REFINED_GOLDENS = Path("data/eval/retrieval_goldens_refined.jsonl")
DEFAULT_OUTPUT = Path("data/eval/retrieval_review_candidates.csv")
CANDIDATE_MODES = ("dense", "lexical", "hybrid")

CANDIDATE_FIELDS = (
    "query_id",
    "query",
    "mode",
    "rank",
    "review_id",
    "score",
    "brand",
    "asin",
    "rating",
    "aspect_codes",
    "sentiments",
    "text_snippet",
    "evidence_quotes",
    "currently_gold",
    "suggested_action",
)


def build_candidate_rows(
    *,
    query_id: str,
    query: str,
    mode: str,
    hits: list[SearchHit],
    gold_ids: list[int],
    top_n: int,
) -> list[dict[str, Any]]:
    """Turn a ranked list of hits into reviewer-friendly candidate rows."""
    gold = {int(g) for g in gold_ids}
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits[:top_n], start=1):
        rid = hit.review_id
        snippet = (hit.text_raw or "").replace("\n", " ").replace("\r", " ").strip()
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "mode": mode,
                "rank": rank,
                "review_id": rid,
                "score": round(float(hit.score), 6),
                "brand": hit.brand,
                "asin": hit.asin,
                "rating": hit.rating,
                "aspect_codes": ";".join(hit.aspect_codes),
                "sentiments": ";".join(hit.sentiments),
                "text_snippet": snippet[:200],
                "evidence_quotes": " | ".join(hit.evidence_quotes),
                "currently_gold": rid in gold if rid is not None else False,
                "suggested_action": "",
            }
        )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _open_client(url: str | None) -> QdrantClient:
    target = (url or QDRANT_URL).strip()
    if target in ("", ":memory:", "memory://"):
        return QdrantClient(":memory:")
    return QdrantClient(url=target)


def _resolve_goldens(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if REFINED_GOLDENS.exists():
        return REFINED_GOLDENS
    return DEFAULT_GOLDENS


def _retrieve(
    *,
    mode: str,
    golden: dict[str, Any],
    client: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: BM25LexicalRetriever,
    top_n: int,
) -> list[SearchHit]:
    query = str(golden.get("query") or "")
    fp = _golden_filter_payload(golden) or None
    if mode == "dense":
        return _dense(
            client=client, collection=collection, embedder=embedder,
            query=query, limit=top_n, filter_payload=fp,
        )
    if mode == "lexical":
        return _lexical(bm25=bm25, query=query, limit=top_n, filter_payload=fp)
    return _hybrid(
        client=client, collection=collection, embedder=embedder, bm25=bm25,
        query=query, limit=top_n, filter_payload=fp, oversample_k=max(50, top_n * 3),
        fusion="rrf_equal", lexical_weight=0.5, dense_weight=0.5,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export retrieval candidates per query for manual review."
    )
    parser.add_argument("--goldens", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--collection", default=None, help=f"Default: {QDRANT_COLLECTION}")
    parser.add_argument("--qdrant-url", default=None, help=f"Default: {QDRANT_URL}")
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--top-n", type=int, default=15, help="Candidates per mode per query")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    goldens_path = _resolve_goldens(args.goldens)
    if not goldens_path.exists():
        print(f"ERROR: goldens file {goldens_path} is missing.", file=sys.stderr)
        return 2
    goldens = _read_jsonl(goldens_path)

    embedder = get_embedding_provider(
        args.embedding_provider or EMBEDDING_PROVIDER,
        model=args.embedding_model or EMBEDDING_MODEL,
    )
    client = _open_client(args.qdrant_url)
    collection = args.collection or QDRANT_COLLECTION
    if not client.collection_exists(collection):
        print(f"ERROR: collection {collection!r} does not exist.", file=sys.stderr)
        return 3
    bm25 = BM25LexicalRetriever.from_qdrant(client, collection)

    all_rows: list[dict[str, Any]] = []
    for golden in goldens:
        gold_ids = [int(g) for g in (golden.get("gold_review_ids") or [])]
        for mode in CANDIDATE_MODES:
            hits = _retrieve(
                mode=mode, golden=golden, client=client, collection=collection,
                embedder=embedder, bm25=bm25, top_n=args.top_n,
            )
            all_rows.extend(
                build_candidate_rows(
                    query_id=str(golden.get("query_id") or ""),
                    query=str(golden.get("query") or ""),
                    mode=mode,
                    hits=hits,
                    gold_ids=gold_ids,
                    top_n=args.top_n,
                )
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CANDIDATE_FIELDS))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Goldens used            : {goldens_path}")
    print(f"Queries x modes         : {len(goldens)} x {len(CANDIDATE_MODES)}")
    print(f"Candidate rows           : {len(all_rows)}")
    print(f"Wrote candidates        : {output}")
    print("Next: hand-edit data/eval/retrieval_goldens_refined.jsonl, then re-run eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
