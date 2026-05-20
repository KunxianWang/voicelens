"""Grid-tune hybrid retrieval fusion over the golden set.

M3B found plain RRF (``rrf_equal``) underperforming the pure lexical
baseline — the weaker dense ranking diluted the fusion. This script
sweeps fusion strategies and lexical/dense weights so the best
configuration is picked from evidence, not guesswork.

It evaluates, on the same goldens:

- ``dense`` and ``lexical`` baselines,
- ``hybrid`` (rrf_equal) and ``lexical_first``,
- ``hybrid_weighted`` at lexical_weight 0.6 / 0.7 / 0.8 / 0.9.

Output: ``data/eval/retrieval_tuning_summary.csv`` — one row per
configuration with the headline metrics, easiest to eyeball as a table.
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

from scripts.evaluate_retrieval import evaluate_retrieval, resolve_goldens_path
from voicelens.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from voicelens.retrieval.embeddings import get_embedding_provider
from voicelens.retrieval.lexical import BM25LexicalRetriever

DEFAULT_OUTPUT = Path("data/eval/retrieval_tuning_summary.csv")
DEFAULT_LEXICAL_WEIGHTS = (0.6, 0.7, 0.8, 0.9)

TUNING_FIELDS = (
    "mode",
    "lexical_weight",
    "dense_weight",
    "hit_at_5",
    "recall_at_5",
    "capped_recall_at_5",
    "recall_at_20",
    "mrr_at_10",
    "ndcg_at_10",
)


def _tuning_row(
    mode: str,
    lexical_weight: float | str,
    dense_weight: float | str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Pick the headline metrics out of an :func:`evaluate_retrieval` summary."""
    return {
        "mode": mode,
        "lexical_weight": lexical_weight,
        "dense_weight": dense_weight,
        "hit_at_5": summary.get("hit_at_5"),
        "recall_at_5": summary.get("recall_at_5"),
        "capped_recall_at_5": summary.get("capped_recall_at_5"),
        "recall_at_20": summary.get("recall_at_20"),
        "mrr_at_10": summary.get("mrr_at_10"),
        "ndcg_at_10": summary.get("ndcg_at_10"),
    }


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-tune hybrid retrieval fusion.")
    parser.add_argument("--goldens", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--collection", default=None, help=f"Default: {QDRANT_COLLECTION}")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    goldens_path = resolve_goldens_path(args.goldens)
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

    def _eval(mode: str, lexical_weight: float, dense_weight: float) -> dict[str, Any]:
        return evaluate_retrieval(
            goldens=goldens, mode=mode, client=client, collection=collection,
            embedder=embedder, bm25=bm25, limit=args.limit,
            lexical_weight=lexical_weight, dense_weight=dense_weight,
        )["summary"]

    rows: list[dict[str, Any]] = []
    # Baselines + fixed fusion strategies.
    rows.append(_tuning_row("dense", "", "", _eval("dense", 0.5, 0.5)))
    rows.append(_tuning_row("lexical", "", "", _eval("lexical", 0.5, 0.5)))
    rows.append(_tuning_row("rrf_equal", 0.5, 0.5, _eval("hybrid", 0.5, 0.5)))
    rows.append(_tuning_row("lexical_first", "", "", _eval("lexical_first", 0.5, 0.5)))
    # Weighted RRF grid.
    for lw in DEFAULT_LEXICAL_WEIGHTS:
        dw = round(1.0 - lw, 2)
        rows.append(
            _tuning_row("rrf_weighted", lw, dw, _eval("hybrid_weighted", lw, dw))
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TUNING_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best = max(rows, key=lambda r: (r["recall_at_20"] or 0.0, r["mrr_at_10"] or 0.0))
    print(f"Goldens used            : {goldens_path}")
    print(f"Configurations evaluated: {len(rows)}")
    print(f"Wrote tuning summary    : {output}")
    print(
        "Best by recall@20       : "
        f"{best['mode']} (lw={best['lexical_weight']}) "
        f"recall@20={best['recall_at_20']} mrr@10={best['mrr_at_10']} "
        f"hit@5={best['hit_at_5']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
