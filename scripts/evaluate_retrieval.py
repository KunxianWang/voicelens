"""Run the retrieval evaluation harness over a golden set.

Loads ``data/eval/retrieval_goldens.jsonl``, runs each query through
the configured retriever (``dense`` / ``lexical`` / ``hybrid``),
computes per-query and aggregate metrics, and writes three artefacts:

- ``data/eval/retrieval_eval_results.csv`` (one row per query)
- ``data/eval/retrieval_eval_summary.json`` (aggregate)
- ``data/eval/retrieval_errors.csv`` (per-query error classification)

Filters from each golden (``expected_aspect`` / ``expected_sentiment``
/ ``expected_brand``) are applied at retrieval time. The unfiltered
ranking is also collected so the error classifier can flag
``filter_too_strict``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from voicelens.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from voicelens.eval.retrieval_eval import (
    aggregate_metrics,
    capped_recall_at_k,
    classify_retrieval_error,
    filter_precision_at_k,
    hit_at_k,
    ndcg_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank,
)
from voicelens.retrieval.embeddings import get_embedding_provider
from voicelens.retrieval.filters import build_search_filter
from voicelens.retrieval.hybrid import hybrid_search, lexical_search
from voicelens.retrieval.lexical import BM25LexicalRetriever
from voicelens.retrieval.search import SearchHit, retrieve

DEFAULT_GOLDENS = Path("data/eval/retrieval_goldens.jsonl")
REFINED_GOLDENS = Path("data/eval/retrieval_goldens_refined.jsonl")
DEFAULT_RESULTS = Path("data/eval/retrieval_eval_results.csv")
DEFAULT_SUMMARY = Path("data/eval/retrieval_eval_summary.json")
DEFAULT_ERRORS = Path("data/eval/retrieval_errors.csv")
DEFAULT_KS = (5, 10, 20)
DEFAULT_HIT_KS = (1, 5, 10, 20)
DEFAULT_MRR_K = 10

VALID_MODES = ("dense", "lexical", "hybrid", "hybrid_weighted", "lexical_first")
# Hybrid-family modes and the fusion strategy each one drives.
_HYBRID_FUSION_BY_MODE = {
    "hybrid": "rrf_equal",
    "hybrid_weighted": "rrf_weighted",
    "lexical_first": "lexical_first",
}


def resolve_goldens_path(arg: str | None) -> Path:
    """Pick the goldens file: explicit arg > refined file > base file.

    Supports the M3C manual-refinement workflow — once a human edits
    ``retrieval_goldens_refined.jsonl`` it is picked up automatically,
    but the weakly-supervised base set is the fallback.
    """
    if arg:
        return Path(arg)
    if REFINED_GOLDENS.exists():
        return REFINED_GOLDENS
    return DEFAULT_GOLDENS


@dataclass
class _QueryRun:
    golden: dict[str, Any]
    ranked_filtered: list[int]
    ranked_unfiltered: list[int]
    top_hits: list[SearchHit]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hit_ids(hits: list[SearchHit]) -> list[int]:
    return [h.review_id for h in hits if h.review_id is not None]


def _open_client(url: str | None) -> QdrantClient:
    target = (url or QDRANT_URL).strip()
    if target in ("", ":memory:", "memory://"):
        return QdrantClient(":memory:")
    return QdrantClient(url=target)


def _dense(
    *,
    client: QdrantClient,
    collection: str,
    embedder: Any,
    query: str,
    limit: int,
    filter_payload: dict[str, Any] | None,
) -> list[SearchHit]:
    filter_ = build_search_filter(**filter_payload) if filter_payload else None
    return retrieve(
        client=client,
        collection=collection,
        embedder=embedder,
        query=query,
        limit=limit,
        filter_=filter_,
    )


def _lexical(
    *,
    bm25: BM25LexicalRetriever,
    query: str,
    limit: int,
    filter_payload: dict[str, Any] | None,
) -> list[SearchHit]:
    kwargs = filter_payload or {}
    return lexical_search(
        bm25=bm25,
        query=query,
        limit=limit,
        brand=kwargs.get("brand"),
        aspect=kwargs.get("aspect"),
        sentiment=kwargs.get("sentiment"),
        rating_min=kwargs.get("rating_min"),
        rating_max=kwargs.get("rating_max"),
    )


def _hybrid(
    *,
    client: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: BM25LexicalRetriever,
    query: str,
    limit: int,
    filter_payload: dict[str, Any] | None,
    oversample_k: int,
    fusion: str,
    lexical_weight: float,
    dense_weight: float,
) -> list[SearchHit]:
    f = filter_payload or {}
    dense_filter = build_search_filter(**f) if f else None
    return hybrid_search(
        client=client,
        collection=collection,
        embedder=embedder,
        bm25=bm25,
        query=query,
        limit=limit,
        dense_filter=dense_filter,
        lexical_brand=f.get("brand"),
        lexical_aspect=f.get("aspect"),
        lexical_sentiment=f.get("sentiment"),
        lexical_rating_min=f.get("rating_min"),
        lexical_rating_max=f.get("rating_max"),
        oversample_k=oversample_k,
        fusion=fusion,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
    )


def _golden_filter_payload(golden: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if golden.get("expected_aspect"):
        out["aspect"] = str(golden["expected_aspect"])
    if golden.get("expected_sentiment"):
        out["sentiment"] = str(golden["expected_sentiment"])
    if golden.get("expected_brand"):
        out["brand"] = str(golden["expected_brand"])
    return out


def _run_query(
    *,
    golden: dict[str, Any],
    mode: str,
    client: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: BM25LexicalRetriever | None,
    limit: int,
    oversample_k: int,
    lexical_weight: float,
    dense_weight: float,
) -> _QueryRun:
    query = str(golden.get("query") or "")
    filter_payload = _golden_filter_payload(golden)

    def _dispatch(fp: dict[str, Any] | None) -> list[SearchHit]:
        if mode == "dense":
            return _dense(
                client=client, collection=collection, embedder=embedder,
                query=query, limit=limit, filter_payload=fp,
            )
        if mode == "lexical":
            assert bm25 is not None
            return _lexical(bm25=bm25, query=query, limit=limit, filter_payload=fp)
        if mode in _HYBRID_FUSION_BY_MODE:
            assert bm25 is not None
            return _hybrid(
                client=client, collection=collection, embedder=embedder,
                bm25=bm25, query=query, limit=limit, filter_payload=fp,
                oversample_k=oversample_k,
                fusion=_HYBRID_FUSION_BY_MODE[mode],
                lexical_weight=lexical_weight, dense_weight=dense_weight,
            )
        raise ValueError(f"unknown retrieval mode {mode!r}")

    filtered = _dispatch(filter_payload or None)
    unfiltered = _dispatch(None) if filter_payload else filtered
    return _QueryRun(
        golden=golden,
        ranked_filtered=_hit_ids(filtered),
        ranked_unfiltered=_hit_ids(unfiltered),
        top_hits=filtered,
    )


def _per_query_metrics(
    runs: Iterable[_QueryRun],
    *,
    ks: tuple[int, ...],
    hit_ks: tuple[int, ...],
    mrr_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        gold = run.golden.get("gold_review_ids") or []
        ranked = run.ranked_filtered
        row: dict[str, Any] = {
            "query_id": run.golden.get("query_id"),
            "query": run.golden.get("query"),
            "expected_aspect": run.golden.get("expected_aspect"),
            "expected_sentiment": run.golden.get("expected_sentiment"),
            "expected_brand": run.golden.get("expected_brand"),
            "gold_count": len(gold),
            "retrieved_count": len(ranked),
        }
        if not gold:
            for hk in hit_ks:
                row[f"hit_at_{hk}"] = None
            for k in ks:
                row[f"recall_at_{k}"] = None
                row[f"ndcg_at_{k}"] = None
            row["capped_recall_at_5"] = None
            row["r_precision"] = None
            row[f"mrr_at_{mrr_k}"] = None
        else:
            for hk in hit_ks:
                row[f"hit_at_{hk}"] = hit_at_k(ranked, gold, hk)
            for k in ks:
                row[f"recall_at_{k}"] = recall_at_k(ranked, gold, k)
                row[f"ndcg_at_{k}"] = ndcg_at_k(ranked, gold, k)
            row["capped_recall_at_5"] = capped_recall_at_k(ranked, gold, 5)
            row["r_precision"] = r_precision(ranked, gold)
            row[f"mrr_at_{mrr_k}"] = reciprocal_rank(ranked, gold, mrr_k)
        row[f"filter_precision_at_{mrr_k}"] = filter_precision_at_k(
            run.top_hits,
            expected_aspect=run.golden.get("expected_aspect"),
            expected_sentiment=run.golden.get("expected_sentiment"),
            expected_brand=run.golden.get("expected_brand"),
            k=mrr_k,
        )
        rows.append(row)
    return rows


def _error_rows(runs: Iterable[_QueryRun], *, mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in runs:
        gold = run.golden.get("gold_review_ids") or []
        top_payload = run.top_hits[0].raw_payload if run.top_hits else None
        error_type = classify_retrieval_error(
            ranked=run.ranked_filtered,
            gold=gold,
            top_hit_payload=top_payload,
            expected_aspect=run.golden.get("expected_aspect"),
            expected_sentiment=run.golden.get("expected_sentiment"),
            retrieval_mode=mode,
            pre_filter_ranked=run.ranked_unfiltered,
        )
        top_hit = run.top_hits[0] if run.top_hits else None
        out.append(
            {
                "query_id": run.golden.get("query_id"),
                "query": run.golden.get("query"),
                "expected_aspect": run.golden.get("expected_aspect"),
                "expected_sentiment": run.golden.get("expected_sentiment"),
                "gold_review_ids": json.dumps(gold),
                "retrieved_review_ids": json.dumps(run.ranked_filtered[:20]),
                "hit_at_5": int(any(int(r) in set(gold) for r in run.ranked_filtered[:5])),
                "hit_at_10": int(any(int(r) in set(gold) for r in run.ranked_filtered[:10])),
                "hit_at_20": int(any(int(r) in set(gold) for r in run.ranked_filtered[:20])),
                "top_result_aspects": ";".join(top_hit.aspect_codes) if top_hit else "",
                "top_result_brand": top_hit.brand if top_hit else "",
                "error_type": error_type,
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_retrieval(
    *,
    goldens: list[dict[str, Any]],
    mode: str,
    client: QdrantClient,
    collection: str,
    embedder: Any,
    bm25: BM25LexicalRetriever | None,
    limit: int = 20,
    oversample_k: int = 50,
    ks: tuple[int, ...] = DEFAULT_KS,
    hit_ks: tuple[int, ...] = DEFAULT_HIT_KS,
    mrr_k: int = DEFAULT_MRR_K,
    lexical_weight: float = 0.75,
    dense_weight: float = 0.25,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if mode != "dense" and bm25 is None:
        raise ValueError(f"mode={mode!r} requires a BM25 retriever")

    runs = [
        _run_query(
            golden=g,
            mode=mode,
            client=client,
            collection=collection,
            embedder=embedder,
            bm25=bm25,
            limit=limit,
            oversample_k=oversample_k,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
        )
        for g in goldens
    ]
    per_query = _per_query_metrics(runs, ks=ks, hit_ks=hit_ks, mrr_k=mrr_k)
    summary = aggregate_metrics(per_query, ks=ks, mrr_k=mrr_k)
    errors = _error_rows(runs, mode=mode)
    error_breakdown = Counter(row["error_type"] for row in errors)
    summary["mode"] = mode
    summary["collection"] = collection
    summary["limit"] = limit
    if mode == "hybrid_weighted":
        summary["lexical_weight"] = lexical_weight
        summary["dense_weight"] = dense_weight
    summary["error_breakdown"] = dict(error_breakdown.most_common())
    summary["queries_with_gold"] = sum(1 for g in goldens if g.get("gold_review_ids"))
    summary["queries_without_gold"] = sum(
        1 for g in goldens if not g.get("gold_review_ids")
    )
    summary["hits_at_5"] = sum(1 for row in errors if row["hit_at_5"])
    summary["hits_at_10"] = sum(1 for row in errors if row["hit_at_10"])
    summary["hits_at_20"] = sum(1 for row in errors if row["hit_at_20"])
    return {
        "summary": summary,
        "per_query": per_query,
        "errors": errors,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality over the golden set.")
    parser.add_argument("--mode", choices=VALID_MODES, default="dense")
    parser.add_argument(
        "--goldens",
        default=None,
        help="Goldens JSONL. Default: refined file if present, else base set.",
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--errors", default=str(DEFAULT_ERRORS))
    parser.add_argument("--collection", default=None, help=f"Default: {QDRANT_COLLECTION}")
    parser.add_argument("--qdrant-url", default=None, help=f"Default: {QDRANT_URL}")
    parser.add_argument("--embedding-provider", default=None, help=f"Default: {EMBEDDING_PROVIDER}")
    parser.add_argument("--embedding-model", default=None, help=f"Default: {EMBEDDING_MODEL}")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--oversample-k", type=int, default=50, help="Pre-fusion K for hybrid mode")
    parser.add_argument(
        "--lexical-weight", type=float, default=0.75,
        help="Lexical ranker weight for mode=hybrid_weighted (default 0.75).",
    )
    parser.add_argument(
        "--dense-weight", type=float, default=0.25,
        help="Dense ranker weight for mode=hybrid_weighted (default 0.25).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    goldens_path = resolve_goldens_path(args.goldens)
    if not goldens_path.exists():
        print(
            f"ERROR: goldens file {goldens_path} is missing. "
            "Run `make build-retrieval-goldens` (or "
            "`python scripts/build_retrieval_goldens.py`) first.",
            file=sys.stderr,
        )
        return 2
    goldens = _read_jsonl(goldens_path)

    embedder = get_embedding_provider(
        args.embedding_provider or EMBEDDING_PROVIDER,
        model=args.embedding_model or EMBEDDING_MODEL,
    )
    client = _open_client(args.qdrant_url)
    collection = args.collection or QDRANT_COLLECTION
    if not client.collection_exists(collection):
        print(
            f"ERROR: Qdrant collection {collection!r} does not exist. "
            "Run `make embed-v2-1k` first.",
            file=sys.stderr,
        )
        return 3

    bm25 = None
    if args.mode != "dense":
        bm25 = BM25LexicalRetriever.from_qdrant(client, collection)

    result = evaluate_retrieval(
        goldens=goldens,
        mode=args.mode,
        client=client,
        collection=collection,
        embedder=embedder,
        bm25=bm25,
        limit=args.limit,
        oversample_k=args.oversample_k,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
    )

    _write_csv(result["per_query"], Path(args.results))
    _write_csv(result["errors"], Path(args.errors))
    # A mode-suffixed copy of the errors so analyze_retrieval_goldens.py can
    # cross-reference no_gold_hit across every mode without them clobbering
    # each other's canonical retrieval_errors.csv.
    errors_path = Path(args.errors)
    mode_errors = errors_path.with_name(f"retrieval_errors_{args.mode}.csv")
    _write_csv(result["errors"], mode_errors)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    summary = result["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"\nGoldens used            : {goldens_path}")
    print(f"Wrote per-query results : {args.results}")
    print(f"Wrote summary           : {args.summary}")
    print(f"Wrote errors            : {args.errors}")
    print(f"Wrote errors (per-mode) : {mode_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
