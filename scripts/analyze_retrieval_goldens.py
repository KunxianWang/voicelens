"""Diagnose retrieval-golden quality before tuning or RAG.

M3B measured retrieval below the promotion gates, but the goldens are
weakly supervised (evidence_quote phrase matching) — a low score may be
the gold set's fault, not the retriever's. This script audits the
goldens themselves and flags the rows a human should review.

For each golden it reports:

- how many gold ids are actually indexed in Qdrant (you cannot retrieve
  what was never embedded),
- ``max_possible_recall_at_{5,10}`` — the ceiling any retriever could
  hit given the indexed/missing split and the gold count,
- whether every evaluated mode missed the query (``no_gold_hit`` in all
  of the ``retrieval_errors_<mode>.csv`` files written by
  ``evaluate_retrieval.py``).

Output: ``data/eval/retrieval_golden_diagnostics.csv``. Nothing is
overwritten in the goldens file — refinement stays a manual step.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from voicelens.config import QDRANT_COLLECTION, QDRANT_URL

DEFAULT_GOLDENS = Path("data/eval/retrieval_goldens.jsonl")
REFINED_GOLDENS = Path("data/eval/retrieval_goldens_refined.jsonl")
DEFAULT_OUTPUT = Path("data/eval/retrieval_golden_diagnostics.csv")
ERROR_GLOB = "data/eval/retrieval_errors_*.csv"

DIAGNOSTIC_FIELDS = (
    "query_id",
    "query",
    "expected_aspect",
    "expected_sentiment",
    "num_gold",
    "gold_review_ids_present",
    "gold_review_ids_missing",
    "max_possible_recall_at_5",
    "max_possible_recall_at_10",
    "notes",
    "needs_manual_review",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def indexed_review_ids(client: QdrantClient, collection: str) -> set[int]:
    """Scroll the whole collection and collect every ``review_id`` payload."""
    ids: set[int] = set()
    offset = None
    while True:
        rows, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=["review_id"],
            with_vectors=False,
        )
        for point in rows:
            payload = getattr(point, "payload", None) or {}
            rid = payload.get("review_id")
            if rid is not None:
                ids.add(int(rid))
        if offset is None or not rows:
            break
    return ids


def load_no_gold_hit_counts(
    error_glob: str = ERROR_GLOB,
) -> tuple[dict[str, int], int]:
    """Count, per query_id, how many modes classified it ``no_gold_hit``.

    Returns ``(counts, n_modes_seen)``. ``n_modes_seen`` is the number of
    distinct per-mode error files found, so the caller can tell "missed
    in every mode" from "missed in the only mode we have a file for".
    """
    counts: dict[str, int] = {}
    files = sorted(glob.glob(error_glob))
    for path in files:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("error_type") == "no_gold_hit":
                    qid = row.get("query_id") or ""
                    counts[qid] = counts.get(qid, 0) + 1
    return counts, len(files)


def diagnose_golden(
    golden: dict[str, Any],
    indexed_ids: set[int],
    *,
    no_gold_hit_counts: dict[str, int],
    n_modes_seen: int,
) -> dict[str, Any]:
    """Build one diagnostics row + the ``needs_manual_review`` verdict."""
    gold = [int(g) for g in (golden.get("gold_review_ids") or [])]
    num_gold = len(gold)
    present = [g for g in gold if g in indexed_ids]
    missing = [g for g in gold if g not in indexed_ids]

    def max_recall(k: int) -> float:
        if num_gold == 0:
            return 0.0
        return round(min(len(present), k) / num_gold, 6)

    mpr5 = max_recall(5)
    mpr10 = max_recall(10)

    qid = str(golden.get("query_id") or "")
    all_modes_miss = (
        n_modes_seen > 0
        and no_gold_hit_counts.get(qid, 0) >= n_modes_seen
    )

    reasons: list[str] = []
    if num_gold == 0:
        reasons.append("no_gold")
    if num_gold > 10:
        reasons.append("too_many_gold")
    if num_gold > 0 and mpr5 < 0.75:
        reasons.append("low_max_recall_at_5")
    if missing:
        reasons.append("gold_missing_from_index")
    if all_modes_miss:
        reasons.append("no_gold_hit_all_modes")

    return {
        "query_id": qid,
        "query": golden.get("query"),
        "expected_aspect": golden.get("expected_aspect"),
        "expected_sentiment": golden.get("expected_sentiment"),
        "num_gold": num_gold,
        "gold_review_ids_present": json.dumps(present),
        "gold_review_ids_missing": json.dumps(missing),
        "max_possible_recall_at_5": mpr5,
        "max_possible_recall_at_10": mpr10,
        "notes": "; ".join(reasons) if reasons else "ok",
        "needs_manual_review": bool(reasons),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(DIAGNOSTIC_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit retrieval golden quality.")
    parser.add_argument("--goldens", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--collection", default=None, help=f"Default: {QDRANT_COLLECTION}")
    parser.add_argument("--qdrant-url", default=None, help=f"Default: {QDRANT_URL}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    goldens_path = _resolve_goldens(args.goldens)
    if not goldens_path.exists():
        print(f"ERROR: goldens file {goldens_path} is missing.", file=sys.stderr)
        return 2
    goldens = _read_jsonl(goldens_path)

    client = _open_client(args.qdrant_url)
    collection = args.collection or QDRANT_COLLECTION
    if not client.collection_exists(collection):
        print(f"ERROR: collection {collection!r} does not exist.", file=sys.stderr)
        return 3
    indexed = indexed_review_ids(client, collection)
    no_gold_hit_counts, n_modes_seen = load_no_gold_hit_counts()

    rows = [
        diagnose_golden(
            g, indexed,
            no_gold_hit_counts=no_gold_hit_counts,
            n_modes_seen=n_modes_seen,
        )
        for g in goldens
    ]
    output = Path(args.output)
    _write_csv(rows, output)

    flagged = [r for r in rows if r["needs_manual_review"]]
    print(f"Goldens analysed        : {len(rows)} ({goldens_path})")
    print(f"Indexed points          : {len(indexed)} in {collection!r}")
    print(f"Per-mode error files    : {n_modes_seen}")
    print(f"Flagged for review      : {len(flagged)}")
    if flagged:
        reason_counts: dict[str, int] = {}
        for r in flagged:
            for reason in str(r["notes"]).split("; "):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<26}: {n}")
    print(f"Wrote diagnostics       : {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
