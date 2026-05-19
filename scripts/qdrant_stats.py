"""Print a quick summary of the Qdrant collection used by the retrieval
layer.

Reports the collection name, point count, vector size, and a small
sample of payloads. Helpful to confirm an indexing run actually landed
something useful before invoking ``retrieval_smoke``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from qdrant_client import QdrantClient

from voicelens.config import QDRANT_COLLECTION, QDRANT_URL


def _open_client(url: str | None) -> QdrantClient:
    target = (url or QDRANT_URL).strip()
    if target in ("", ":memory:", "memory://"):
        return QdrantClient(":memory:")
    return QdrantClient(url=target)


def _vector_size(info: Any) -> int | None:
    cfg = getattr(info, "config", None)
    if cfg is None:
        return None
    params = getattr(cfg, "params", None)
    if params is None:
        return None
    vectors = getattr(params, "vectors", None)
    if vectors is None:
        return None
    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors, dict):
        for v in vectors.values():
            inner = getattr(v, "size", None)
            if isinstance(inner, int):
                return inner
    return None


def _scroll_payloads(
    client: QdrantClient, collection: str, *, scan_limit: int = 1000
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset = None
    pulled = 0
    while pulled < scan_limit:
        chunk, offset = client.scroll(
            collection_name=collection,
            limit=min(256, scan_limit - pulled),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not chunk:
            break
        for point in chunk:
            payload = dict(getattr(point, "payload", None) or {})
            payloads.append(payload)
        pulled += len(chunk)
        if offset is None:
            break
    return payloads


def collect_qdrant_stats(
    client: QdrantClient, collection: str, *, scan_limit: int = 1000
) -> dict[str, Any]:
    if not client.collection_exists(collection):
        return {"collection": collection, "exists": False}
    info = client.get_collection(collection)
    point_count = client.count(collection_name=collection, exact=True).count
    payloads = _scroll_payloads(client, collection, scan_limit=scan_limit)

    aspect_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for payload in payloads:
        for code in payload.get("aspect_codes") or []:
            if isinstance(code, str):
                aspect_counts[code] += 1
        for sent in payload.get("sentiments") or []:
            if isinstance(sent, str) and sent:
                sentiment_counts[sent] += 1
        for sev in payload.get("severities") or []:
            if isinstance(sev, str) and sev:
                severity_counts[sev] += 1
        brand = payload.get("brand")
        if isinstance(brand, str) and brand:
            brand_counts[brand] += 1
        status = payload.get("absa_status")
        if isinstance(status, str) and status:
            status_counts[status] += 1
    sample = payloads[:3]
    for payload in sample:
        if isinstance(payload.get("text_raw"), str) and len(payload["text_raw"]) > 240:
            payload["text_raw"] = payload["text_raw"][:240].rstrip() + "…"

    return {
        "collection": collection,
        "exists": True,
        "point_count": int(point_count),
        "vector_size": _vector_size(info),
        "scanned_payloads": len(payloads),
        "aspect_distribution": dict(aspect_counts.most_common()),
        "sentiment_distribution": dict(sentiment_counts.most_common()),
        "severity_distribution": dict(severity_counts.most_common()),
        "brand_distribution": dict(brand_counts.most_common(10)),
        "absa_status_distribution": dict(status_counts.most_common()),
        "payload_sample": sample,
    }


def _print(stats: dict[str, Any]) -> None:
    if not stats.get("exists"):
        print(f"Qdrant collection {stats['collection']!r}: does not exist")
        return
    print(f"== Qdrant collection: {stats['collection']!r} ==")
    print(f"  point_count   : {stats['point_count']}")
    print(f"  vector_size   : {stats['vector_size']}")
    print(f"  scanned       : {stats['scanned_payloads']} payloads")
    print()
    print("-- aspect distribution (over scanned payloads) --")
    for code, n in stats["aspect_distribution"].items():
        print(f"  {code:<16} {n}")
    if not stats["aspect_distribution"]:
        print("  (no aspect_codes on scanned payloads)")
    print()
    print("-- sentiment distribution --")
    for sent, n in stats["sentiment_distribution"].items():
        print(f"  {sent:<10} {n}")
    print()
    print("-- severity distribution --")
    for sev, n in stats["severity_distribution"].items():
        print(f"  {sev:<10} {n}")
    print()
    print("-- absa_status distribution --")
    for status, n in stats["absa_status_distribution"].items():
        print(f"  {status:<14} {n}")
    print()
    print("-- top brands --")
    for brand, n in stats["brand_distribution"].items():
        print(f"  {brand:<16} {n}")
    print()
    print("-- payload sample (first 3 points) --")
    for i, payload in enumerate(stats["payload_sample"], start=1):
        print(f"  [{i}]")
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Qdrant collection stats.")
    parser.add_argument("--collection", default=None, help=f"Default: {QDRANT_COLLECTION}")
    parser.add_argument("--qdrant-url", default=None, help=f"Default: {QDRANT_URL}")
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=1000,
        help="Max payloads to scroll for distribution stats",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    client = _open_client(args.qdrant_url)
    collection = args.collection or QDRANT_COLLECTION
    stats = collect_qdrant_stats(client, collection, scan_limit=args.scan_limit)
    _print(stats)
    return 0 if stats.get("exists") else 2


if __name__ == "__main__":
    sys.exit(main())
