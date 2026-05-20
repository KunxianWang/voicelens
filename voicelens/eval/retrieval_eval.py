"""Retrieval-quality metrics + Reciprocal Rank Fusion (RRF).

Pure functions. All inputs are either ranked lists of ``review_id``
ints or ``set[int]`` of gold ids — no Qdrant client, no embeddings.
The runner in ``scripts/evaluate_retrieval.py`` is responsible for
producing those lists; this module turns them into numbers.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def recall_at_k(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """Fraction of ``gold`` review_ids present in the top-``k`` of ``ranked``.

    Defined as 0.0 when ``gold`` is empty so an empty-gold query is
    treated as "no signal" rather than an error.
    """
    gold_set = {int(g) for g in gold}
    if not gold_set or k <= 0:
        return 0.0
    head = list(ranked[:k])
    hit = sum(1 for rid in head if int(rid) in gold_set)
    return round(hit / len(gold_set), 6)


def hit_at_k(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """1.0 if **any** gold doc appears in the top-``k``, else 0.0.

    A "did the user see at least one good answer" signal — the most
    UX-relevant metric for a RAG answerer that quotes a few reviews.
    """
    gold_set = {int(g) for g in gold}
    if not gold_set or k <= 0:
        return 0.0
    return 1.0 if any(int(rid) in gold_set for rid in ranked[:k]) else 0.0


def capped_recall_at_k(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """Recall@k with the denominator capped at ``min(num_gold, k)``.

    Plain Recall@5 is misleadingly low when a query has more than 5
    gold docs — the top-5 physically cannot contain them all. Capping
    the denominator measures "of the slots available, how many landed"
    so high-gold queries are not unfairly penalised.
    """
    gold_set = {int(g) for g in gold}
    if not gold_set or k <= 0:
        return 0.0
    head = list(ranked[:k])
    hit = sum(1 for rid in head if int(rid) in gold_set)
    denom = min(len(gold_set), k)
    return round(hit / denom, 6) if denom else 0.0


def reciprocal_rank(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """``1 / rank`` of the first gold hit within the top-``k``; 0.0 if none."""
    gold_set = {int(g) for g in gold}
    if not gold_set or k <= 0:
        return 0.0
    for rank, rid in enumerate(ranked[:k], start=1):
        if int(rid) in gold_set:
            return round(1.0 / rank, 6)
    return 0.0


def r_precision(ranked: Sequence[int], gold: Iterable[int]) -> float:
    """Precision at rank ``R``, where ``R`` is the number of gold docs.

    A single rank-aware number that needs no ``k``: it asks "if we cut
    the ranking at exactly as many results as there are gold docs, what
    fraction are correct?". 0.0 when gold is empty.
    """
    gold_set = {int(g) for g in gold}
    r = len(gold_set)
    if r == 0:
        return 0.0
    hit = sum(1 for rid in ranked[:r] if int(rid) in gold_set)
    return round(hit / r, 6)


def mean_reciprocal_rank_at_k(
    rankings: Sequence[Sequence[int]],
    golds: Sequence[Iterable[int]],
    k: int,
) -> float:
    """Mean of :func:`reciprocal_rank` across paired ``(ranking, gold)``."""
    if not rankings or not golds:
        return 0.0
    if len(rankings) != len(golds):
        raise ValueError("rankings and golds must be the same length")
    rrs = [reciprocal_rank(r, g, k) for r, g in zip(rankings, golds, strict=True)]
    return round(sum(rrs) / len(rrs), 6) if rrs else 0.0


def dcg_at_k(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """Binary-relevance DCG: gain 1 if doc is in gold, else 0."""
    gold_set = {int(g) for g in gold}
    if not gold_set or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, rid in enumerate(ranked[:k], start=1):
        if int(rid) in gold_set:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ideal_dcg_at_k(gold: Iterable[int], k: int) -> float:
    """DCG of the perfect ranking (all gold docs at the top)."""
    gold_count = len({int(g) for g in gold})
    if gold_count == 0 or k <= 0:
        return 0.0
    n_relevant = min(gold_count, k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, n_relevant + 1))


def ndcg_at_k(ranked: Sequence[int], gold: Iterable[int], k: int) -> float:
    """Binary-relevance NDCG@k. 0.0 when gold is empty."""
    idcg = ideal_dcg_at_k(gold, k)
    if idcg == 0:
        return 0.0
    return round(dcg_at_k(ranked, gold, k) / idcg, 6)


def filter_precision_at_k(
    hits: Sequence[Mapping[str, Any]],
    *,
    expected_aspect: str | None = None,
    expected_sentiment: str | None = None,
    expected_brand: str | None = None,
    k: int = 10,
) -> float | None:
    """Of the top-``k`` hits, what fraction match the expected facets?

    Returns ``None`` when no expectation is set so the runner can show
    a clear ``"n/a"`` instead of a misleading 0.0. Each hit is a dict
    or any object with ``brand`` / ``aspect_codes`` / ``sentiments``
    accessible via ``__getitem__`` — designed to accept both raw
    Qdrant payloads and :class:`voicelens.retrieval.search.SearchHit`.
    """
    if expected_aspect is None and expected_sentiment is None and expected_brand is None:
        return None
    if not hits or k <= 0:
        return 0.0
    relevant = 0
    examined = 0
    for hit in hits[:k]:
        examined += 1
        payload = _payload_view(hit)
        ok = True
        if expected_aspect is not None:
            ok = ok and expected_aspect in payload.get("aspect_codes", [])
        if expected_sentiment is not None:
            ok = ok and expected_sentiment in payload.get("sentiments", [])
        if expected_brand is not None:
            ok = ok and (payload.get("brand") or "") == expected_brand
        if ok:
            relevant += 1
    return round(relevant / examined, 6) if examined else 0.0


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    k_constant: int = 60,
    top_k: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[int]:
    """Combine multiple ranked lists into one via Reciprocal Rank Fusion.

    Score for doc ``d`` is ``sum(w_i / (k_constant + rank_i(d)))`` over
    rankers that placed ``d``. The default ``k_constant=60`` follows the
    Cormack/Clarke/Buettcher RRF paper. Returns the fused doc list in
    descending fused-score order, optionally trimmed to ``top_k``.

    ``weights`` (one per input ranking) lets a stronger retriever pull
    harder on the fused order — e.g. ``weights=[0.25, 0.75]`` to favour
    a lexical ranker over a weaker dense one. When ``None`` every ranker
    contributes equally (classic RRF).

    Ties are broken by the lowest minimum rank across input rankings so
    a doc that ranked #1 anywhere beats a doc with two #4 placements at
    equal score.
    """
    if not rankings:
        return []
    if k_constant <= 0:
        raise ValueError("k_constant must be positive")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for idx, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else float(weights[idx])
        for rank, rid in enumerate(ranking, start=1):
            rid_int = int(rid)
            scores[rid_int] = scores.get(rid_int, 0.0) + weight / (k_constant + rank)
            if rid_int not in best_rank or rank < best_rank[rid_int]:
                best_rank[rid_int] = rank
    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], best_rank.get(kv[0], 1 << 30), kv[0]),
    )
    fused = [rid for rid, _ in ordered]
    if top_k is not None and top_k > 0:
        fused = fused[:top_k]
    return fused


def priority_fill_fusion(
    primary: Sequence[int],
    secondary: Sequence[int],
    *,
    top_k: int | None = None,
) -> list[int]:
    """Keep ``primary``'s order verbatim, then append ``secondary``-only docs.

    This is the ``lexical_first`` fusion: trust the stronger retriever's
    ranking entirely and use the weaker one only to backfill slots the
    primary ranker left empty. De-duplicates by doc id.
    """
    seen: set[int] = set()
    fused: list[int] = []
    for rid in list(primary) + list(secondary):
        rid_int = int(rid)
        if rid_int in seen:
            continue
        seen.add(rid_int)
        fused.append(rid_int)
    if top_k is not None and top_k > 0:
        fused = fused[:top_k]
    return fused


# Metric keys the aggregator averages automatically. ``filter_precision``
# is deliberately excluded — it is averaged separately so the summary can
# also report how many queries actually carried a filter expectation.
_METRIC_KEY_RE = re.compile(
    r"^(?:hit_at_\d+|recall_at_\d+|capped_recall_at_\d+|ndcg_at_\d+"
    r"|mrr_at_\d+|r_precision)$"
)


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate_metrics(
    per_query: Sequence[Mapping[str, float | None]],
    *,
    ks: Sequence[int] = (5, 10, 20),  # kept for backward-compatible callers
    mrr_k: int = 10,
) -> dict[str, float | int]:
    """Average per-query metric dicts into a single summary dict.

    Any key naming a known metric family — ``hit_at_{k}``,
    ``recall_at_{k}``, ``capped_recall_at_{k}``, ``ndcg_at_{k}``,
    ``mrr_at_{k}`` or ``r_precision`` — is averaged over the rows that
    carry a non-None numeric value for it, so adding a metric to the
    per-query rows is enough to surface it here. None values are
    excluded from their average. ``filter_precision_at_{mrr_k}`` is
    handled separately: ``n_queries_with_filter_expectation`` counts the
    queries that contributed to it.
    """
    summary: dict[str, float | int] = {"n_queries": len(per_query)}
    if not per_query:
        return summary

    keys: set[str] = set()
    for row in per_query:
        keys.update(row.keys())

    for key in sorted(keys):
        if not _METRIC_KEY_RE.match(key):
            continue
        vals = [float(row[key]) for row in per_query if _numeric(row.get(key))]
        if vals:
            summary[key] = round(sum(vals) / len(vals), 6)

    fp_key = f"filter_precision_at_{mrr_k}"
    fp_vals = [float(row[fp_key]) for row in per_query if _numeric(row.get(fp_key))]
    summary["n_queries_with_filter_expectation"] = len(fp_vals)
    if fp_vals:
        summary[fp_key] = round(sum(fp_vals) / len(fp_vals), 6)

    return summary


def _payload_view(hit: Any) -> Mapping[str, Any]:
    if isinstance(hit, Mapping):
        return hit
    payload = getattr(hit, "raw_payload", None)
    if isinstance(payload, Mapping):
        return payload
    # Fall back to attribute-style access via a SearchHit-shaped dict.
    return {
        "brand": getattr(hit, "brand", ""),
        "aspect_codes": getattr(hit, "aspect_codes", []),
        "sentiments": getattr(hit, "sentiments", []),
    }


ERROR_GOOD = "good"
ERROR_NO_GOLD_HIT = "no_gold_hit"
ERROR_WRONG_ASPECT = "wrong_aspect"
ERROR_WRONG_SENTIMENT = "wrong_sentiment"
ERROR_LEXICAL_MISS = "lexical_miss"
ERROR_FILTER_TOO_STRICT = "filter_too_strict"


def classify_retrieval_error(
    *,
    ranked: Sequence[int],
    gold: Iterable[int],
    top_hit_payload: Mapping[str, Any] | None,
    expected_aspect: str | None,
    expected_sentiment: str | None,
    retrieval_mode: str,
    pre_filter_ranked: Sequence[int] | None = None,
) -> str:
    """Bucket the most likely root cause of a missed query.

    Order of checks (first hit wins):

    - ``good`` — at least one gold doc in the top-10.
    - ``filter_too_strict`` — applying expected_aspect/sentiment cut
      gold docs that the unfiltered ranking would have surfaced.
    - ``wrong_aspect`` / ``wrong_sentiment`` — the top-1 hit doesn't
      match the expected facet, suggesting the query phrasing fooled
      the retriever about which aspect/sentiment the user meant.
    - ``lexical_miss`` — dense mode in particular: nothing in the
      top-10 even shares vocabulary with a gold quote (proxied here
      by "no gold in top-50" when ``pre_filter_ranked`` is supplied).
    - ``no_gold_hit`` — catch-all fallback.
    """
    gold_set = {int(g) for g in gold}
    head = [int(r) for r in ranked[:10]]
    if any(r in gold_set for r in head):
        return ERROR_GOOD

    if pre_filter_ranked is not None:
        pre_head = [int(r) for r in pre_filter_ranked[:10]]
        if any(r in gold_set for r in pre_head):
            return ERROR_FILTER_TOO_STRICT

    if top_hit_payload is not None:
        top_aspects = list(top_hit_payload.get("aspect_codes") or [])
        top_sentiments = list(top_hit_payload.get("sentiments") or [])
        if expected_aspect is not None and expected_aspect not in top_aspects:
            return ERROR_WRONG_ASPECT
        if expected_sentiment is not None and expected_sentiment not in top_sentiments:
            return ERROR_WRONG_SENTIMENT

    if retrieval_mode == "dense" and pre_filter_ranked is not None:
        wide = [int(r) for r in pre_filter_ranked[:50]]
        if not any(r in gold_set for r in wide):
            return ERROR_LEXICAL_MISS

    return ERROR_NO_GOLD_HIT
