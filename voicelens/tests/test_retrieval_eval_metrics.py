from __future__ import annotations

import pytest

from voicelens.eval.retrieval_eval import (
    ERROR_FILTER_TOO_STRICT,
    ERROR_GOOD,
    ERROR_LEXICAL_MISS,
    ERROR_NO_GOLD_HIT,
    ERROR_WRONG_ASPECT,
    ERROR_WRONG_SENTIMENT,
    aggregate_metrics,
    capped_recall_at_k,
    classify_retrieval_error,
    dcg_at_k,
    filter_precision_at_k,
    hit_at_k,
    ndcg_at_k,
    priority_fill_fusion,
    r_precision,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)

# ---- recall@k -----------------------------------------------------------


def test_recall_at_k_hits_two_of_three():
    ranked = [10, 11, 12, 13, 14]
    gold = [10, 12, 99]
    assert recall_at_k(ranked, gold, 5) == pytest.approx(2 / 3, rel=1e-6)


def test_recall_at_k_zero_when_no_hits_in_head():
    ranked = [1, 2, 3]
    assert recall_at_k(ranked, [4, 5], 3) == 0.0


def test_recall_at_k_handles_empty_gold():
    assert recall_at_k([1, 2, 3], [], 5) == 0.0


def test_recall_at_k_rejects_nonpositive_k():
    assert recall_at_k([1, 2], [1], 0) == 0.0


# ---- reciprocal rank ----------------------------------------------------


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([7, 8, 9], [7], 10) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([1, 2, 7], [7], 10) == pytest.approx(1 / 3, rel=1e-6)


def test_reciprocal_rank_no_hit_within_k():
    assert reciprocal_rank([1, 2, 3], [7], 3) == 0.0


def test_reciprocal_rank_no_hit_at_all():
    assert reciprocal_rank([1, 2, 3], [7], 10) == 0.0


# ---- NDCG ---------------------------------------------------------------


def test_dcg_at_k_perfect_ranking():
    """Perfect ranking gain matches the ideal DCG."""
    ranked = [1, 2, 3]
    gold = [1, 2, 3]
    assert dcg_at_k(ranked, gold, 3) == pytest.approx(
        1.0 / 1 + 1.0 / 1.5849625007211563 + 1.0 / 2.0
    )


def test_ndcg_at_k_perfect_is_one():
    assert ndcg_at_k([1, 2, 3], [1, 2, 3], 3) == 1.0


def test_ndcg_at_k_zero_when_no_gold():
    assert ndcg_at_k([1, 2, 3], [], 3) == 0.0


def test_ndcg_at_k_partial_hits_below_one():
    score = ndcg_at_k([1, 2, 3], [1, 99], 3)
    assert 0 < score < 1


# ---- filter precision ---------------------------------------------------


def test_filter_precision_returns_none_without_expectation():
    hits = [{"brand": "Anker", "aspect_codes": ["battery"], "sentiments": ["negative"]}]
    assert filter_precision_at_k(hits) is None


def test_filter_precision_counts_aspect_match():
    hits = [
        {"aspect_codes": ["battery"], "sentiments": ["negative"], "brand": "Anker"},
        {"aspect_codes": ["charging"], "sentiments": ["negative"], "brand": "Anker"},
        {"aspect_codes": ["battery"], "sentiments": ["positive"], "brand": "Anker"},
    ]
    # 2/3 hits contain the expected aspect.
    assert filter_precision_at_k(hits, expected_aspect="battery", k=3) == pytest.approx(2 / 3)


def test_filter_precision_combines_predicates():
    hits = [
        {"aspect_codes": ["battery"], "sentiments": ["negative"], "brand": "Anker"},
        {"aspect_codes": ["battery"], "sentiments": ["positive"], "brand": "Anker"},
        {"aspect_codes": ["battery"], "sentiments": ["negative"], "brand": "Bose"},
    ]
    # Aspect battery AND sentiment negative AND brand Anker => only hit 1.
    p = filter_precision_at_k(
        hits,
        expected_aspect="battery",
        expected_sentiment="negative",
        expected_brand="Anker",
        k=3,
    )
    assert p == pytest.approx(1 / 3)


# ---- RRF fusion ---------------------------------------------------------


def test_rrf_single_ranking_is_pass_through():
    fused = reciprocal_rank_fusion([[10, 11, 12]])
    assert fused == [10, 11, 12]


def test_rrf_two_rankings_top_overlap_wins():
    dense = [10, 20, 30]
    lexical = [20, 10, 40]
    fused = reciprocal_rank_fusion([dense, lexical], top_k=3)
    # 20 ranks 2 and 1 -> highest fused score
    # 10 ranks 1 and 2 -> tied with 20 by score? recompute:
    #  20: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.03259
    #  10: 1/(60+1) + 1/(60+2) = same                ~ 0.03259
    # tie-break = min rank: both have min 1 -> id tie-break = 10 first.
    assert fused[:2] == [10, 20]
    assert 30 in fused or 40 in fused


def test_rrf_doc_only_in_one_ranking_still_fuses():
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]], top_k=4)
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_top_k_trims():
    fused = reciprocal_rank_fusion([[1, 2, 3, 4, 5]], top_k=2)
    assert fused == [1, 2]


def test_rrf_rejects_nonpositive_k_constant():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1, 2]], k_constant=0)


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []


# ---- error classifier ---------------------------------------------------


def test_classify_good_when_top10_contains_gold():
    error = classify_retrieval_error(
        ranked=[1, 2, 3, 7],
        gold=[7],
        top_hit_payload={"aspect_codes": ["battery"], "sentiments": ["negative"]},
        expected_aspect="battery",
        expected_sentiment="negative",
        retrieval_mode="dense",
    )
    assert error == ERROR_GOOD


def test_classify_filter_too_strict_when_pre_filter_had_gold():
    error = classify_retrieval_error(
        ranked=[1, 2, 3],
        gold=[7],
        top_hit_payload={"aspect_codes": ["battery"], "sentiments": ["negative"]},
        expected_aspect="battery",
        expected_sentiment="negative",
        retrieval_mode="dense",
        pre_filter_ranked=[7, 8, 9],
    )
    assert error == ERROR_FILTER_TOO_STRICT


def test_classify_wrong_aspect():
    error = classify_retrieval_error(
        ranked=[1, 2, 3],
        gold=[7],
        top_hit_payload={"aspect_codes": ["price"], "sentiments": ["negative"]},
        expected_aspect="battery",
        expected_sentiment="negative",
        retrieval_mode="dense",
        pre_filter_ranked=[1, 2, 3],
    )
    assert error == ERROR_WRONG_ASPECT


def test_classify_wrong_sentiment_only_when_aspect_matches():
    error = classify_retrieval_error(
        ranked=[1, 2, 3],
        gold=[7],
        top_hit_payload={"aspect_codes": ["battery"], "sentiments": ["positive"]},
        expected_aspect="battery",
        expected_sentiment="negative",
        retrieval_mode="dense",
        pre_filter_ranked=[1, 2, 3],
    )
    assert error == ERROR_WRONG_SENTIMENT


def test_classify_lexical_miss_in_dense_mode():
    # gold=[7] absent from both filtered and unfiltered top-50 -> the
    # query embedding never even retrieved it. In dense mode that's a
    # vocabulary-mismatch signal (the lexical retriever might have hit).
    error = classify_retrieval_error(
        ranked=[1, 2, 3],
        gold=[7],
        top_hit_payload={"aspect_codes": ["battery"], "sentiments": ["negative"]},
        expected_aspect="battery",
        expected_sentiment="negative",
        retrieval_mode="dense",
        pre_filter_ranked=list(range(100, 200)),
    )
    assert error == ERROR_LEXICAL_MISS


def test_classify_no_gold_hit_fallback():
    error = classify_retrieval_error(
        ranked=[1, 2, 3],
        gold=[7],
        top_hit_payload={"aspect_codes": ["battery"], "sentiments": ["negative"]},
        expected_aspect=None,
        expected_sentiment=None,
        retrieval_mode="lexical",
    )
    assert error == ERROR_NO_GOLD_HIT


# ---- aggregator ---------------------------------------------------------


def test_aggregate_metrics_averages_only_non_none_values():
    per_query = [
        {"recall_at_5": 1.0, "recall_at_10": 1.0, "recall_at_20": 1.0,
         "mrr_at_10": 1.0, "filter_precision_at_10": 1.0,
         "ndcg_at_5": 1.0, "ndcg_at_10": 1.0, "ndcg_at_20": 1.0},
        {"recall_at_5": 0.0, "recall_at_10": 0.5, "recall_at_20": 1.0,
         "mrr_at_10": 0.5, "filter_precision_at_10": None,
         "ndcg_at_5": 0.0, "ndcg_at_10": 0.5, "ndcg_at_20": 0.7},
        {"recall_at_5": None, "recall_at_10": None, "recall_at_20": None,
         "mrr_at_10": None, "filter_precision_at_10": None},
    ]
    summary = aggregate_metrics(per_query)
    assert summary["n_queries"] == 3
    assert summary["recall_at_5"] == pytest.approx(0.5)
    assert summary["recall_at_10"] == pytest.approx(0.75)
    assert summary["mrr_at_10"] == pytest.approx(0.75)
    assert summary["n_queries_with_filter_expectation"] == 1
    assert summary["filter_precision_at_10"] == pytest.approx(1.0)


def test_aggregate_metrics_handles_empty():
    assert aggregate_metrics([])["n_queries"] == 0


def test_aggregate_metrics_averages_new_metric_families():
    """Hit@k / capped_recall / r_precision are picked up automatically."""
    per_query = [
        {"hit_at_5": 1.0, "capped_recall_at_5": 1.0, "r_precision": 1.0},
        {"hit_at_5": 0.0, "capped_recall_at_5": 0.5, "r_precision": 0.0},
    ]
    summary = aggregate_metrics(per_query)
    assert summary["hit_at_5"] == pytest.approx(0.5)
    assert summary["capped_recall_at_5"] == pytest.approx(0.75)
    assert summary["r_precision"] == pytest.approx(0.5)


# ---- Hit@k --------------------------------------------------------------


def test_hit_at_k_is_one_when_any_gold_in_head():
    assert hit_at_k([1, 2, 3, 7], [7, 99], 5) == 1.0


def test_hit_at_k_is_zero_when_gold_below_k():
    assert hit_at_k([1, 2, 3, 7], [7], 3) == 0.0


def test_hit_at_k_zero_for_empty_gold():
    assert hit_at_k([1, 2, 3], [], 5) == 0.0


# ---- capped recall ------------------------------------------------------


def test_capped_recall_caps_denominator_for_high_gold():
    # 8 gold docs, 3 land in top-5: plain recall = 3/8, capped = 3/5.
    ranked = [1, 2, 3, 98, 99]
    gold = [1, 2, 3, 4, 5, 6, 7, 8]
    assert recall_at_k(ranked, gold, 5) == pytest.approx(3 / 8)
    assert capped_recall_at_k(ranked, gold, 5) == pytest.approx(3 / 5)


def test_capped_recall_matches_recall_when_gold_below_k():
    # 2 gold, both retrieved -> 2/min(2,5) = 1.0.
    assert capped_recall_at_k([1, 2, 9], [1, 2], 5) == pytest.approx(1.0)


def test_capped_recall_zero_for_empty_gold():
    assert capped_recall_at_k([1, 2], [], 5) == 0.0


# ---- R-precision --------------------------------------------------------


def test_r_precision_uses_gold_count_as_cutoff():
    # R=3: top-3 is [1, 2, 99] -> 2 of 3 correct.
    assert r_precision([1, 2, 99, 3], [1, 2, 3]) == pytest.approx(2 / 3)


def test_r_precision_perfect_is_one():
    assert r_precision([5, 6, 7, 8], [5, 6, 7]) == 1.0


def test_r_precision_zero_for_empty_gold():
    assert r_precision([1, 2, 3], []) == 0.0


# ---- weighted RRF -------------------------------------------------------


def test_weighted_rrf_favours_heavily_weighted_ranker():
    dense = [1, 2]      # ranker 0
    lexical = [3, 4]    # ranker 1
    # lexical gets 9x the weight -> its rank-1 doc (3) must beat dense's (1).
    fused = reciprocal_rank_fusion(
        [dense, lexical], weights=[0.1, 0.9], top_k=4
    )
    assert fused[0] == 3
    assert fused.index(3) < fused.index(1)


def test_weighted_rrf_rejects_weight_length_mismatch():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1, 2], [3, 4]], weights=[1.0])


def test_equal_weights_match_unweighted_rrf():
    rankings = [[10, 20, 30], [20, 10, 40]]
    assert reciprocal_rank_fusion(rankings, weights=[1.0, 1.0]) == (
        reciprocal_rank_fusion(rankings)
    )


# ---- priority-fill fusion (lexical_first) -------------------------------


def test_priority_fill_keeps_primary_order_verbatim():
    fused = priority_fill_fusion([3, 1, 2], [9, 8])
    assert fused == [3, 1, 2, 9, 8]


def test_priority_fill_dedupes_secondary_overlap():
    fused = priority_fill_fusion([5, 6], [6, 7, 8])
    assert fused == [5, 6, 7, 8]


def test_priority_fill_trims_to_top_k():
    fused = priority_fill_fusion([1, 2, 3], [4, 5], top_k=2)
    assert fused == [1, 2]
