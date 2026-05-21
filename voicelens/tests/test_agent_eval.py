"""Tests for the M6C agent-workflow evaluation (voicelens.eval.agent_eval).

All pure / injected — no agent run, no DB, no LLM.
"""
from __future__ import annotations

import json

from scripts.evaluate_agent import run_evaluation, write_results
from voicelens.config import PROJECT_ROOT
from voicelens.eval.agent_eval import (
    AgentGolden,
    aggregate_metrics,
    load_goldens,
    score_record,
)

_GOLDENS_FILE = PROJECT_ROOT / "data" / "eval" / "agent_smoke_goldens.jsonl"


def _summary(
    route: str,
    *,
    answer: str = "an answer",
    n_citations: int = 0,
    warnings: list[str] | None = None,
) -> dict:
    """A minimal ``result_summary``-shaped dict for scoring."""
    return {
        "route": route,
        "answer": answer,
        "citations": [{"citation_id": i + 1} for i in range(n_citations)],
        "warnings": list(warnings or []),
        "filters": {},
        "retrieved_review_ids": [],
    }


# ---- score_record -------------------------------------------------------


def test_score_record_route_match_and_mismatch():
    golden = AgentGolden(question="q", expected_route="analytics_summary")
    assert score_record(golden, _summary("analytics_summary"))["route_match"] is True
    mismatch = score_record(golden, _summary("retrieval_answer"))
    assert mismatch["route_match"] is False
    assert mismatch["passed"] is False


def test_score_record_citation_requirement_detects_missing():
    golden = AgentGolden(
        question="q", expected_route="retrieval_answer",
        expected_citation_required=True, expected_min_citations=2,
    )
    missing = score_record(golden, _summary("retrieval_answer", n_citations=0))
    assert missing["citation_ok"] is False
    assert missing["passed"] is False

    enough = score_record(golden, _summary("retrieval_answer", n_citations=3))
    assert enough["citation_ok"] is True
    assert enough["passed"] is True


def test_score_record_contains_check():
    golden = AgentGolden(
        question="q", expected_route="analytics_summary",
        expected_contains=["distribution"],
    )
    ok = score_record(golden, _summary("analytics_summary", answer="aspect distribution: ..."))
    assert ok["contains_ok"] is True
    bad = score_record(golden, _summary("analytics_summary", answer="no stats here"))
    assert bad["contains_ok"] is False
    assert bad["passed"] is False


def test_score_record_no_citation_required_passes_without_citations():
    golden = AgentGolden(question="q", expected_route="incident_summary")
    scored = score_record(golden, _summary("incident_summary", n_citations=0))
    assert scored["citation_ok"] is True
    assert scored["passed"] is True


# ---- aggregate_metrics --------------------------------------------------


def test_aggregate_metrics_computes_rates():
    goldens = [
        AgentGolden("r1", "retrieval_answer", expected_citation_required=True),
        AgentGolden("r2", "retrieval_answer", expected_citation_required=True),
        AgentGolden("a1", "analytics_summary"),
        AgentGolden("i1", "incident_summary"),
        AgentGolden("u1", "insufficient_scope"),
    ]
    summaries = [
        _summary("retrieval_answer", n_citations=4),               # route ok
        _summary("analytics_summary", n_citations=0),              # route WRONG
        _summary("analytics_summary", answer="distribution ..."),  # route ok
        _summary("incident_summary", answer="incident ...", warnings=["w"]),
        _summary("insufficient_scope", answer="can't answer ..."),
    ]
    scored = [score_record(g, s) for g, s in zip(goldens, summaries, strict=True)]
    metrics = aggregate_metrics(scored)

    assert metrics["n_questions"] == 5
    # 4 of 5 routes correct (r2 retrieved-as-analytics is wrong)
    assert metrics["route_accuracy"] == 0.8
    # only r1 produced citations -> 1 of 2 retrieval goldens
    assert metrics["retrieval_citation_rate"] == 0.5
    assert metrics["average_citations_for_retrieval"] == 2.0  # (4 + 0) / 2
    assert metrics["unsupported_refusal_rate"] == 1.0
    assert metrics["analytics_answer_nonempty_rate"] == 1.0
    assert metrics["incident_answer_nonempty_rate"] == 1.0
    assert metrics["warning_rate"] == 0.2  # 1 of 5


def test_aggregate_metrics_unsupported_refusal_detects_leak():
    goldens = [
        AgentGolden("u1", "insufficient_scope"),
        AgentGolden("u2", "insufficient_scope"),
    ]
    summaries = [
        _summary("insufficient_scope", answer="can't answer"),
        _summary("retrieval_answer", answer="leaked an answer"),  # NOT refused
    ]
    scored = [score_record(g, s) for g, s in zip(goldens, summaries, strict=True)]
    metrics = aggregate_metrics(scored)
    assert metrics["unsupported_refusal_rate"] == 0.5


# ---- load_goldens -------------------------------------------------------


def test_load_goldens_reads_the_shipped_golden_set():
    goldens = load_goldens(_GOLDENS_FILE)
    assert len(goldens) == 20
    routes = [g.expected_route for g in goldens]
    assert routes.count("retrieval_answer") == 6
    assert routes.count("analytics_summary") == 5
    assert routes.count("incident_summary") == 5
    assert routes.count("insufficient_scope") == 4
    # retrieval goldens require citations
    for g in goldens:
        if g.expected_route == "retrieval_answer":
            assert g.expected_citation_required is True


def test_load_goldens_from_tmp_file(tmp_path):
    path = tmp_path / "tiny.jsonl"
    path.write_text(
        json.dumps({"question": "q1", "expected_route": "analytics_summary"}) + "\n",
        encoding="utf-8",
    )
    goldens = load_goldens(path)
    assert len(goldens) == 1
    assert goldens[0].expected_route == "analytics_summary"
    assert goldens[0].expected_min_citations == 1  # default


# ---- evaluate_agent runner ----------------------------------------------


def test_run_evaluation_with_injected_summary_fn(tmp_path):
    """evaluate_agent works end-to-end on a tiny fake golden set."""
    goldens = [
        AgentGolden(
            "reliability complaints", "retrieval_answer",
            expected_citation_required=True, expected_min_citations=1,
        ),
        AgentGolden(
            "aspect distribution", "analytics_summary",
            expected_contains=["distribution"],
        ),
        AgentGolden("buy stock?", "insufficient_scope"),
    ]
    canned = {
        "reliability complaints": _summary("retrieval_answer", n_citations=5),
        "aspect distribution": _summary(
            "analytics_summary", answer="aspect distribution: charging=10"
        ),
        "buy stock?": _summary("insufficient_scope", answer="can't answer that"),
    }
    scored, metrics = run_evaluation(
        goldens, summary_fn=lambda q: canned[q]
    )
    assert metrics["route_accuracy"] == 1.0
    assert metrics["pass_rate"] == 1.0
    assert metrics["retrieval_citation_rate"] == 1.0

    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "summary.json"
    write_results(scored, metrics, csv_path=csv_path, json_path=json_path)
    assert csv_path.exists() and json_path.exists()
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["route_accuracy"] == 1.0
    assert "question" in csv_path.read_text(encoding="utf-8").splitlines()[0]
