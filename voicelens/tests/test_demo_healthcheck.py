"""Light tests for the demo healthcheck helpers (scripts/demo_healthcheck.py).

The Postgres / Qdrant checks need live infra and are exercised by
running ``make demo-healthcheck``; here we only cover the pure helpers
and the streamlit-import check.
"""
from __future__ import annotations

from scripts.demo_healthcheck import Check, check_dashboard_import, summarize


def test_summarize_all_pass_is_ready():
    checks = [
        Check("Postgres", True, "reachable, 100 reviews"),
        Check("Qdrant", True, "1,000 points"),
    ]
    all_ok, text = summarize(checks)
    assert all_ok is True
    assert "READY TO DEMO" in text
    assert "[PASS] Postgres" in text


def test_summarize_any_fail_is_not_ready():
    checks = [
        Check("Postgres", True, "ok"),
        Check("Qdrant", False, "collection missing"),
    ]
    all_ok, text = summarize(checks)
    assert all_ok is False
    assert "NOT READY" in text
    assert "[FAIL] Qdrant" in text


def test_summarize_empty_check_list_is_vacuously_ready():
    all_ok, text = summarize([])
    assert all_ok is True
    assert "VoiceLens demo healthcheck" in text


def test_check_dashboard_import_succeeds():
    """The Streamlit app module must import cleanly for the demo."""
    result = check_dashboard_import()
    assert result.ok is True
    assert "voicelens.ui.app" in result.detail
