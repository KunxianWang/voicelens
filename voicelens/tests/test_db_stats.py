from __future__ import annotations

from voicelens.config import AMAZON_FIXTURE_PATH, BRAND_ALLOWLIST


def test_db_stats_on_empty_db(fresh_engine, capsys):
    from scripts.db_stats import main as db_stats_main

    rc = db_stats_main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "total_reviews : 0" in out
    assert "no runs" in out or "latest ingest_run" in out


def test_db_stats_after_ingest(fresh_engine, capsys):
    from scripts.db_stats import collect_stats
    from voicelens.pipeline.flows.amazon_ingest_flow import amazon_ingest_flow

    amazon_ingest_flow(input_path=AMAZON_FIXTURE_PATH, brands=BRAND_ALLOWLIST)

    stats = collect_stats()

    assert stats["total_reviews"] == 12
    assert stats["total_brands"] >= 6
    assert stats["total_skus"] >= 7

    assert "Anker" in stats["reviews_by_brand"]
    assert "Soundcore" in stats["reviews_by_brand"]

    assert sum(stats["reviews_by_rating"].values()) == 12

    assert stats["dq_failures_by_check"].get("duplicate_review_id", 0) == 2

    latest = stats["latest_ingest_run"]
    assert latest is not None
    assert latest["source"] == "amazon_reviews_2023"
    assert latest["status"] == "completed"
    assert latest["n_rows"] == 12
