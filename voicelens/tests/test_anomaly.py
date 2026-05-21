"""Pure anomaly-detection tests (no DB)."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from voicelens.analytics.anomaly import (
    AnomalyResult,
    MentionEvent,
    aggregate_weekly,
    build_incident_summary,
    detect_anomalies,
    ewma_baseline,
    severity_weight,
    week_start,
    z_score,
)

_MONDAY = date(2024, 1, 1)  # a Monday


def _wk(i: int) -> date:
    """Monday of week ``i`` (0-based) from the fixed base."""
    return date.fromordinal(_MONDAY.toordinal() + 7 * i)


def _event(
    *,
    week: int,
    review_id: int,
    cluster_id: int = 1,
    aspect: str = "charging",
    severity: str | None = "medium",
    rating: int = 1,
    quote: str = "charging port failed",
) -> MentionEvent:
    return MentionEvent(
        review_id=review_id,
        cluster_id=cluster_id,
        aspect_code=aspect,
        severity=severity,
        posted_at=datetime.combine(_wk(week), datetime.min.time()),
        rating=rating,
        evidence_quote=quote,
    )


# ---- severity weights ---------------------------------------------------


def test_severity_weights_are_steeper_for_m4b():
    assert severity_weight("low") == 1.0
    assert severity_weight("medium") == 2.0
    assert severity_weight("high") == 4.0


def test_severity_weight_defaults_for_unknown():
    assert severity_weight(None) == 1.0
    assert severity_weight("bogus") == 1.0


def test_week_start_is_monday():
    assert week_start(date(2024, 1, 3)) == date(2024, 1, 1)  # Wed -> Mon
    assert week_start(datetime(2024, 1, 7, 23, 0)) == date(2024, 1, 1)  # Sun -> Mon


# ---- weekly aggregation -------------------------------------------------


def test_aggregate_weekly_groups_by_series_and_week():
    events = [
        _event(week=0, review_id=1),
        _event(week=0, review_id=2),
        _event(week=1, review_id=3),
    ]
    buckets = aggregate_weekly(events, granularity="cluster")
    series = buckets["cluster:1"]
    assert [b.week_start for b in series] == [_wk(0), _wk(1)]
    assert series[0].observed_volume == 2
    assert series[1].observed_volume == 1


def test_aggregate_weekly_severity_weighted_volume():
    events = [
        _event(week=0, review_id=1, severity="high"),    # 4
        _event(week=0, review_id=2, severity="medium"),  # 2
        _event(week=0, review_id=3, severity="low"),     # 1
    ]
    buckets = aggregate_weekly(events, granularity="cluster")
    assert buckets["cluster:1"][0].severity_weighted_volume == 7.0


def test_aggregate_weekly_unique_reviews_and_avg_rating():
    events = [
        _event(week=0, review_id=1, rating=1),
        _event(week=0, review_id=1, rating=1),  # same review twice
        _event(week=0, review_id=2, rating=3),
    ]
    bucket = aggregate_weekly(events, granularity="cluster")["cluster:1"][0]
    assert bucket.observed_volume == 3
    assert bucket.unique_review_count == 2
    assert bucket.avg_rating == pytest.approx((1 + 1 + 3) / 3, abs=1e-3)


def test_aggregate_weekly_aspect_granularity_keys_by_aspect():
    events = [
        _event(week=0, review_id=1, cluster_id=1, aspect="charging"),
        _event(week=0, review_id=2, cluster_id=2, aspect="charging"),
    ]
    buckets = aggregate_weekly(events, granularity="aspect")
    assert set(buckets) == {"aspect:charging"}
    assert buckets["aspect:charging"][0].observed_volume == 2


def test_aggregate_weekly_cluster_granularity_skips_null_cluster():
    events = [_event(week=0, review_id=1, cluster_id=None)]
    assert aggregate_weekly(events, granularity="cluster") == {}


# ---- EWMA ---------------------------------------------------------------


def test_ewma_baseline_none_for_empty_history():
    assert ewma_baseline([], span=4) is None


def test_ewma_baseline_flat_series_equals_value():
    assert ewma_baseline([5.0, 5.0, 5.0], span=4) == pytest.approx(5.0)


def test_ewma_baseline_recent_weeks_dominate():
    rising = ewma_baseline([1, 1, 1, 10], span=4)
    # last week (10) pulls the estimate well above the early baseline of 1
    assert rising > 3.0


# ---- z-score ------------------------------------------------------------


def test_z_score_uses_min_std_floor():
    # std below the floor -> floor is used, no division blow-up
    assert z_score(10, 2, 0.0, min_std=1.0) == pytest.approx(8.0)
    assert z_score(10, 2, 0.1, min_std=1.0) == pytest.approx(8.0)


def test_z_score_uses_real_std_when_above_floor():
    assert z_score(10, 2, 4.0, min_std=1.0) == pytest.approx(2.0)


# ---- detect_anomalies ---------------------------------------------------


def _spike_series(spike_volume: int, *, severity: str = "medium") -> list[MentionEvent]:
    """Five quiet weeks of 2 mentions, then a spike week."""
    events: list[MentionEvent] = []
    rid = 0
    for week in range(5):
        for _ in range(2):
            rid += 1
            events.append(_event(week=week, review_id=rid, severity=severity))
    for _ in range(spike_volume):
        rid += 1
        events.append(_event(week=5, review_id=rid, severity=severity))
    return events


def test_detect_anomalies_flags_a_clear_spike():
    buckets = aggregate_weekly(_spike_series(12), granularity="cluster")
    incidents = detect_anomalies(buckets, min_history_weeks=3, z_threshold=2.0)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.week_start == _wk(5)
    assert inc.observed_volume == 12
    assert inc.z_score >= 2.0
    # baseline reflects only the 5 prior weeks of volume 2 -> ~2.0
    assert inc.baseline_volume == pytest.approx(2.0, abs=0.5)


def test_detect_anomalies_baseline_excludes_current_week():
    buckets = aggregate_weekly(_spike_series(12), granularity="cluster")
    incidents = detect_anomalies(buckets, min_history_weeks=3, z_threshold=2.0)
    # if the spike week were in its own baseline, baseline would be >> 2
    assert incidents[0].baseline_volume < 5.0


def test_detect_anomalies_min_history_blocks_early_weeks():
    # only 2 prior weeks before the spike at week 2 -> below min_history 3
    events = [
        _event(week=0, review_id=1),
        _event(week=1, review_id=2),
        *[_event(week=2, review_id=10 + i) for i in range(15)],
    ]
    buckets = aggregate_weekly(events, granularity="cluster")
    assert detect_anomalies(buckets, min_history_weeks=3, z_threshold=2.0) == []


def test_detect_anomalies_min_volume_suppresses_small_spike():
    """A 5-mention spike has a high z-score but is suppressed by min_volume."""
    buckets = aggregate_weekly(_spike_series(5, severity="high"), granularity="cluster")
    # min_volume 3 -> the spike (5) is flagged
    flagged = detect_anomalies(buckets, min_history_weeks=3, z_threshold=2.0, min_volume=3)
    assert len(flagged) == 1
    # min_volume 10 -> the same 5-mention spike is now suppressed
    suppressed = detect_anomalies(
        buckets, min_history_weeks=3, z_threshold=2.0, min_volume=10
    )
    assert suppressed == []


def test_detect_anomalies_min_severity_suppresses_low_severity_spike():
    # 6-mention spike, all "low" severity -> severity score 6
    buckets = aggregate_weekly(_spike_series(6, severity="low"), granularity="cluster")
    assert detect_anomalies(
        buckets, min_history_weeks=3, z_threshold=2.0,
        min_volume=3, min_severity_score=20.0,
    ) == []


def test_detect_anomalies_no_spike_in_flat_series():
    events = []
    rid = 0
    for week in range(8):
        for _ in range(3):
            rid += 1
            events.append(_event(week=week, review_id=rid))
    buckets = aggregate_weekly(events, granularity="cluster")
    assert detect_anomalies(buckets, min_history_weeks=3, z_threshold=2.0) == []


# ---- summary ------------------------------------------------------------


def test_build_incident_summary_cluster_level():
    result = AnomalyResult(
        series_key="cluster:1", granularity="cluster", cluster_id=1,
        aspect_code="reliability", week_start=date(2024, 3, 18),
        observed_volume=12, baseline_volume=3.1, rolling_std=2.0,
        z_score=2.8, severity_score=24.0, unique_review_count=12,
        avg_rating=1.2, example_quotes=["Stopped working after three weeks."],
    )
    summary = build_incident_summary(result, cluster_label="stopped working week")
    assert "Reliability complaints" in summary
    assert "stopped working week" in summary
    assert "12 mentions" in summary
    assert "2024-03-18" in summary
    assert "z=2.8" in summary
    assert "Stopped working after three weeks." in summary


def test_build_incident_summary_aspect_level_has_no_cluster_label():
    result = AnomalyResult(
        series_key="aspect:charging", granularity="aspect", cluster_id=None,
        aspect_code="charging", week_start=date(2024, 3, 18),
        observed_volume=9, baseline_volume=2.0, rolling_std=1.5,
        z_score=3.0, severity_score=18.0, unique_review_count=9,
        avg_rating=1.5, example_quotes=[],
    )
    summary = build_incident_summary(result, cluster_label=None)
    assert "overall" in summary
    assert "Charging complaints" in summary
