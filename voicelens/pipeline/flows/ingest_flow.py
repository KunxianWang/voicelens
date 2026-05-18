from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger
from sqlalchemy.orm import Session

from voicelens.config import SAMPLE_REVIEWS_PATH
from voicelens.db.engine import get_session
from voicelens.db.models import DQEvent, IngestRun
from voicelens.pipeline.tasks.dq import ALL_CHECKS, DQOutcome, run_dq
from voicelens.pipeline.tasks.normalize import load_reviews


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def open_ingest_run(session: Session, source: str) -> IngestRun:
    run = IngestRun(source=source, status="running", started_at=datetime.utcnow())
    session.add(run)
    session.flush()
    return run


def write_dq_events(session: Session, run_id: int, outcome: DQOutcome) -> None:
    for check in ALL_CHECKS:
        n_failed = int(outcome.per_check_failed.get(check, 0))
        reason_codes = None
        if n_failed > 0:
            reason_codes = {
                "examples": [
                    {
                        "source": r.get("source"),
                        "source_id": r.get("source_id"),
                        "asin": r.get("asin"),
                    }
                    for r in outcome.rejected.get(check, [])[:5]
                ]
            }
        session.add(
            DQEvent(
                ingest_run_id=run_id,
                check_name=check,
                n_rows=outcome.total_rows,
                n_failed=n_failed,
                reason_codes=reason_codes,
            )
        )
    session.flush()


def close_ingest_run(session: Session, run: IngestRun, n_rows: int, status: str) -> None:
    run.completed_at = datetime.utcnow()
    run.n_rows = n_rows
    run.status = status
    session.flush()


@flow(name="ingest_flow")
def ingest_flow(path: str | Path | None = None, source: str = "amazon_reviews_2023") -> dict[str, Any]:
    logger = get_run_logger()
    input_path = Path(path) if path else SAMPLE_REVIEWS_PATH
    logger.info(f"Reading reviews from {input_path}")

    rows = read_jsonl(input_path)
    outcome = run_dq(rows)
    logger.info(
        f"DQ summary: total={outcome.total_rows}, passed={outcome.passed_rows}, "
        f"failed={outcome.failed_rows}, pass_rate={outcome.pass_rate:.3f}"
    )

    with get_session() as session:
        run = open_ingest_run(session, source)
        run_id = run.id
        loaded = load_reviews(session, outcome.valid_rows, ingest_run_id=run_id)
        write_dq_events(session, run_id, outcome)
        close_ingest_run(session, run, n_rows=loaded, status="completed")

    summary = {
        "total_rows": outcome.total_rows,
        "passed_rows": outcome.passed_rows,
        "failed_rows": outcome.failed_rows,
        "dq_pass_rate": round(outcome.pass_rate, 4),
        "loaded_reviews": loaded,
        "per_check_failed": dict(outcome.per_check_failed),
    }
    logger.info(f"Ingest summary: {summary}")
    return summary


if __name__ == "__main__":
    result = ingest_flow()
    print(json.dumps(result, indent=2, sort_keys=True))
