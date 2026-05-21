"""Demo readiness healthcheck for VoiceLens.

Run this right before a recruiter / interviewer demo to confirm the
moving parts are up: Postgres is reachable and populated, the Qdrant
``reviews_v2`` collection has points, and the Streamlit dashboard
imports cleanly. It is read-only and never calls an LLM.

    python scripts/demo_healthcheck.py     # or: make demo-healthcheck

Exit code 0 = ready to demo, 1 = something needs attention.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from voicelens.config import QDRANT_COLLECTION, QDRANT_URL

# Tables that must hold rows for every dashboard page to render with data.
REQUIRED_TABLES = (
    "review",
    "aspect_mention",
    "absa_review_status",
    "cluster",
    "review_cluster",
    "incident",
    "dq_event",
    "ingest_run",
)


@dataclass
class Check:
    """One healthcheck line: a name, a pass/fail, and a short detail."""

    name: str
    ok: bool
    detail: str


def check_postgres() -> Check:
    """Confirm Postgres is reachable and the review table is non-empty."""
    try:
        from sqlalchemy import func, select

        from voicelens.db.engine import get_session
        from voicelens.db.models import Review

        with get_session() as s:
            n = int(s.scalar(select(func.count()).select_from(Review)) or 0)
        if n == 0:
            return Check("Postgres", False, "reachable but 'review' table is empty")
        return Check("Postgres", True, f"reachable, {n:,} reviews")
    except Exception as exc:  # noqa: BLE001 - report any failure as a check
        return Check("Postgres", False, f"unreachable: {exc}")


def check_tables() -> list[Check]:
    """Row-count every table the dashboard depends on."""
    try:
        from sqlalchemy import func, select, table

        from voicelens.db.engine import get_session

        checks: list[Check] = []
        with get_session() as s:
            for name in REQUIRED_TABLES:
                n = int(s.scalar(select(func.count()).select_from(table(name))) or 0)
                checks.append(
                    Check(f"table:{name}", n > 0, f"{n:,} rows")
                )
        return checks
    except Exception as exc:  # noqa: BLE001
        return [Check("tables", False, f"could not count tables: {exc}")]


def check_qdrant() -> Check:
    """Confirm the Qdrant retrieval collection exists and has points."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(QDRANT_COLLECTION):
            return Check(
                "Qdrant", False, f"collection {QDRANT_COLLECTION!r} does not exist"
            )
        count = int(client.count(QDRANT_COLLECTION).count)
        if count == 0:
            return Check("Qdrant", False, f"{QDRANT_COLLECTION!r} has 0 points")
        return Check("Qdrant", True, f"{QDRANT_COLLECTION!r}: {count:,} points")
    except Exception as exc:  # noqa: BLE001
        return Check("Qdrant", False, f"unreachable: {exc}")


def check_dashboard_import() -> Check:
    """Confirm the Streamlit dashboard module imports without error."""
    try:
        import importlib

        importlib.import_module("voicelens.ui.app")
        return Check("Dashboard import", True, "voicelens.ui.app imports OK")
    except Exception as exc:  # noqa: BLE001
        return Check("Dashboard import", False, f"import failed: {exc}")


def run_healthcheck() -> list[Check]:
    """Run every check and return the flat list of results."""
    checks: list[Check] = [check_postgres()]
    checks.extend(check_tables())
    checks.append(check_qdrant())
    checks.append(check_dashboard_import())
    return checks


def summarize(checks: list[Check]) -> tuple[bool, str]:
    """Render the checklist; return ``(all_passed, text)``."""
    lines = ["=== VoiceLens demo healthcheck ==="]
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name:24} {c.detail}")
    all_ok = all(c.ok for c in checks)
    verdict = "READY TO DEMO" if all_ok else "NOT READY — see FAIL lines above"
    lines.append(f"--- {verdict} ---")
    return all_ok, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    checks = run_healthcheck()
    all_ok, text = summarize(checks)
    print(text)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
