from __future__ import annotations

from sqlalchemy import func, select

from scripts.seed_aspect_ontology import ASPECTS_V1, seed_aspect_ontology
from voicelens.db.models import AspectOntology


def test_seed_inserts_seven_aspects(session):
    result = seed_aspect_ontology(session)

    assert result == {"inserted": len(ASPECTS_V1), "updated": 0}
    rows = list(session.execute(select(AspectOntology)).scalars())
    codes = sorted(r.code for r in rows)
    assert codes == sorted(spec["code"] for spec in ASPECTS_V1)
    assert all(r.version == "v1" for r in rows)
    assert all(r.severity_applicable is True for r in rows)


def test_seed_is_idempotent(session):
    first = seed_aspect_ontology(session)
    second = seed_aspect_ontology(session)
    third = seed_aspect_ontology(session)

    assert first["inserted"] == len(ASPECTS_V1)
    assert second == {"inserted": 0, "updated": 0}
    assert third == {"inserted": 0, "updated": 0}

    total = session.scalar(select(func.count()).select_from(AspectOntology))
    assert total == len(ASPECTS_V1)


def test_seed_updates_changed_label(session):
    seed_aspect_ontology(session)
    row = session.scalar(
        select(AspectOntology).where(AspectOntology.code == "battery")
    )
    row.label = "stale label"
    session.flush()

    result = seed_aspect_ontology(session)

    assert result["inserted"] == 0
    assert result["updated"] == 1
    refreshed = session.scalar(
        select(AspectOntology).where(AspectOntology.code == "battery")
    )
    assert refreshed.label.startswith("Battery life")
