"""Seed the MVP v1 aspect ontology into Postgres.

Idempotent: re-running upserts the seven canonical aspects without creating
duplicates. Adding a new aspect to ``ASPECTS_V1`` and re-running will insert
the new row; modifying an existing row's label updates it in place.
"""
from __future__ import annotations

import sys

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import Base, get_engine, get_session
from voicelens.db.models import AspectOntology

ASPECT_VERSION = "v1"

ASPECTS_V1: tuple[dict[str, object], ...] = (
    {"code": "battery", "label": "Battery life / capacity / charging cycles", "severity_applicable": True},
    {"code": "charging", "label": "Charging speed, USB-C compatibility, cable", "severity_applicable": True},
    {"code": "overheating", "label": "Device gets too hot in use", "severity_applicable": True},
    {"code": "sound_quality", "label": "Audio fidelity, volume, distortion, ANC", "severity_applicable": True},
    {"code": "bluetooth", "label": "Bluetooth pairing, drops, range, latency", "severity_applicable": True},
    {"code": "delivery", "label": "Shipping speed, packaging damage, missing items", "severity_applicable": True},
    {"code": "price", "label": "Value for money, price-vs-competitor", "severity_applicable": True},
)


def seed_aspect_ontology(session: Session, version: str = ASPECT_VERSION) -> dict[str, int]:
    """Upsert the seven v1 aspects. Returns ``{"inserted": N, "updated": M}``."""
    inserted = 0
    updated = 0
    for spec in ASPECTS_V1:
        row = session.scalar(
            select(AspectOntology).where(
                AspectOntology.version == version,
                AspectOntology.code == spec["code"],
            )
        )
        if row is None:
            session.add(
                AspectOntology(
                    version=version,
                    code=str(spec["code"]),
                    label=str(spec["label"]),
                    severity_applicable=bool(spec["severity_applicable"]),
                )
            )
            inserted += 1
        else:
            changed = False
            if row.label != spec["label"]:
                row.label = str(spec["label"])
                changed = True
            if row.severity_applicable != spec["severity_applicable"]:
                row.severity_applicable = bool(spec["severity_applicable"])
                changed = True
            if changed:
                updated += 1
    session.flush()
    return {"inserted": inserted, "updated": updated}


def main() -> int:
    engine = get_engine()
    print(f"Connecting to {engine.url}")
    Base.metadata.create_all(engine)

    with get_session() as session:
        result = seed_aspect_ontology(session)
        total = session.scalar(select(func.count()).select_from(AspectOntology))
    print(
        f"Aspect ontology {ASPECT_VERSION}: "
        f"inserted={result['inserted']}, updated={result['updated']}, total_rows={total}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
