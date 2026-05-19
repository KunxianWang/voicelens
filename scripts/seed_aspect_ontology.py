"""Seed aspect ontology versions into Postgres.

Each version is a closed set of aspect codes. The script is idempotent
per ``(version, code)`` so re-running it touches only what changed.
v1 is preserved when newer versions are seeded so historical
``aspect_mention`` and ``absa_review_status`` rows stay interpretable.

Default seed target is the latest version
(``LATEST_ONTOLOGY_VERSION``). Pass ``--version v1`` to seed only v1,
or ``--all`` to seed every known version.
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voicelens.db import models  # noqa: F401
from voicelens.db.engine import Base, get_engine, get_session
from voicelens.db.models import AspectOntology
from voicelens.nlp.absa.schema import LATEST_ONTOLOGY_VERSION

# Single source of truth for aspect labels.
_LABELS_V1: dict[str, str] = {
    "battery": "Battery life / capacity / charging cycles",
    "charging": "Charging speed, USB-C compatibility, cable",
    "overheating": "Device gets too hot in use",
    "sound_quality": "Audio fidelity, volume, distortion, ANC",
    "bluetooth": "Bluetooth pairing, drops, range, latency",
    "delivery": "Shipping speed, packaging damage, missing items",
    "price": "Value for money, price-vs-competitor",
}

# v2 adds a general product-failure / durability aspect for reviews that
# describe the unit dying, breaking, or arriving defective without
# pointing at a specific component. The prompt + validator continue to
# prefer specific aspects when the failed component is clear.
_LABELS_V2: dict[str, str] = {
    **_LABELS_V1,
    "reliability": "Reliability / Durability / General product failure",
}

ASPECT_LABELS_BY_VERSION: dict[str, dict[str, str]] = {
    "v1": _LABELS_V1,
    "v2": _LABELS_V2,
}

# Back-compat alias for callers that imported ASPECTS_V1 directly. Kept
# as a tuple of dicts so the original shape is preserved.
ASPECTS_V1: tuple[dict[str, object], ...] = tuple(
    {"code": code, "label": label, "severity_applicable": True}
    for code, label in _LABELS_V1.items()
)
ASPECTS_V2: tuple[dict[str, object], ...] = tuple(
    {"code": code, "label": label, "severity_applicable": True}
    for code, label in _LABELS_V2.items()
)
ASPECTS_BY_VERSION: dict[str, tuple[dict[str, object], ...]] = {
    "v1": ASPECTS_V1,
    "v2": ASPECTS_V2,
}

ASPECT_VERSION = LATEST_ONTOLOGY_VERSION  # back-compat for older imports


def seed_aspect_ontology(
    session: Session, version: str = LATEST_ONTOLOGY_VERSION
) -> dict[str, int]:
    """Upsert the aspect ontology for ``version``.

    Returns ``{"inserted": N, "updated": M}``. Idempotent — a row whose
    label/severity already match is left untouched. Rows in other
    versions are NEVER deleted; v1 and v2 coexist.
    """
    if version not in ASPECTS_BY_VERSION:
        known = sorted(ASPECTS_BY_VERSION)
        raise ValueError(f"Unknown ontology version {version!r}. Known versions: {known}")
    inserted = 0
    updated = 0
    for spec in ASPECTS_BY_VERSION[version]:
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the aspect_ontology table.")
    parser.add_argument(
        "--version",
        default=LATEST_ONTOLOGY_VERSION,
        choices=sorted(ASPECTS_BY_VERSION),
        help=(
            "Ontology version to seed. Defaults to the latest version "
            f"({LATEST_ONTOLOGY_VERSION}). Use --all to seed every known version."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Seed every known ontology version (v1 + v2 + ...).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    engine = get_engine()
    print(f"Connecting to {engine.url}")
    Base.metadata.create_all(engine)

    versions = sorted(ASPECTS_BY_VERSION) if args.all else [args.version]
    with get_session() as session:
        per_version: list[tuple[str, dict[str, int]]] = []
        for version in versions:
            result = seed_aspect_ontology(session, version=version)
            per_version.append((version, result))
        total = session.scalar(select(func.count()).select_from(AspectOntology))
    for version, result in per_version:
        print(
            f"Aspect ontology {version}: "
            f"inserted={result['inserted']}, updated={result['updated']}"
        )
    print(f"aspect_ontology total_rows={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
