"""Create all VoiceLens tables in the configured database."""
from __future__ import annotations

import sys

from voicelens.config import DATABASE_URL

# Importing models registers them with Base.metadata.
from voicelens.db import models  # noqa: F401
from voicelens.db.engine import Base, get_engine


def main() -> int:
    print(f"Connecting to {DATABASE_URL}")
    engine = get_engine()
    Base.metadata.create_all(engine)
    table_names = sorted(Base.metadata.tables.keys())
    print("Created/verified tables:")
    for name in table_names:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
