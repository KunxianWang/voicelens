from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session, sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")

from voicelens.db import models  # noqa: F401,E402
from voicelens.db.engine import Base, reset_engine  # noqa: E402


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/voicelens_test.db"


@pytest.fixture
def session(db_url: str, monkeypatch) -> Iterator[Session]:
    monkeypatch.setenv("DATABASE_URL", db_url)
    reset_engine()

    from voicelens.db.engine import get_engine
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    sess = SessionLocal()
    try:
        yield sess
    finally:
        sess.close()


@pytest.fixture
def fresh_engine(db_url: str, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", db_url)
    reset_engine()
    from voicelens.db.engine import get_engine
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    reset_engine()


@pytest.fixture
def sample_row_factory():
    def _make(**overrides):
        base = {
            "source": "amazon_reviews_2023",
            "source_id": "R0000001",
            "asin": "B0ANK10001",
            "brand": "Anker",
            "model_number": "PowerCore-10K",
            "category": "power_bank",
            "rating": 5,
            "verified": True,
            "posted_at": "2025-03-01T10:00:00",
            "helpful_count": 3,
            "text_raw": "Charges my phone two full times before needing a top up.",
            "language": "en",
            "lang_confidence": 0.95,
            "locale": "en-US",
        }
        base.update(overrides)
        return base
    return _make
