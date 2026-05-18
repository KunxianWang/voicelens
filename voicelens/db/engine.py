from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from voicelens.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    global _engine, _SessionLocal
    if url is None:
        if _engine is not None:
            return _engine
        target = DATABASE_URL
    else:
        target = url
    if _engine is None or str(_engine.url) != target:
        _engine = create_engine(target, echo=echo, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


@contextmanager
def get_session(url: str | None = None) -> Iterator[Session]:
    get_engine(url)
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
