from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


DATABASE_URL: str = _env(
    "DATABASE_URL",
    "postgresql+psycopg2://voicelens:voicelens@localhost:5432/voicelens",
)
QDRANT_URL: str = _env("QDRANT_URL", "http://localhost:6333")

DATA_DIR: Path = (PROJECT_ROOT / _env("DATA_DIR", "voicelens/data")).resolve()
SAMPLE_REVIEWS_PATH: Path = DATA_DIR / "sample_reviews.jsonl"

LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")

DQ_MIN_TEXT_LEN: int = 20
DQ_MIN_LANG_CONFIDENCE: float = 0.7
ALLOWED_LANGUAGES: tuple[str, ...] = ("en",)
