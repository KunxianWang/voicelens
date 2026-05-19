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
QDRANT_COLLECTION: str = _env("QDRANT_COLLECTION", "reviews_v2")

EMBEDDING_PROVIDER: str = _env("EMBEDDING_PROVIDER", "mock")
EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_BATCH_SIZE: int = int(_env("EMBEDDING_BATCH_SIZE", "64"))


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


DATA_DIR: Path = _resolve(_env("DATA_DIR", "voicelens/data"))
SAMPLE_REVIEWS_PATH: Path = DATA_DIR / "sample_reviews.jsonl"
AMAZON_FIXTURE_PATH: Path = DATA_DIR / "amazon_reviews_fixture.jsonl"

EXTERNAL_DATA_DIR: Path = _resolve(_env("EXTERNAL_DATA_DIR", "data"))
AMAZON_REVIEWS_PATH: Path = _resolve(_env("AMAZON_REVIEWS_PATH", "data/Electronics.jsonl"))
_amazon_metadata_path_str = _env("AMAZON_METADATA_PATH", "")
AMAZON_METADATA_PATH: Path | None = _resolve(_amazon_metadata_path_str) if _amazon_metadata_path_str else None

BRAND_CANDIDATES: tuple[str, ...] = tuple(
    b.strip()
    for b in _env("BRAND_CANDIDATES", "Anker,Soundcore,Bose,JBL,UGREEN,RAVPower,Aukey").split(",")
    if b.strip()
)
BRAND_ALLOWLIST: tuple[str, ...] = tuple(
    b.strip()
    for b in _env("BRAND_ALLOWLIST", ",".join(BRAND_CANDIDATES)).split(",")
    if b.strip()
)
INGEST_LIMIT: int = int(_env("INGEST_LIMIT", "50000"))

LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")

DQ_MIN_TEXT_LEN: int = 20
DQ_MIN_LANG_CONFIDENCE: float = 0.7
ALLOWED_LANGUAGES: tuple[str, ...] = ("en",)
