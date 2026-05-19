"""Pydantic schema for ABSA provider output.

The schema is intentionally strict: anything that fails the model-level
validation is rejected up front and never reaches ``validate_absa_output``,
which adds review-grounded checks (verbatim quote, ontology membership,
per-review uniqueness).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ONTOLOGY_CODES_V1: tuple[str, ...] = (
    "battery",
    "charging",
    "overheating",
    "sound_quality",
    "bluetooth",
    "delivery",
    "price",
)

SENTIMENTS: tuple[str, ...] = ("positive", "neutral", "negative")
SEVERITIES: tuple[str, ...] = ("low", "medium", "high")

Sentiment = Literal["positive", "neutral", "negative"]
Severity = Literal["low", "medium", "high"]


class AspectMentionOut(BaseModel):
    """One aspect detected on a single review."""

    model_config = ConfigDict(extra="forbid")

    aspect_code: str
    sentiment: Sentiment
    severity: Severity | None = None
    evidence_quote: str = Field(min_length=1)

    @field_validator("evidence_quote")
    @classmethod
    def _quote_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_quote must not be blank")
        return value

    @model_validator(mode="after")
    def _severity_iff_negative(self) -> AspectMentionOut:
        if self.sentiment == "negative" and self.severity is None:
            raise ValueError("severity is required when sentiment == 'negative'")
        if self.sentiment != "negative" and self.severity is not None:
            raise ValueError("severity must be null when sentiment != 'negative'")
        return self


class ABSAOutput(BaseModel):
    """Top-level provider output for one review."""

    model_config = ConfigDict(extra="forbid")

    aspects: list[AspectMentionOut] = Field(default_factory=list)
