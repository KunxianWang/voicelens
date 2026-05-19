"""Thin per-review extraction helper.

Wraps ``provider.extract`` with the post-hoc validators so the flow only
ever sees a ``ValidationResult`` — no half-validated provider output leaks
into the database layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from voicelens.nlp.absa.providers import ABSAProvider
from voicelens.nlp.absa.schema import ONTOLOGY_CODES_LATEST
from voicelens.nlp.absa.validators import ValidationResult, validate_absa_output


@dataclass
class ExtractionOutcome:
    review_id: int
    result: ValidationResult
    raw_aspect_count: int


def extract_for_review(
    provider: ABSAProvider,
    review_id: int,
    review_text: str,
    ontology_codes: tuple[str, ...] = ONTOLOGY_CODES_LATEST,
) -> ExtractionOutcome:
    raw = provider.extract(review_text)
    result = validate_absa_output(review_text, raw, ontology_codes)
    return ExtractionOutcome(
        review_id=review_id,
        result=result,
        raw_aspect_count=len(raw.aspects),
    )
