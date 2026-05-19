"""Validators for ABSA provider output.

Pydantic enforces schema-level invariants (sentiment enum, severity coupling).
This module adds review-grounded validation: ontology-membership, verbatim
quote, and per-review aspect uniqueness. Invalid mentions are dropped with
a structured error so the flow can record per-check failure counts without
poisoning the database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from voicelens.nlp.absa.schema import ABSAOutput, AspectMentionOut

ERR_NOT_IN_ONTOLOGY = "aspect_code_not_in_ontology"
ERR_NON_VERBATIM = "evidence_quote_not_verbatim"
ERR_DUPLICATE_ASPECT = "duplicate_aspect_in_review"
ERR_SCHEMA = "schema_validation_failed"


@dataclass
class ValidationError:
    code: str
    message: str
    aspect_code: str | None = None


@dataclass
class ValidationResult:
    valid_mentions: list[AspectMentionOut] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _parse_output(raw: Any) -> tuple[ABSAOutput | None, ValidationError | None]:
    if isinstance(raw, ABSAOutput):
        return raw, None
    if isinstance(raw, dict):
        try:
            return ABSAOutput.model_validate(raw), None
        except PydanticValidationError as exc:
            return None, ValidationError(code=ERR_SCHEMA, message=str(exc))
    return None, ValidationError(
        code=ERR_SCHEMA,
        message=f"unsupported provider output type: {type(raw).__name__}",
    )


def validate_absa_output(
    review_text: str,
    output: Any,
    ontology_codes: tuple[str, ...] | list[str] | set[str],
) -> ValidationResult:
    """Filter provider output down to mentions safe to insert.

    Drops:
    - mentions with aspect_code not in ``ontology_codes``
    - mentions whose evidence_quote is not a verbatim substring of
      ``review_text``
    - duplicate aspect_code within the same review (keep first occurrence)

    Returns the surviving mentions plus a structured list of dropped
    reasons; callers use ``errors`` to populate per-check failure counts.
    """
    parsed, parse_error = _parse_output(output)
    if parsed is None:
        return ValidationResult(errors=[parse_error] if parse_error else [])

    allowed: set[str] = set(ontology_codes)
    seen_codes: set[str] = set()
    result = ValidationResult()

    for mention in parsed.aspects:
        if mention.aspect_code not in allowed:
            result.errors.append(
                ValidationError(
                    code=ERR_NOT_IN_ONTOLOGY,
                    message=f"aspect_code '{mention.aspect_code}' not in ontology",
                    aspect_code=mention.aspect_code,
                )
            )
            continue
        if mention.evidence_quote not in review_text:
            result.errors.append(
                ValidationError(
                    code=ERR_NON_VERBATIM,
                    message=(
                        f"evidence_quote is not a verbatim substring of the review "
                        f"(aspect={mention.aspect_code})"
                    ),
                    aspect_code=mention.aspect_code,
                )
            )
            continue
        if mention.aspect_code in seen_codes:
            result.errors.append(
                ValidationError(
                    code=ERR_DUPLICATE_ASPECT,
                    message=f"duplicate aspect '{mention.aspect_code}' in same review",
                    aspect_code=mention.aspect_code,
                )
            )
            continue
        seen_codes.add(mention.aspect_code)
        result.valid_mentions.append(mention)

    return result
