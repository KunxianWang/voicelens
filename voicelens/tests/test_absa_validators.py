from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from voicelens.nlp.absa.schema import ONTOLOGY_CODES_V1, ABSAOutput, AspectMentionOut
from voicelens.nlp.absa.validators import (
    ERR_DUPLICATE_ASPECT,
    ERR_NON_VERBATIM,
    ERR_NOT_IN_ONTOLOGY,
    ERR_SCHEMA,
    validate_absa_output,
)

REVIEW = "The battery dies after two hours and the charging cable broke on day one."


def _make_output(*aspects: dict) -> dict:
    return {"aspects": list(aspects)}


def test_pydantic_schema_rejects_negative_without_severity():
    with pytest.raises(PydanticValidationError):
        AspectMentionOut(
            aspect_code="battery",
            sentiment="negative",
            severity=None,
            evidence_quote="battery",
        )


def test_pydantic_schema_rejects_severity_on_positive():
    with pytest.raises(PydanticValidationError):
        AspectMentionOut(
            aspect_code="battery",
            sentiment="positive",
            severity="low",
            evidence_quote="battery",
        )


def test_pydantic_schema_rejects_unknown_sentiment():
    with pytest.raises(PydanticValidationError):
        AspectMentionOut.model_validate(
            {"aspect_code": "battery", "sentiment": "mixed", "evidence_quote": "battery"}
        )


def test_pydantic_schema_rejects_blank_quote():
    with pytest.raises(PydanticValidationError):
        AspectMentionOut.model_validate(
            {"aspect_code": "battery", "sentiment": "neutral", "evidence_quote": "   "}
        )


def test_validator_accepts_verbatim_quote():
    payload = _make_output(
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "The battery dies after two hours",
        }
    )
    result = validate_absa_output(REVIEW, payload, ONTOLOGY_CODES_V1)
    assert result.is_valid
    assert len(result.valid_mentions) == 1


def test_validator_rejects_non_verbatim_quote():
    payload = _make_output(
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "the battery is bad",
        }
    )
    result = validate_absa_output(REVIEW, payload, ONTOLOGY_CODES_V1)
    assert not result.is_valid
    assert result.valid_mentions == []
    assert any(err.code == ERR_NON_VERBATIM for err in result.errors)


def test_validator_rejects_unknown_aspect_code():
    payload = _make_output(
        {
            "aspect_code": "looks_pretty",
            "sentiment": "positive",
            "evidence_quote": "battery",
        }
    )
    result = validate_absa_output(REVIEW, payload, ONTOLOGY_CODES_V1)
    assert result.valid_mentions == []
    assert any(err.code == ERR_NOT_IN_ONTOLOGY for err in result.errors)


def test_validator_dedupes_aspect_within_review():
    payload = _make_output(
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "low",
            "evidence_quote": "The battery dies after two hours",
        },
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "battery",
        },
    )
    result = validate_absa_output(REVIEW, payload, ONTOLOGY_CODES_V1)
    assert len(result.valid_mentions) == 1
    assert any(err.code == ERR_DUPLICATE_ASPECT for err in result.errors)


def test_validator_returns_schema_error_for_malformed_payload():
    result = validate_absa_output(REVIEW, {"aspects": "not-a-list"}, ONTOLOGY_CODES_V1)
    assert result.valid_mentions == []
    assert len(result.errors) == 1
    assert result.errors[0].code == ERR_SCHEMA


def test_validator_accepts_absa_output_directly():
    direct = ABSAOutput(
        aspects=[
            AspectMentionOut(
                aspect_code="charging",
                sentiment="negative",
                severity="medium",
                evidence_quote="charging cable broke",
            )
        ]
    )
    result = validate_absa_output(REVIEW, direct, ONTOLOGY_CODES_V1)
    assert result.is_valid
    assert result.valid_mentions[0].aspect_code == "charging"


def test_validator_handles_empty_aspects():
    result = validate_absa_output(REVIEW, {"aspects": []}, ONTOLOGY_CODES_V1)
    assert result.is_valid
    assert result.valid_mentions == []
