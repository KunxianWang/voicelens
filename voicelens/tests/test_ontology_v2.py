"""Milestone 2C.2: ontology v2 (reliability aspect)."""
from __future__ import annotations

from sqlalchemy import func, select

from scripts.seed_aspect_ontology import (
    ASPECT_LABELS_BY_VERSION,
    ASPECTS_BY_VERSION,
    seed_aspect_ontology,
)
from voicelens.db.models import AspectOntology
from voicelens.nlp.absa import (
    LATEST_ONTOLOGY_VERSION,
    ONTOLOGY_CODES_LATEST,
    ONTOLOGY_CODES_V1,
    ONTOLOGY_CODES_V2,
    MockABSAProvider,
    ontology_codes,
    validate_absa_output,
)
from voicelens.nlp.absa.prompts import ABSA_SYSTEM_PROMPT

# --------------------------------------------------------------------------
# Schema + module-level invariants
# --------------------------------------------------------------------------


def test_v2_extends_v1_with_reliability():
    assert "reliability" not in ONTOLOGY_CODES_V1
    assert "reliability" in ONTOLOGY_CODES_V2
    assert set(ONTOLOGY_CODES_V1).issubset(set(ONTOLOGY_CODES_V2))
    assert len(ONTOLOGY_CODES_V1) == 7
    assert len(ONTOLOGY_CODES_V2) == 8


def test_latest_alias_points_at_v2():
    assert LATEST_ONTOLOGY_VERSION == "v2"
    assert ONTOLOGY_CODES_LATEST == ONTOLOGY_CODES_V2


def test_ontology_codes_helper_returns_per_version_set():
    assert ontology_codes("v1") == ONTOLOGY_CODES_V1
    assert ontology_codes("v2") == ONTOLOGY_CODES_V2
    assert ontology_codes() == ONTOLOGY_CODES_LATEST


def test_ontology_codes_helper_rejects_unknown_version():
    import pytest

    with pytest.raises(ValueError, match="Unknown ontology version"):
        ontology_codes("v99")


# --------------------------------------------------------------------------
# Seed script
# --------------------------------------------------------------------------


def test_seed_v2_inserts_8_aspects(session):
    result = seed_aspect_ontology(session, version="v2")
    assert result["inserted"] == 8

    rows = session.execute(
        select(AspectOntology).where(AspectOntology.version == "v2")
    ).scalars().all()
    codes = sorted(r.code for r in rows)
    assert codes == sorted(ASPECT_LABELS_BY_VERSION["v2"].keys())
    assert "reliability" in codes


def test_seed_v2_keeps_v1_intact(session):
    seed_aspect_ontology(session, version="v1")
    n_v1_before = session.scalar(
        select(func.count()).select_from(AspectOntology).where(AspectOntology.version == "v1")
    )

    seed_aspect_ontology(session, version="v2")
    n_v1_after = session.scalar(
        select(func.count()).select_from(AspectOntology).where(AspectOntology.version == "v1")
    )
    n_v2_after = session.scalar(
        select(func.count()).select_from(AspectOntology).where(AspectOntology.version == "v2")
    )

    assert n_v1_before == n_v1_after == 7
    assert n_v2_after == 8


def test_seed_v2_is_idempotent(session):
    first = seed_aspect_ontology(session, version="v2")
    second = seed_aspect_ontology(session, version="v2")
    third = seed_aspect_ontology(session, version="v2")

    assert first["inserted"] == 8
    assert second == {"inserted": 0, "updated": 0}
    assert third == {"inserted": 0, "updated": 0}


def test_seed_rejects_unknown_version(session):
    import pytest

    with pytest.raises(ValueError, match="Unknown ontology version"):
        seed_aspect_ontology(session, version="v99")


def test_aspects_by_version_table_is_consistent():
    """Sanity check the lookup table the script + tests both rely on."""
    assert set(ASPECTS_BY_VERSION) == {"v1", "v2"}
    assert {a["code"] for a in ASPECTS_BY_VERSION["v1"]} == set(ONTOLOGY_CODES_V1)
    assert {a["code"] for a in ASPECTS_BY_VERSION["v2"]} == set(ONTOLOGY_CODES_V2)


# --------------------------------------------------------------------------
# Validators accept reliability under v2 and reject it under v1
# --------------------------------------------------------------------------


def test_validator_accepts_reliability_under_v2():
    text = "Stopped working after one week."
    payload = {
        "aspects": [
            {
                "aspect_code": "reliability",
                "sentiment": "negative",
                "severity": "medium",
                "evidence_quote": "Stopped working after one week",
            }
        ]
    }
    result = validate_absa_output(text, payload, ONTOLOGY_CODES_V2)
    assert result.is_valid
    assert result.valid_mentions[0].aspect_code == "reliability"


def test_validator_rejects_reliability_under_v1():
    text = "Stopped working after one week."
    payload = {
        "aspects": [
            {
                "aspect_code": "reliability",
                "sentiment": "negative",
                "severity": "medium",
                "evidence_quote": "Stopped working after one week",
            }
        ]
    }
    result = validate_absa_output(text, payload, ONTOLOGY_CODES_V1)
    assert not result.is_valid
    assert any(err.code == "aspect_code_not_in_ontology" for err in result.errors)


def test_validator_still_rejects_unknown_aspect():
    text = "Stopped working after one week."
    payload = {
        "aspects": [
            {
                "aspect_code": "made_up",
                "sentiment": "negative",
                "severity": "low",
                "evidence_quote": "Stopped working after one week",
            }
        ]
    }
    result = validate_absa_output(text, payload, ONTOLOGY_CODES_V2)
    assert not result.is_valid


# --------------------------------------------------------------------------
# MockABSAProvider reliability detection
# --------------------------------------------------------------------------


def test_mock_extracts_reliability_for_stopped_working():
    provider = MockABSAProvider()
    text = "Stopped working after one week."

    output = provider.extract(text)
    codes = {m.aspect_code: m for m in output.aspects}

    assert "reliability" in codes
    mention = codes["reliability"]
    assert mention.sentiment == "negative"
    assert mention.severity == "medium"
    assert mention.evidence_quote in text


def test_mock_extracts_reliability_for_dead_on_arrival():
    provider = MockABSAProvider()
    text = "Dead on arrival. Returned it immediately."

    output = provider.extract(text)
    codes = {m.aspect_code for m in output.aspects}

    assert "reliability" in codes


def test_mock_does_not_extract_reliability_for_delivery_only_damage():
    provider = MockABSAProvider()
    text = "Package was damaged but product works."

    output = provider.extract(text)
    codes = {m.aspect_code for m in output.aspects}

    assert "reliability" not in codes
    # ``delivery`` is the right aspect for shipping/package damage.
    assert "delivery" in codes


def test_mock_prefers_specific_aspect_over_reliability():
    provider = MockABSAProvider()
    text = "Battery died after a week of light use."

    output = provider.extract(text)
    codes = {m.aspect_code for m in output.aspects}

    assert "battery" in codes
    assert "reliability" not in codes, "specific aspect must beat reliability catch-all"


def test_mock_no_reliability_for_neutral_text():
    provider = MockABSAProvider()
    text = "Bought this last weekend and have been using it ever since."

    output = provider.extract(text)
    codes = {m.aspect_code for m in output.aspects}
    assert "reliability" not in codes


def test_mock_reliability_quote_is_verbatim_substring():
    provider = MockABSAProvider()
    text = "Bought this and it broke after two days. So disappointed."

    output = provider.extract(text)
    for mention in output.aspects:
        assert mention.evidence_quote in text


def test_mock_reliability_emits_negative_with_severity():
    """Negative sentiment must carry a severity per the schema rule."""
    provider = MockABSAProvider()
    text = "This unit is defective."

    output = provider.extract(text)
    [mention] = [m for m in output.aspects if m.aspect_code == "reliability"]
    assert mention.sentiment == "negative"
    assert mention.severity in {"low", "medium", "high"}


# --------------------------------------------------------------------------
# Prompt content checks
# --------------------------------------------------------------------------


def test_prompt_lists_reliability_in_closed_ontology():
    assert "reliability" in ABSA_SYSTEM_PROMPT


def test_prompt_includes_disambiguation_rules():
    """Specific-aspect / delivery / ambiguity guidance must be in the prompt."""
    assert "Specific aspects beat" in ABSA_SYSTEM_PROMPT
    assert "shipping or package damage" in ABSA_SYSTEM_PROMPT
    assert "ambiguous" in ABSA_SYSTEM_PROMPT


def test_prompt_includes_worked_examples():
    for snippet in (
        "Stopped working after one week",
        "Dead on arrival",
        "Started smoking while charging",
        "USB-C port stopped charging",
        "Battery died after a week",
        "Package was damaged but product works",
    ):
        assert snippet in ABSA_SYSTEM_PROMPT, f"prompt missing example: {snippet!r}"
