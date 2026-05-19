from __future__ import annotations

import pytest

from voicelens.retrieval.qdrant_index import (
    build_embedding_text,
    build_payload,
    build_point_id,
)


def test_build_point_id_is_deterministic():
    a = build_point_id(123, "v2", "anthropic", "claude-opus-4.6")
    b = build_point_id(123, "v2", "anthropic", "claude-opus-4.6")
    assert a == b


def test_build_point_id_changes_with_each_key_component():
    base = build_point_id(123, "v2", "anthropic", "claude-opus-4.6")
    assert base != build_point_id(124, "v2", "anthropic", "claude-opus-4.6")
    assert base != build_point_id(123, "v1", "anthropic", "claude-opus-4.6")
    assert base != build_point_id(123, "v2", "openai", "claude-opus-4.6")
    assert base != build_point_id(123, "v2", "anthropic", "gpt-4o-mini")


def test_build_point_id_is_uuid_string():
    pid = build_point_id(1, "v2", "anthropic", "claude-opus-4.6")
    # UUID5 strings are 36 chars with dashes; tests pin the shape so a
    # future "use ints" change has to update this test deliberately.
    assert isinstance(pid, str)
    assert len(pid) == 36
    assert pid.count("-") == 4


def test_build_embedding_text_with_mentions():
    text = build_embedding_text(
        "Battery died after a week.",
        [
            {"aspect_code": "battery", "sentiment": "negative", "severity": "medium"},
            {"aspect_code": "price", "sentiment": "positive", "severity": None},
        ],
    )
    assert text.startswith("Review: Battery died after a week.")
    assert "Aspects: battery negative medium; price positive" in text


def test_build_embedding_text_with_no_mentions_emits_marker():
    text = build_embedding_text("Bought this last weekend.", [])
    assert text == "Review: Bought this last weekend.\nAspects: (none)"


def test_build_embedding_text_handles_none_input():
    text = build_embedding_text(None, None)  # type: ignore[arg-type]
    assert text == "Review: \nAspects: (none)"


def test_build_payload_assembles_parallel_lists():
    review = {
        "review_id": 42,
        "source_id": "R-42",
        "source": "amazon_reviews_2023",
        "sku_id": 7,
        "asin": "B0ANK",
        "brand": "Anker",
        "rating": 1,
        "verified": True,
        "posted_at": "2024-01-01T00:00:00",
        "text_raw": "The battery died and the bluetooth dropped.",
    }
    mentions = [
        {
            "aspect_code": "battery",
            "sentiment": "negative",
            "severity": "medium",
            "evidence_quote": "The battery died",
        },
        {
            "aspect_code": "bluetooth",
            "sentiment": "negative",
            "severity": "low",
            "evidence_quote": "bluetooth dropped",
        },
    ]
    payload = build_payload(
        review=review,
        mentions=mentions,
        aspect_version="v2",
        provider="anthropic",
        model_name="claude-opus-4.6",
        absa_status="success",
    )
    assert payload["review_id"] == 42
    assert payload["brand"] == "Anker"
    assert payload["aspect_codes"] == ["battery", "bluetooth"]
    assert payload["sentiments"] == ["negative", "negative"]
    assert payload["severities"] == ["medium", "low"]
    assert payload["evidence_quotes"] == ["The battery died", "bluetooth dropped"]
    assert payload["mention_count"] == 2
    assert payload["absa_status"] == "success"
    assert payload["aspect_version"] == "v2"
    assert payload["provider"] == "anthropic"
    assert payload["model_name"] == "claude-opus-4.6"


def test_build_payload_no_mentions_yields_empty_arrays():
    review = {
        "review_id": 99,
        "source": "amazon_reviews_2023",
        "source_id": "R-99",
        "sku_id": 1,
        "asin": "B0X",
        "brand": "Bose",
        "rating": 5,
        "verified": False,
        "posted_at": "2024-02-01T00:00:00",
        "text_raw": "Bought this. Works fine.",
    }
    payload = build_payload(
        review=review,
        mentions=[],
        aspect_version="v2",
        provider="anthropic",
        model_name="claude-opus-4.6",
        absa_status="no_mentions",
    )
    assert payload["aspect_codes"] == []
    assert payload["sentiments"] == []
    assert payload["severities"] == []
    assert payload["evidence_quotes"] == []
    assert payload["mention_count"] == 0
    assert payload["absa_status"] == "no_mentions"


def test_build_payload_skips_mention_without_aspect_code():
    review = {
        "review_id": 1,
        "rating": 3,
        "asin": "B0X",
        "brand": "Anker",
        "text_raw": "ok",
    }
    payload = build_payload(
        review=review,
        mentions=[
            {"sentiment": "positive", "severity": None, "evidence_quote": "ok"},
            {"aspect_code": "delivery", "sentiment": "neutral", "severity": None,
             "evidence_quote": "Arrived on time"},
        ],
        aspect_version="v2",
        provider="anthropic",
        model_name="claude-opus-4.6",
        absa_status="success",
    )
    assert payload["aspect_codes"] == ["delivery"]
    assert payload["mention_count"] == 1


def test_ensure_collection_creates_then_is_idempotent():
    """Verify create-once + create-again-noop against in-memory Qdrant."""
    from qdrant_client import QdrantClient

    from voicelens.retrieval.qdrant_index import ensure_collection

    client = QdrantClient(":memory:")
    created_first = ensure_collection(client, "smoke_v2", vector_size=16)
    created_second = ensure_collection(client, "smoke_v2", vector_size=16)
    assert created_first is True
    assert created_second is False


def test_ensure_collection_rejects_dimension_mismatch():
    from qdrant_client import QdrantClient

    from voicelens.retrieval.qdrant_index import ensure_collection

    client = QdrantClient(":memory:")
    ensure_collection(client, "smoke_dim", vector_size=16)
    with pytest.raises(RuntimeError, match="vector_size"):
        ensure_collection(client, "smoke_dim", vector_size=32)


def test_ensure_collection_recreate_drops_and_recreates():
    from qdrant_client import QdrantClient

    from voicelens.retrieval.qdrant_index import ensure_collection

    client = QdrantClient(":memory:")
    ensure_collection(client, "smoke_recreate", vector_size=16)
    # Changing dim with recreate=True should succeed.
    created = ensure_collection(
        client, "smoke_recreate", vector_size=32, recreate=True
    )
    assert created is True
