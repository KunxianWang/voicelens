from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from scripts.export_absa_predictions_for_labeling import export_predictions
from scripts.seed_aspect_ontology import seed_aspect_ontology
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    IngestRun,
    Review,
    Sku,
)
from voicelens.eval.absa_eval import evaluate_absa_rows


def test_evaluate_absa_computes_expected_metrics_on_tiny_fixture():
    rows = [
        {
            "text_raw": "Battery is great but bluetooth drops.",
            "gold_aspects": [
                {
                    "aspect_code": "battery",
                    "sentiment": "positive",
                    "severity": None,
                    "evidence_quote": "Battery is great",
                },
                {
                    "aspect_code": "bluetooth",
                    "sentiment": "negative",
                    "severity": "low",
                    "evidence_quote": "bluetooth drops",
                },
            ],
            "predicted_aspects": [
                {
                    "aspect_code": "battery",
                    "sentiment": "positive",
                    "severity": None,
                    "evidence_quote": "Battery is great",
                },
                {
                    "aspect_code": "charging",
                    "sentiment": "neutral",
                    "severity": None,
                    "evidence_quote": "Battery",
                },
            ],
        }
    ]

    metrics = evaluate_absa_rows(rows)

    assert metrics["aspect_extraction"]["precision"] == 0.5
    assert metrics["aspect_extraction"]["recall"] == 0.5
    assert metrics["sentiment_accuracy"] == 1.0
    assert metrics["evidence_quote_verbatim_rate"] == 1.0
    assert metrics["aspect_extraction"]["per_aspect"]["battery"]["tp"] == 1
    assert metrics["aspect_extraction"]["per_aspect"]["bluetooth"]["fn"] == 1
    assert metrics["aspect_extraction"]["per_aspect"]["charging"]["fp"] == 1


def test_export_predictions_works_on_tiny_db(tmp_path: Path, session):
    seed_aspect_ontology(session)
    brand = Brand(name="Anker")
    session.add(brand)
    session.flush()
    sku = Sku(brand_id=brand.id, asin="B0ANK")
    session.add(sku)
    session.flush()
    run = IngestRun(source="amazon_reviews_2023_mvp_subset", status="completed")
    session.add(run)
    session.flush()
    review = Review(
        source="amazon_reviews_2023",
        source_id="SRC1",
        sku_id=sku.id,
        language="en",
        lang_confidence=0.95,
        rating=5,
        posted_at=datetime(2024, 1, 1),
        text_raw="The battery is great.",
        char_len=21,
        ingest_run_id=run.id,
    )
    session.add(review)
    session.flush()
    aspect_id = session.scalar(select(AspectOntology.id).where(AspectOntology.code == "battery"))
    session.add(
        AspectMention(
            review_id=review.id,
            aspect_id=aspect_id,
            sentiment="positive",
            severity=None,
            evidence_quote="battery is great",
            model_name="gpt-test",
            aspect_version="v1",
        )
    )
    session.add(
        ABSAReviewStatus(
            review_id=review.id,
            aspect_version="v1",
            provider="openai",
            model_name="gpt-test",
            status="success",
            n_mentions=1,
            n_errors=0,
            error_codes_json=None,
        )
    )
    session.commit()

    holdout = tmp_path / "holdout.jsonl"
    output = tmp_path / "predictions.jsonl"
    with open(holdout, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "review_id": review.id,
                    "source_id": "SRC1",
                    "brand": "Anker",
                    "asin": "B0ANK",
                    "rating": 5,
                    "text_raw": "The battery is great.",
                    "gold_aspects": [],
                }
            )
            + "\n"
        )

    rows = export_predictions(
        holdout_path=holdout,
        output_path=output,
        provider="openai",
        model_name="gpt-test",
    )

    assert len(rows) == 1
    assert output.exists()
    assert rows[0]["predicted_aspects"][0]["aspect_code"] == "battery"
    assert rows[0]["validation_errors"] is None
