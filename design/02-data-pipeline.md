# Data Pipeline — End-to-End

Prefect-orchestrated. Medallion-layered (Bronze/Silver/Gold) on S3 + Postgres + ClickHouse + Qdrant.

```
┌─────────────────┐
│  Sources        │  Amazon Reviews 2023 · Trustpilot · Reddit · Mock SP-API ·
│                 │  Mock Shopify · Mock TikTok Shop · YouTube transcripts
└──────┬──────────┘
       │
       ▼
┌─────────────────┐                                  Bronze:
│  ingest_flow    │ ─────────────────────────────►   s3://voicelens/bronze/{source}/{date}/*.jsonl
└──────┬──────────┘                                  raw, untouched, append-only
       │
       ▼
┌─────────────────┐  dedupe (source_id hash)       Silver:
│  normalize_flow │  langid (fastText)             s3://voicelens/silver/reviews/
│                 │  translate (Qwen-family /      partitioned by {source, locale, date}
│                 │   hosted MT, post-MVP)         
│                 │  PII redact (presidio)         parquet, schema-stable
│                 │  schema-validate (pydantic)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐  Aspect ontology vN              Gold:
│  absa_flow      │  Constrained-JSON LLM           postgres.aspect_mention
│                 │  Sentiment + salience           clickhouse.aspect_volume (hourly MV)
└──────┬──────────┘
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
┌─────────────────┐                ┌─────────────────┐
│  cluster_flow   │                │  embed_flow      │
│  BERTopic per   │                │  bge-m3 dense    │
│  {sku, locale,  │                │  + BM25 sparse   │
│   week}         │                │                  │
│  drift via JSD  │                │  → Qdrant        │
└──────┬──────────┘                └─────────────────┘
       │
       ▼
┌─────────────────┐                Gold:
│  anomaly_flow   │  EWMA + STL    postgres.incident
│                 │  severity      (linked clusters,
│                 │  scoring       severity, hypothesis)
│                 │                Slack/email alert
└─────────────────┘
```

## Schemas (Silver — Parquet)

### `silver.reviews`

| col | type | notes |
|---|---|---|
| review_id | string | source_prefix + source_id |
| source | enum | amazon, trustpilot, reddit, shopify, tiktok |
| sku_id | int64 | nullable for non-product mentions |
| asin | string | nullable |
| locale | string | ISO (en-US, de-DE, ja-JP, ...) |
| language | string | detected; may differ from locale |
| lang_confidence | float | fastText score |
| rating | int8 | 1–5; null on Reddit |
| verified_purchase | bool | nullable per source |
| posted_at | timestamp[ms, UTC] | |
| helpful_count | int32 | nullable |
| text_raw | string | original |
| text_translated_en | string | null if language == en |
| char_len | int32 | derived |
| ingest_run_id | string | Prefect flow run |
| ingest_ts | timestamp | |

Partitioning: `source/locale/dt=YYYY-MM-DD`.

## Idempotency contract

- All ingest writes are idempotent on `(source, source_id)` — re-running a Prefect run overwrites the same row.
- Downstream tasks key on `(review_id, model_version, aspect_version)` so re-running ABSA after an ontology bump produces a new generation, old kept for backfill diffs.
- ClickHouse materialized views are `ReplacingMergeTree` on `(ts, sku_id, locale, aspect_id, generation)`.

## ABSA contract (LLM tool-use)

```json
{
  "name": "extract_aspects",
  "description": "Extract product aspects and their sentiments from a customer review.",
  "input_schema": {
    "type": "object",
    "required": ["aspects"],
    "properties": {
      "aspects": {
        "type": "array",
        "maxItems": 8,
        "items": {
          "type": "object",
          "required": ["aspect_code", "sentiment", "salience", "evidence_quote"],
          "properties": {
            "aspect_code": {"type": "string", "description": "Must match aspect_ontology.code for the active version."},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
            "salience": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_quote": {"type": "string", "description": "Exact substring from the review."}
          }
        }
      }
    }
  }
}
```

The model receives the active aspect ontology (max 200 codes) in-context, prompt-cached. Outputs that fail JSON schema validation are retried once with a corrective system message, then dropped to a dead-letter bucket.

## Backfill strategy

Ontology bumps:
1. Write new `aspect_version` row.
2. Trigger `absa_flow` over reviews where `aspect_version_processed < new`.
3. Eval gate: macro-F1 on labeled holdout must not regress > 1pp; otherwise the new version is pulled.

## Cost guardrails

- Per-flow Prefect tag with daily token budget.
- LLM client wraps Anthropic/OpenAI/vLLM with a budget gate that blocks at 120% of budget and pages on-call.
- Daily Langfuse digest of cost-per-source for the analyst team.
