# VoiceLens — VoC Data Engineering & Analytics Platform for Cross-Border E-commerce

> Ingests, normalizes, and analyzes hundreds of thousands of multilingual customer reviews for consumer-electronics brands (Anker-class, Soundcore, Eufy archetype) selling across Amazon and other marketplaces. Surfaces emerging quality issues, aspect-level sentiment trends, competitor mentions, and anomaly alerts — exposed through a dashboard and an agentic-RAG query layer.

> 📦 **Current focus = 4–6 week MVP**, scope locked in [`design/04-mvp-spec.md`](design/04-mvp-spec.md). This README leads with the MVP; the bigger target architecture is documented at the end (§11) and in [`design/`](design/) as the planned extension surface.

---

## 1. Project Summary

VoiceLens is a **data-engineering-first** platform for cross-border e-commerce VoC analytics. The core of the project is the ETL + ML pipeline that turns raw reviews into governed, queryable, aspect-labeled facts. On top of that data, an LLM-driven query layer answers analyst questions with inline citations.

**MVP stack:** Python · Prefect · Postgres · Qdrant · BERTopic / HDBSCAN · FastAPI · Streamlit · LangGraph · Docker

**Target stack (post-MVP, planned not built):** ClickHouse · Mem0 · Next.js · Langfuse · OpenTelemetry · K8s · vLLM · reranker fine-tuning

---

## 2. Why this project

Cross-border CE brands like **Anker** ship hundreds of SKUs into 10+ marketplaces and 6+ languages. The most valuable, hardest-to-extract data asset they own is **unstructured customer voice** — Amazon reviews, Q&A, social comments, support tickets.

Today this is processed by:
- Manual ops reading spreadsheets of translated reviews — slow, lossy, lagging by 2–4 weeks.
- Off-the-shelf "review aggregators" — keyword counts only, no aspect reasoning, no SKU-level RCA.
- One-off ChatGPT prompts — no traceability, no eval, no governance.

**VoiceLens** is the engineered replacement: a streaming VoC pipeline + governed analytical store + dashboard + an agentic-RAG query layer that a PM, Quality Engineer, or Customer Insights analyst can talk to.

> **Example questions the platform answers:**
> - "What's the biggest complaint on the *PowerCore 24K* this month?"
> - "Has `overheating` been rising over the last 30 days for SKU A2670?"
> - "What competitors do users name when leaving negative reviews of *Soundcore Liberty*?"
> - "Quote 5 reviews about `bluetooth` drops for SKU A3947."

---

## 3. MVP Scope

Hard, locked. The detailed spec lives in [`design/04-mvp-spec.md`](design/04-mvp-spec.md).

**In scope:**

| | |
|---|---|
| Dataset | **Amazon Reviews 2023** (McAuley Lab, UCSD), Electronics category. **English only.** **3–5 brands** (recommended: Anker, Soundcore, RAVPower/Aukey, Bose/JBL, optional UGREEN). 200k–500k reviews after filter. |
| Pipeline | ingest → **data-quality checks** → normalize → Postgres → ABSA → embed → cluster → anomaly |
| ABSA | Closed 7-aspect ontology: `battery`, `charging`, `overheating`, `sound_quality`, `bluetooth`, `delivery`, `price`. Sentiment ∈ {pos, neu, neg}. Severity ∈ {low, med, high}. Verbatim `evidence_quote`. |
| Retrieval | Qdrant hybrid (BM25 + bge-m3 dense) + off-the-shelf bge-reranker-v2-m3. |
| Agent | LangGraph 3-node graph (`planner` → `retrieve_or_sql` → `respond_with_citations`). Four locked question templates. |
| Dashboard | Streamlit, 5 pages. |

**Explicitly OUT of MVP** (designed in §11, not built):

- Multi-locale (DE / JP) and translation-augmented embeddings.
- Mem0 / 3-tier memory.
- Critique node + auto-retry.
- Human-in-the-loop action tools (Jira draft, Slack post, email factory).
- Reranker LoRA fine-tune case study.
- Mock SP-API / Shopify / TikTok Shop integration surface.
- ClickHouse, K8s, Terraform, Helm.
- Cost router across multiple LLM providers; local vLLM-served models.
- Next.js + AG-UI frontend.
- Langfuse + OpenTelemetry full observability stack.

---

## 4. MVP Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                              │
│   SKU trend · Aspect dist. · Emerging issues · Chat · Alerts        │
└──────────────┬─────────────────────────────────────────────────────┘
               │ SSE
┌──────────────▼─────────────────────────────────────────────────────┐
│                       FastAPI (async)                              │
└──────────────┬─────────────────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────────────────┐
│             LangGraph supervisor (3 nodes)                          │
│   planner ──► retrieve_or_sql ──► respond_with_citations            │
└──────────────┬─────────────────────────────────────────────────────┘
               │
        ┌──────┴──────────────┐
        │                     │
┌───────▼──────┐    ┌─────────▼──────────────────────────┐
│  Postgres    │    │  Qdrant (single collection: en)     │
│  - review    │    │  hybrid: BM25 + bge-m3 dense        │
│  - aspect_   │    │  payload: sku_id, aspect_codes,     │
│    mention   │    │  posted_at, rating, cluster_id      │
│  - cluster   │    │  rerank: bge-reranker-v2-m3         │
│  - incident  │    └─────────────────────────────────────┘
└──────▲───────┘                       ▲
       │                               │ writes
       │ writes                        │
┌──────┴───────────────────────────────┴─────────────────────────────┐
│                         Offline Pipeline (Prefect 3)                │
│                                                                     │
│   Amazon Reviews 2023 (jsonl on S3 / local)                          │
│            │                                                        │
│            ▼                                                        │
│    ingest_flow ──► dq_flow ──► normalize_flow ──► absa_flow ──►     │
│    embed_flow ──► cluster_flow ──► anomaly_flow                     │
│                                                                     │
│   All flows idempotent; bronze (raw) / silver (parquet) / gold      │
│   (Postgres + Qdrant)                                               │
└────────────────────────────────────────────────────────────────────┘
```

### Why this shape

- **Postgres as the only analytical store in MVP** — at 200k–500k rows it's fast and removes ops surface. ClickHouse comes back in the target architecture when event-grain volume warrants it.
- **Single LangGraph supervisor** (no multi-agent). Defended in [`design/01-architecture-decisions.md`](design/01-architecture-decisions.md) ADR-001.
- **Hybrid retrieval > pure vector** for review data (full of model numbers, brand strings). ADR-002.
- **Aspect ontology is a versioned database artifact**, not a prompt blob — enables backfill and drift monitoring.

---

## 5. Data Pipeline

Prefect 3, one Docker container in MVP. Medallion-layered: **Bronze** (raw, append-only) → **Silver** (deduped, normalized parquet) → **Gold** (Postgres + Qdrant).

```
[Amazon Reviews 2023 jsonl]
        │
        ▼
   ingest_flow         resolved brand allowlist filter; partition by brand/month → bronze parquet
        │
        ▼
   dq_flow             quality gates (§5.1); rejects to dead-letter; metrics emitted
        │
        ▼
   normalize_flow      dedupe; PII redact; pydantic schema-validate → silver parquet → Postgres upsert
        │
        ▼
   absa_flow           §6
        │
        ├──► embed_flow      bge-m3 dense + BM25 sparse → Qdrant
        │
        ├──► cluster_flow    §7 (BERTopic per SKU+week on negatives)
        │
        └──► anomaly_flow    §7 (EWMA z-score → incident)
```

### 5.1 Data Quality (DQ) module

All checks run **before Gold load**. Each row either passes, is corrected, or is routed to dead-letter with a reason code. Metrics surface in the Prefect run dashboard and a daily DQ report in the Streamlit dashboard.

| Check | Rule | On failure |
|---|---|---|
| Duplicate `review_id` | Hash on `(source, asin, reviewer_id, posted_at, sha1(text))` | Drop duplicates; keep earliest |
| Missing SKU / ASIN | `asin` must be non-null and resolvable in the SKU map | Route to `dl_missing_sku` |
| Invalid rating range | `rating ∈ [1, 5]` (or null only if source explicitly allows) | Route to `dl_invalid_rating` |
| Language confidence | fastText `lang_confidence ≥ 0.7` and `language ∈ {en}` for MVP | Route to `dl_low_lang_conf` |
| Empty / spam text | `len(text.strip()) ≥ 20`; spam regex (URLs-only, ALL-CAPS-only) | Route to `dl_empty_or_spam` |
| ABSA schema validation | LLM output must be valid JSON, `aspect_code` ∈ ontology, `evidence_quote` is a verbatim substring | Retry once; then route to `dl_absa_invalid` |

Per-flow DQ metrics exposed:
- `dq_pass_rate` per check, per source.
- `dl_volume` by reason code, weekly.
- Alert: `dq_pass_rate` drops > 5pp week-over-week.

### 5.2 Postgres schema (Gold)

```
brand(id, name)
sku(id, brand_id, asin, model_number, category, launch_date)
aspect_ontology(id, version, code, label, severity_applicable)
review(id, source, source_id, sku_id, locale, language, lang_confidence,
       rating, verified, posted_at, helpful_count, text_raw, char_len,
       ingest_run_id, ingest_ts)
aspect_mention(id, review_id, aspect_id, sentiment, severity,
               evidence_quote, model_name, aspect_version, created_at)
cluster(id, sku_id, iso_week, label_llm, label_human, size,
        topic_keywords[], created_at)
review_cluster(review_id, cluster_id, similarity)
incident(id, sku_id, aspect_id, week_start, severity_score, z_score,
         baseline_volume, observed_volume, summary, status, opened_at)
ingest_run(id, source, started_at, completed_at, n_rows, status, dag_run_id)
dq_event(id, ingest_run_id, check_name, n_rows, n_failed, reason_codes jsonb)
```

---

## 6. ABSA / NLP Pipeline

### 6.1 Aspect ontology v1 (closed, MVP)

| code | label | applies_to |
|---|---|---|
| `battery` | Battery life / capacity / charging cycles | chargers, power banks, headphones |
| `charging` | Charging speed, USB-C compat, cables | chargers, cables, power banks |
| `overheating` | Device gets too hot in use | chargers, power banks, headphones |
| `sound_quality` | Audio fidelity, volume, ANC | headphones, speakers, earbuds |
| `bluetooth` | Pairing, drops, range, latency | headphones, speakers, earbuds |
| `delivery` | Shipping, packaging, missing items | all |
| `price` | Value for money, price-vs-competitor | all |

Stored in `aspect_ontology` with a `version` column. Ontology bumps trigger a backfill flow.

### 6.2 LLM tool-use contract

Primary LLM: **Claude / OpenAI models** for structured ABSA (called as tool-use with strict JSON schema).
Local LLM (target): **Qwen-family model served via vLLM** for low-cost batch inference at scale.

```json
{
  "name": "extract_aspects",
  "input_schema": {
    "type": "object",
    "required": ["aspects"],
    "properties": {
      "aspects": {
        "type": "array",
        "maxItems": 8,
        "items": {
          "type": "object",
          "required": ["aspect_code", "sentiment", "evidence_quote"],
          "properties": {
            "aspect_code": {"type": "string", "enum": ["battery","charging","overheating","sound_quality","bluetooth","delivery","price"]},
            "sentiment": {"type": "string", "enum": ["positive","neutral","negative"]},
            "severity": {"type": "string", "enum": ["low","medium","high"]},
            "evidence_quote": {"type": "string"}
          }
        }
      }
    }
  }
}
```

Validations applied after the LLM call (in `dq_flow` for ABSA outputs):
- JSON-schema validation.
- `evidence_quote` must be a verbatim substring of `review.text_raw`.
- `severity` is required iff `sentiment == "negative"`.
- Per-review aspect codes must be unique.

### 6.3 Severity rubric (in prompt)

- `high` — safety/fire/dangerous wording, refused refund, hospital, smoke, burn.
- `medium` — product unusable, returned, broken, never worked.
- `low` — minor complaint, mild annoyance, "could be better".

---

## 7. Clustering & Anomaly Detection

### 7.1 BERTopic clustering

- Scope: only `sentiment = "negative"` aspect-mentions.
- Granularity: per `(sku_id, iso_week)`.
- Embedding: bge-m3 dense.
- UMAP → HDBSCAN.
- LLM-generated cluster labels (`label_llm`), human can override (`label_human`).
- Output: `cluster` + `review_cluster` rows.

### 7.2 EWMA anomaly

Per `(sku_id, aspect_id, iso_week)`:

```
complaint_volume_w     = count of negative aspect_mentions
baseline_w             = EWMA over the last 8 weeks (span=8)
std_w                  = rolling 8-week std
z_score_w              = (complaint_volume_w - baseline_w) / std_w
```

Flag `incident` when `z_score_w ≥ 2.5` AND `complaint_volume_w ≥ 5` (volume guardrail prevents spurious flags on tiny SKUs).

`severity_score = z_score * mean(severity_weight)` where severity weights = `{low: 1, medium: 2, high: 4}`.

Backtested on a hand-built set of 12 historical "known events" (verified by reading review spikes) — target P@10 ≥ 0.6.

---

## 8. RAG Query Layer

### 8.1 Four locked question templates (MVP)

| # | Template | Behavior |
|---|---|---|
| Q1 | "Biggest complaint on `<SKU>` recently?" | self-query → filter sku + negative + last 30d → cluster + count → narrate top cluster with quotes |
| Q2 | "Has `<aspect>` been rising over the last 30 days for `<SKU>`?" | SQL trend over `aspect_mention` weekly counts → phrasing + delta + 3 citations |
| Q3 | "What competitors do users mention when leaving negative reviews of `<SKU>`?" | retrieve negatives → entity-extract over a closed brand list → rank + quote |
| Q4 | "Quote N original reviews about `<topic>` for `<SKU>`." | hybrid search → return N citations with verbatim text |

### 8.2 Stack

- **Qdrant hybrid**: BM25 + bge-m3 dense, RRF fusion, top 50.
- **Reranker**: off-the-shelf `bge-reranker-v2-m3` (no fine-tune in MVP), top 8 to the LLM.
- **Self-query**: LLM extracts `{sku_id, aspect_code, date_range, sentiment}` filters from NL.
- **LangGraph 3-node**: `planner` → `retrieve_or_sql` (branches on Q1–Q4) → `respond_with_citations`.
- **No critique node in MVP** — added post-MVP if eval shows it's worth the latency.

### 8.3 Citation contract

Every claim in the response is bound to a `Citation(review_id, span_start, span_end)`. The dashboard renders inline footnotes that open the underlying review with the cited span highlighted.

---

## 9. Evaluation

Three layers, all CI-integrated. Full detail in [`design/03-eval-and-observability.md`](design/03-eval-and-observability.md). MVP scope:

### 9.1 Pipeline ML eval (offline, deterministic)

| Component | Metric | MVP gate |
|---|---|---|
| Language ID | Accuracy macro | ≥ 0.98 |
| ABSA aspect extraction | Macro-F1 vs. 200-review hand-labeled holdout | ≥ 0.65 |
| ABSA sentiment | Cohen's κ vs. labeler | ≥ 0.60 |
| ABSA evidence verbatim | % rows where `evidence_quote` is verbatim substring | ≥ 0.98 |
| Anomaly detector | P@10 on 12 known historical events | ≥ 0.6 |

### 9.2 Retrieval eval

- 50 hand-built `(question, gold_review_ids[])` pairs across the 4 templates.
- Metrics: Recall@5, Recall@20, MRR@10.
- MVP gate: Recall@5 ≥ 0.80.

### 9.3 Agent eval (smoke)

- 20 scenarios across Q1–Q4.
- LLM-as-judge rubric: correctness, citation faithfulness, completeness.
- MVP gate: ≥ 80% of scenarios scored correctness ≥ 3/5.

All eval results land in the README at the end of week 5 (replacing the placeholders in §12).

---

## 10. Dashboard (Streamlit)

5 pages:

1. **SKU Overview** — picker (brand → SKU); weekly complaint volume with EWMA band; incident markers.
2. **Aspect Distribution** — stacked bar of aspect × sentiment for selected SKU, last 90 days.
3. **Top Emerging Issues** — `incident` rows sorted by `severity_score`; click expands cluster reviews.
4. **Chat** — Q1–Q4 with streamed response and clickable inline citations.
5. **Anomaly Alerts** — list view of incidents with `posted_at`, `aspect`, `severity`, `z_score`, LLM-narrated summary.

A 6th page, **Data Quality**, shows DQ pass rates per check and dead-letter volumes by reason code — directly demonstrates the data-engineering surface.

---

## 11. Target Architecture / Post-MVP

The MVP is a deliberate slice of a larger design. The pieces below are **planned, ADR-documented, not built**. They are listed so the architecture story is intact for an interview, not as resume claims.

```
                     Users / Slack / Email digests
                              │
                    Next.js (AG-UI) + FastAPI
                              │
        LangGraph supervisor (single agent, fan-out only on cross-domain)
        ├── planner → retrieve / SQL / chart / action (HITL gated)
        ├── critique node (auto-retry on faithfulness failure)
        └── Mem0 memory layer (short-term / episodic / semantic / procedural)
                              │
        ┌─────────────────────┼─────────────────────────────┐
        │                     │                             │
   Postgres (OLTP)     Qdrant per locale            ClickHouse (events)
   + pgvector           (en, de, ja, ...)            review_facts,
   (Mem0)               hybrid + reranker (fine-     aspect_volume MV
                        tuned via LoRA on hard
                        negatives mined in-domain)
                              │
                     Offline Prefect DAG
       ingest (Amazon Reviews + mock SP-API / Shopify / TikTok Shop +
               Reddit + Trustpilot + YouTube transcripts)
       → dq → normalize → translate (Qwen-family on vLLM) → ABSA
       → embed → cluster → anomaly → reindex

       Observability:  Langfuse + OpenTelemetry → Grafana
       Eval:           Promptfoo + RAGAS + LLM-judge + nightly trace replay
       Deploy:         Docker → K8s (Helm) + KEDA + Terraform
```

Post-MVP backlog (in priority order):

1. Add DE / JP locales with translation-augmented embeddings (ADR-003).
2. Move event-grain analytics to ClickHouse; add hourly MV refresh.
3. Mem0 3-tier memory in the agent.
4. Critique node + max-2-retry loop.
5. HITL action tools with LangGraph `interrupt`/`resume`.
6. Reranker LoRA fine-tune case study (hard-negative mining).
7. Cost router across LLM providers; local Qwen-family inference on vLLM for bulk ABSA.
8. Langfuse + OTel; replace LangSmith-only setup.
9. Next.js + AG-UI frontend replacing Streamlit.
10. Mock SP-API / Shopify / TikTok Shop integration surface.
11. K8s + Helm + Terraform.

Each item has an ADR or design note pointing to the rationale.

---

## 12. Resume Bullets

**MVP-honest** (numbers filled at end of week 5 from real eval runs):

- Built **VoiceLens**, a VoC data-engineering platform over **200k+ Amazon reviews** of consumer-electronics brands, with a Prefect-orchestrated medallion pipeline (Bronze → Silver → Gold) into Postgres + Qdrant.
- Implemented an automated **Data Quality** layer (duplicate detection, SKU/ASIN validation, rating-range check, language-confidence threshold, empty/spam filter, ABSA schema + verbatim-quote validation) with dead-letter routing and per-check pass-rate metrics surfaced to dashboards.
- Built an **LLM-based ABSA pipeline** (Claude / OpenAI structured outputs) against a 7-aspect ontology versioned in Postgres; **macro-F1 \<X\>** on a 200-review hand-labeled holdout; bulk-tagged 200k+ reviews.
- Engineered **anomaly detection** with EWMA + z-score on per-{SKU, aspect, week} complaint volume, severity-weighted; P@10 = **\<P\>** on 12 known historical events in backtest.
- Built **agentic-RAG query layer** with hybrid retrieval (Qdrant BM25 + bge-m3) + reranker + self-query filter extraction; **Recall@5 = \<R\>** on 50 hand-built pairs; LangGraph 3-node supervisor producing inline-citation responses.
- Shipped Streamlit dashboard (SKU trend, aspect distribution, emerging issues, chat, alerts, **DQ pass-rate page**); full local reproduction via `docker compose up`.

**Aspirational bullets** (only after corresponding post-MVP items in §11 are built — do not use on resume yet):

- Extended ingestion to DE / JP locales with translation-augmented multilingual embeddings.
- Added 3-tier memory (Mem0 / pgvector) reducing repeat clarifying-question rate by N%.
- LoRA-fine-tuned bge-reranker on in-domain hard negatives, lifting NDCG@10 by +Xpp.
- Added critique + auto-retry node lifting RAGAS faithfulness from A → B.
- Cost router across Claude / OpenAI / Qwen-on-vLLM reducing $/query by N%.

---

## 13. Status / Scope Honesty

- ✅ Target architecture designed (this README §11 + `design/` ADRs).
- 🚧 **4–6 week MVP in flight** — scope locked in [`design/04-mvp-spec.md`](design/04-mvp-spec.md). English-only, Amazon-only, 3–5 brands, 200k–500k reviews, Streamlit dashboard, no HITL writes, no Mem0, no critique node.
- 📋 Post-MVP backlog (§11) is **designed, not built**.
- ❌ Not connected to a real seller account — uses public datasets only (Amazon Reviews 2023, McAuley Lab). Architecture preserves the integration surface a real deployment would need (SP-API auth, throttling, idempotency) so a future port is contained.

---

## 14. How to run (when implemented)

```bash
# infra
docker compose -f ops/docker/compose.yaml up -d   # postgres + qdrant + prefect

# data
make seed                                         # downloads & filters Amazon Reviews 2023 subset

# run offline pipeline
prefect deploy pipeline/flows/full_pipeline.py
prefect deployment run "full-pipeline/mvp"

# API + dashboard
uvicorn voicelens.api.main:app --reload
streamlit run voicelens/ui/app.py

# ask a question
curl -N -X POST localhost:8000/chat -d '{"q": "Top emerging complaints on PowerCore 24K this month"}'
```

---

## 15. References

- Shopify — *Building Production-Ready Agentic Systems* (Sidekick, ICML 2025). [Shopify Engineering](https://shopify.engineering/building-production-ready-agentic-systems)
- Amazon Science — *The technology behind Rufus*. [amazon.science](https://www.amazon.science/blog/the-technology-behind-amazons-genai-powered-shopping-assistant-rufus)
- LangGraph 1.0 (Oct 2025) — stateful agents, checkpointing, HITL primitives.
- BGE-M3 / bge-reranker-v2-m3 — multilingual hybrid retrieval.
- BERTopic — modular topic modeling.
- Amazon Reviews 2023 dataset (McAuley Lab, UCSD).
- RAGAS — retrieval-aware evaluation.

---

## License

MIT (intended) — code under MIT, data subject to source licenses.
