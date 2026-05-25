# VoiceLens — VoC Data Engineering & Analytics Platform for Cross-Border E-commerce


## 1. Project Summary

VoiceLens is a **data-engineering-first** VoC analytics platform for cross-border e-commerce. It is an end-to-end pipeline — **ingestion → data-quality → ABSA → retrieval → clustering → anomaly detection → dashboard** — that turns raw consumer-electronics reviews into governed, queryable, aspect-labelled facts and surfaces emerging quality issues. A citation-grounded RAG layer and a controlled LangGraph router are implemented on top; the full autonomous agent architecture remains a post-MVP extension.

**Implemented stack:** Python 3.11 · Prefect 3 · Postgres · Qdrant · scikit-learn (TF-IDF + KMeans) · BM25 + bge dense hybrid retrieval · Claude (structured ABSA) · LangGraph · Streamlit · Docker · pytest / ruff

**Planned stack (post-MVP, designed not built):** autonomous agent memory/actions · FastAPI · BERTopic/HDBSCAN · reranker · ClickHouse · Mem0 · Next.js · Langfuse · OpenTelemetry · K8s · vLLM

---

## Current MVP Status

> Snapshot as of **Milestone 5B**. Every number below is from a real pipeline run on the working dataset — no projections. See [`docs/demo.md`](docs/demo.md) for the live demo walkthrough.

### Implemented vs planned

| Stage | Status | Notes |
|---|---|---|
| Ingestion + Data Quality | Implemented | Deterministic MVP subset; DQ gates with dead-letter routing |
| LLM ABSA (8-aspect ontology v2) | Implemented + eval + 10k coverage | Ran on a **10,000-review real Claude batch**; the full 245k pass is not done |
| Embedding + Qdrant index | Implemented | ABSA-backed review evidence indexed |
| Hybrid retrieval + evaluation | Implemented | BM25 + dense, RRF fusion; refined-golden eval harness |
| Issue clustering | Implemented | TF-IDF + KMeans over negative aspect mentions |
| Anomaly detection | Implemented | EWMA z-score; cluster-level + aspect-level fallback |
| Streamlit dashboard | Implemented | 7 pages — see §10 |
| Citation-grounded answer generation | Implemented | M6A — retrieve-then-answer with citation guardrails |
| LangGraph router workflow | Implemented | M6B — deterministic route → one tool; single pass |
| Autonomous agent (memory, actions, planning) | Designed, not built | §8 / §11 — not an autonomous system |

### Real metrics

**Data ingestion & quality**
- **43.88M** Amazon review rows + **1.61M** product-metadata rows streamed to resolve target consumer-electronics brands via metadata-based brand resolution.
- **253.5k**-review deterministic MVP subset built (reproducible by seed).
- **245,961** valid MVP reviews loaded into Postgres after DQ — **97.03%** DQ pass rate, failures bucketed by reason code.

**ABSA (10k batch — full corpus pass not yet run)**
- **10,000**-review real Claude ABSA batch under the 8-aspect ontology **v2** (`battery`, `charging`, `overheating`, `sound_quality`, `bluetooth`, `delivery`, `price`, `reliability/durability`).
- Holdout eval: **macro-F1 0.7074**, **Cohen's κ 0.9336**.

**Retrieval**
- ABSA-backed review evidence indexed in Qdrant (`reviews_v2`).
- Refined-golden eval: **Hit@5 0.959**, **Recall@20 0.886**, **MRR@10 0.849**, filter precision@10 1.00.

**Analytics**
- **11,335** negative aspect mentions clustered into **44** issue clusters.
- **237** anomaly incidents generated from weekly volume, EWMA baseline, rolling std and z-score.

**Engineering**
- **350** automated tests passing; `ruff` lint clean.

> **Honesty note:** ABSA has been validated end-to-end but only *run* on a 10,000-review batch — VoiceLens does **not** claim full 245k ABSA coverage. Scaling that batch is a budgeted follow-up, not a code change.

---

## Demo Walkthrough

Full step-by-step script (with talking points) lives in [`docs/demo.md`](docs/demo.md). Short version:

```bash
make up                # 1. start Postgres + Qdrant
make init-db           #    (only if schema not yet created)
make demo-healthcheck  # 2. confirm infra is demo-ready  -> "READY TO DEMO"
make dashboard         # 3. launch the Streamlit dashboard
```

Then, in the dashboard, move through the seven pages in order:

1. **Overview** — pipeline at a glance: review / ABSA / cluster / incident counts and distribution charts.
2. **ABSA Distribution** — aspect × sentiment matrix, severity breakdown, and the **reliability spotlight** (the dominant negative signal).
3. **Issue Clusters** — clusters ranked by severity-weighted size, drill-down into member reviews.
4. **Emerging Incidents** — EWMA-detected spikes; toggle cluster-level vs aspect-level granularity.
5. **Retrieval Search** — hybrid search demo plus a beta citation-grounded answer generator.
6. **Agent Q&A** — ask a question; a deterministic LangGraph router picks retrieval / incident / analytics and shows the chosen route.
7. **Data Quality** — DQ pass/fail by reason, ABSA processing-reliability rates.

The retrieval and agent pages cover answer generation and routing; a full autonomous agent (memory, actions, planning) is not built.

---

## Agentic Workflow

VoiceLens ships a **lightweight LangGraph router** (M6B), not an autonomous multi-agent system. One question makes one routing decision and runs one tool — there is no memory, no autonomous actions, no multi-agent collaboration, and no multi-step planning.

A deterministic keyword router sends each question to one of four routes:

| Route | Handles | Grounding |
|---|---|---|
| `retrieval_answer` | complaints, quotes, "what customers say" about an aspect | **citation-grounded** — every claim cites a retrieved review |
| `analytics_summary` | counts, distributions, "top aspect", "most common" | **database-grounded** — aggregate ABSA stats from Postgres |
| `incident_summary` | spikes, anomalies, emerging issues | **database-grounded** — the anomaly `incident` table |
| `insufficient_scope` | anything outside VoC analytics | a safe refusal — no hallucinated answer |

Retrieval answers are produced by the M6A citation-grounded generator (unsupported citation markers are stripped; unanswerable queries return "insufficient evidence"). Analytics and incident answers are read directly from the database. The workflow is evaluated by `make evaluate-agent` against a 20-question golden set (route accuracy, citation rate, refusal rate). The full autonomous agent — planner, memory, tool-use, HITL — remains designed-only (§8, §11).

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
| ABSA | Closed 8-aspect ontology: `battery`, `charging`, `overheating`, `sound_quality`, `bluetooth`, `delivery`, `price`, `reliability` / durability. Sentiment ∈ {pos, neu, neg}. Severity ∈ {low, med, high}. Verbatim `evidence_quote`. |
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

### 6.1 Aspect ontology v2 (closed, MVP)

| code | label | applies_to |
|---|---|---|
| `battery` | Battery life / capacity / charging cycles | chargers, power banks, headphones |
| `charging` | Charging speed, USB-C compat, cables | chargers, cables, power banks |
| `overheating` | Device gets too hot in use | chargers, power banks, headphones |
| `sound_quality` | Audio fidelity, volume, ANC | headphones, speakers, earbuds |
| `bluetooth` | Pairing, drops, range, latency | headphones, speakers, earbuds |
| `delivery` | Shipping, packaging, missing items | all |
| `price` | Value for money, price-vs-competitor | all |
| `reliability` | Reliability / durability / general product failure | all |

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
            "aspect_code": {"type": "string", "enum": ["battery","charging","overheating","sound_quality","bluetooth","delivery","price","reliability"]},
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

### 7.1 Issue clustering

- Scope: only `sentiment = "negative"` aspect-mentions.
- Input scale: **11,335** negative aspect mentions.
- Algorithm: TF-IDF + KMeans per aspect, with BERTopic as an optional extension when installed.
- Ranking: cluster size + severity-weighted size.
- Output: `cluster` + `review_cluster` rows.
- Result: **44** issue clusters.

### 7.2 EWMA anomaly

Per `(sku_id, aspect_id, iso_week)`:

```
complaint_volume_w     = count of negative aspect_mentions
baseline_w             = EWMA over the last 8 weeks (span=8)
std_w                  = rolling 8-week std
z_score_w              = (complaint_volume_w - baseline_w) / std_w
```

Flag `incident` when the weekly observed volume exceeds the EWMA baseline by the configured z-score and volume thresholds.

`severity_score = z_score * mean(severity_weight)`.

Current run generated **237** anomaly incidents from weekly volume, EWMA baseline, rolling std and z-score.

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

The headline eval results are summarized in *Current MVP Status*.

---

## 10. Dashboard (Streamlit)

**Implemented** — 7 pages, launched with `make dashboard` (`voicelens/ui/`):

1. **Overview** — headline counts (reviews, ABSA-processed, mentions, clusters, incidents, Qdrant points) and distribution charts (brand, ABSA status, aspect, sentiment).
2. **ABSA Distribution** — aspect × sentiment matrix, severity distribution for negatives, negative examples with evidence quotes, and a reliability spotlight. Filterable by brand / aspect / sentiment / severity / rating.
3. **Issue Clusters** — M4A clusters ranked by severity-weighted size, with drill-down into keywords, representative quotes and member reviews.
4. **Emerging Incidents** — M4B anomaly incidents ranked by severity score, with the deterministic pipeline summary; filterable by aspect and granularity (cluster-level / aspect-level fallback).
5. **Retrieval Search** — hybrid / dense / lexical search over the Qdrant index, plus a beta citation-grounded answer generator (M6A).
6. **Agent Q&A** — ask a question; a deterministic LangGraph router (M6B) sends it to retrieval / incident / analytics and shows the route, answer and citations.
7. **Data Quality** — ingest-run summaries, DQ pass/fail by reason, ABSA status distribution, processed / mention coverage rates and LLM-batch failed/invalid rates.

The dashboard is **read-only** — it visualises existing pipeline outputs and never re-runs a flow. The M6B agent is a single deterministic routing pass; a full autonomous agent (memory, actions, multi-step planning) is **designed, not built** (§8, §11).

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

## 12. Status / Scope Honesty

- **MVP pipeline implemented** — ingestion → DQ → ABSA → embed → cluster → anomaly → dashboard. See *Current MVP Status* above for real metrics.
- **ABSA coverage is a 10,000-review batch**, not the full 245k loaded corpus — bounded deliberately to control LLM cost while proving the pipeline and eval harness. The full pass is a budgeted follow-up.
- **Citation-grounded answer generation + a LangGraph router** are built (M6A / M6B) — but this is a single deterministic routing pass, **not** an autonomous agent. No memory, no autonomous actions, no multi-step planning; the full agent in §8 / §11 is still designed-only.
- Post-MVP backlog (§11) is **designed, not built**.
- Not connected to a real seller account — uses public datasets only (Amazon Reviews 2023, McAuley Lab). Architecture preserves the integration surface a real deployment would need (SP-API auth, throttling, idempotency) so a future port is contained.

---

## 13. How to run

Full developer guide in [`README_DEV.md`](README_DEV.md); demo walkthrough in [`docs/demo.md`](docs/demo.md). Quickstart:

```bash
# 1. infra — Postgres + Qdrant
make up
make init-db                # only if the schema is not yet created

# 2. confirm the demo is ready
make demo-healthcheck        # -> "READY TO DEMO"

# 3. launch the dashboard
make dashboard               # streamlit run voicelens/ui/app.py

# checks
make test                    # 350 tests, SQLite in-memory
make lint                    # ruff
```

Rebuilding the pipeline data (ingestion → ABSA → embed → cluster → anomaly) is documented in `README_DEV.md` and `docs/demo.md` — re-running the ABSA batch consumes LLM budget, so the working dataset ships pre-built.

---

## 14. References

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
