# MVP Spec — 4-to-6 Week Buildable Slice

This is the **locked, narrowed** scope. The earlier 12-week roadmap in README §13 is replaced by what's below. Anything not listed here is **explicitly post-MVP**.

---

## 1. Goals of the MVP

A demoable, recruiter-shareable slice that proves:
1. A real multi-stage **data + ML pipeline** (ingest → ABSA → cluster → anomaly).
2. A real **agentic-RAG** query layer with citations.
3. A real **eval surface** with numbers in the README.
4. A clickable **dashboard** that an analyst persona can use.

Out of scope for MVP: multi-locale (DE/JP), multi-channel beyond Amazon, listing-ops agent, support agent, ads monitor, HITL action tools, reranker fine-tune case study, k8s deploy.

---

## 2. Data scope

### Source
- **Amazon Reviews 2023** (McAuley Lab, UCSD), `Electronics` category only.

### Brand filter
- **3–5 brands**, English only.
- Recommended set (all have enough reviews + are CE-out-of-China-adjacent for narrative fit):

| Brand | Why |
|---|---|
| Anker | Hero brand for the story |
| Soundcore | Audio sub-brand — yields `sound_quality` / `bluetooth` aspects |
| RAVPower or Aukey | Direct competitor — needed for "competitor mentions" question |
| Bose **or** JBL | Premium competitor in audio — for cross-brand quoting |
| (optional) UGREEN | Another China-out competitor |

If a real brand doesn't have enough data after filtering, we substitute with a **virtual brand** that re-labels another seller's reviews — documented as synthetic in the README.

### Volume target
- ~200k–500k reviews after filter. Small enough to fit on a laptop SSD, large enough to make ABSA + anomaly non-trivial.

### Language
- **English only** (`en-US`) in MVP. The schema already has `locale`/`language` columns so adding DE/JP later is non-breaking.

---

## 3. Pipeline (must-build)

```
[Amazon Reviews 2023 jsonl on disk / S3]
        │
        ▼
   ingest_flow
        │  - read & filter brand allowlist
        │  - drop rows missing rating/text/asin
        │  - write bronze parquet partitioned by brand/month
        ▼
   dq_flow                 ─── §3.1 Data Quality gates
        │  - duplicate review_id detection
        │  - missing SKU / ASIN validation
        │  - invalid rating range
        │  - language confidence threshold
        │  - empty / spam text
        │  - dead-letter routing + metrics
        ▼
   normalize_flow
        │  - schema-validate (pydantic)
        │  - basic PII redact (regex emails, phones)
        │  - write silver parquet
        │  - upsert into postgres.review
        ▼
   absa_flow
        │  - prompt LLM with constrained JSON schema
        │  - aspect ontology v1 (closed set, see §4)
        │  - sentiment + severity + evidence_quote
        │  - validate aspect_code ∈ ontology + verbatim quote
        │  - upsert into postgres.aspect_mention
        ▼
   embed_flow
        │  - bge-m3 dense + BM25 sparse
        │  - write to Qdrant (single collection: reviews_en)
        ▼
   cluster_flow
        │  - BERTopic over reviews tagged "negative" per SKU
        │  - per {sku_id, iso_week}
        │  - persist cluster + cluster_member
        ▼
   anomaly_flow
        │  - per {sku_id, aspect, week}: complaint_volume
        │  - EWMA(span=8 weeks), z-score
        │  - flag z > 2.5 + volume ≥ 5 → postgres.incident
```

### 3.1 Data Quality (DQ) module — must-build

| Check | Rule | Failure handling |
|---|---|---|
| Duplicate `review_id` | Hash on `(source, asin, reviewer_id, posted_at, sha1(text))` | Drop dups; keep earliest |
| Missing SKU / ASIN | `asin` non-null and resolvable in SKU map | Route to `dl_missing_sku` |
| Invalid rating | `rating ∈ [1, 5]` | Route to `dl_invalid_rating` |
| Language confidence | fastText `lang_confidence ≥ 0.7`, `language == "en"` for MVP | Route to `dl_low_lang_conf` |
| Empty / spam | `len(text.strip()) ≥ 20`; URLs-only / ALL-CAPS-only filtered | Route to `dl_empty_or_spam` |
| ABSA schema | Valid JSON; `aspect_code ∈ ontology_v1`; `evidence_quote` verbatim substring; `severity` set iff `sentiment == negative` | Retry once; then `dl_absa_invalid` |

Per-flow DQ metrics persisted to `dq_event` and surfaced in:
- Prefect run dashboard
- Streamlit "Data Quality" page (page 6)
- Daily console digest

Alerting: `dq_pass_rate` drop > 5pp week-over-week → log warning (no paging in MVP).

### Orchestrator
- **Prefect 3**, single Docker container.
- Each flow is independently runnable + idempotent.

### Storage in MVP
- **Postgres** for everything analytical (review, aspect_mention, cluster, incident).
- **Qdrant** for vectors.
- **S3 (or local minio)** for bronze/silver parquet.
- **No ClickHouse** in MVP — Postgres can serve dashboard queries at this volume. (Mentioned in README §3 as the path forward; not built now.)

---

## 4. ABSA — locked aspect ontology v1 (closed set)

| code | label | applies_to |
|---|---|---|
| `battery` | Battery life / capacity / charging cycles | chargers, power banks, headphones |
| `charging` | Charging speed, USB-C compatibility, cable | chargers, cables, power banks |
| `overheating` | Device gets too hot in use | chargers, power banks, headphones |
| `sound_quality` | Audio fidelity, volume, distortion, ANC | headphones, speakers, earbuds |
| `bluetooth` | Bluetooth pairing, drops, range, latency | headphones, speakers, earbuds |
| `delivery` | Shipping speed, packaging damage, missing items | all |
| `price` | Value for money, price-vs-competitor | all |

### Output contract (LLM tool-use)

```json
{
  "aspects": [
    {
      "aspect_code": "overheating",
      "sentiment": "negative",
      "severity": "high",
      "evidence_quote": "After 20 minutes it was too hot to hold."
    }
  ]
}
```

- `aspect_code` ∈ closed set above (anything else → drop).
- `sentiment` ∈ {positive, neutral, negative}.
- `severity` ∈ {low, medium, high}, only set when `sentiment = negative`.
- `evidence_quote` must be a **verbatim substring** of the review (validated; failures retried once, then dropped to dead-letter).

### Severity rubric (in prompt)
- `high` = safety, fire, dangerous, refused refund, smoke, burned, hospital
- `medium` = product unusable, returned, never worked, broken
- `low` = minor complaint, mild annoyance, "could be better"

### LLM choice in MVP
- A **small/fast hosted LLM** (Claude Haiku-class or OpenAI mini-class) for bulk ABSA — picked at implementation time based on current availability and pricing. Target cost ≤ $0.001/review at our prompt size.
- **Skip self-hosted local LLM** in MVP to keep ops surface small — local Qwen-family on vLLM moves into post-MVP when bulk-translation or higher volume justifies it.

---

## 5. RAG — locked question types

These four must work end-to-end with citations:

| # | Template question | What the agent must do |
|---|---|---|
| Q1 | "What's the biggest complaint on `<SKU>` recently?" | self-query → filter sku + sentiment=negative + last 30d → cluster + count → narrate top cluster with quotes |
| Q2 | "Has `<aspect>` been rising over the last 30 days for `<SKU>`?" | SQL on `aspect_mention` weekly counts → trend phrasing + delta + cite top 3 reviews |
| Q3 | "What competitors do users mention when leaving negative reviews of `<SKU>`?" | retrieve negatives for SKU → entity-extract brand names (LLM, constrained list of known brands) → rank by mention count → quote |
| Q4 | "Quote 5 original reviews about `<topic>` for `<SKU>`." | hybrid search → return 5 citations with original text + source URL stub |

### Stack
- Qdrant hybrid (BM25 + bge-m3).
- bge-reranker-v2-m3 (off-the-shelf, no fine-tune in MVP).
- LangGraph supervisor with three nodes: `planner` → `retrieve_or_sql` → `respond_with_citations`. No critique node in MVP (added post-MVP).
- Self-query node uses Claude Haiku to extract `{sku_id, aspect_code, date_range, sentiment}` filters.

---

## 6. Dashboard (Next.js or Streamlit — pick one)

**Recommendation: Streamlit for MVP.** Faster to ship, sufficient for the demo. Switch to Next.js post-MVP once feature surface justifies it.

### Pages

1. **SKU Overview**
   - Picker: brand → SKU.
   - Complaint volume time series (weekly) with EWMA band overlay.
   - Anomaly markers where `incident` rows exist.
2. **Aspect Distribution**
   - Stacked bar of aspect × sentiment for selected SKU, last 90 days.
3. **Top Emerging Issues**
   - Table of `incident` rows, sorted by `severity_score`. Click → expands to cluster reviews.
4. **Chat**
   - Box for Q1–Q4. Streams response. Inline citations clickable → opens review modal.
5. **Anomaly Alerts**
   - List of incidents with `posted_at`, `aspect`, `severity`, `z_score`, narrative.

---

## 7. Weekly breakdown (5-week target, 1-week buffer)

| Wk | Goal | Concrete done-when |
|---|---|---|
| **1** | Scaffold + ingest + DQ | `docker compose up` boots Qdrant + Postgres + Prefect. Amazon Reviews 2023 Electronics subset downloaded & filtered. `ingest_flow` writes bronze parquet. `dq_flow` implements all six DQ checks with dead-letter routing and `dq_event` metrics. Project lints + CI skeleton. |
| **2** | Normalize + Postgres + ABSA v1 on 10k sample | Silver parquet → Postgres. ABSA on 10k reviews with macro-F1 reported on 200-review hand-labeled holdout (user labels these). Aspect ontology v1 in `aspect_ontology` table. ABSA schema validation runs through `dq_flow`. |
| **3** | Full ABSA backfill + embeddings + Qdrant | ABSA over full 200k–500k reviews. Embeddings + payload filters in Qdrant. Retrieval eval (50 hand-built `(question, gold_ids)` pairs) reports Recall@5. |
| **4** | Clustering + anomaly + LangGraph RAG | BERTopic per SKU+week on negatives. EWMA anomaly → `incident`. LangGraph 3-node agent answering Q1–Q4 with citations end-to-end. |
| **5** | Streamlit dashboard + eval writeup + README polish | 5 dashboard pages working. README updated with: actual numbers from eval, screenshots, 90-second Loom demo link, resume bullets calibrated to what was actually built. |
| **6 (buffer)** | Polish, demo recording, blog post (optional) | Loom recorded. Optional: blog post draft on "What I learned about ABSA at portfolio scale." |

---

## 8. Definition of Done for MVP

Hard gates — without these the MVP is not done:

- [ ] Data: ≥ 200k reviews loaded into Postgres, 100% with at least one `aspect_mention` row (or explicitly tagged as "no aspect detected").
- [ ] Data Quality: all 6 DQ checks implemented with dead-letter routing; per-check pass-rate metrics visible in dashboard page 6.
- [ ] ABSA: macro-F1 ≥ 0.65 on the 200-review hand-labeled holdout. Numbers in README.
- [ ] Retrieval: Recall@5 ≥ 0.80 on 50 hand-built pairs. Number in README.
- [ ] Agent: all four Q-templates pass on a 20-question smoke set with ≥ 80% LLM-judge "correctness ≥ 3/5". Number in README.
- [ ] Dashboard: 5 main pages + DQ page render real data, chat streams, citations open the underlying review.
- [ ] Repo: `docker compose up && make seed && make demo` produces a working local instance from scratch in ≤ 10 minutes on a laptop.
- [ ] Loom: ≤ 90s demo recorded and linked in README.

---

## 9. Resume bullets — MVP-honest version

(Use these once the MVP DoD is met. Replace the longer set in README §14 with this on shipping.)

- Built **VoiceLens**, a VoC data-engineering platform over 200k+ Amazon reviews of consumer-electronics brands, with Prefect-orchestrated medallion pipeline (Bronze → Silver → Gold) into Postgres + Qdrant.
- Implemented automated **Data Quality** layer (duplicate detection, SKU/ASIN validation, rating-range, language-confidence threshold, empty/spam filter, ABSA schema + verbatim-quote validation) with dead-letter routing and per-check pass-rate metrics.
- Built **LLM-based ABSA** (structured JSON outputs) against a 7-aspect ontology versioned in Postgres; **macro-F1 \<X\>** on 200-review hand-labeled holdout; bulk-tagged 200k+ reviews at \<$Y\>/1k.
- Built **anomaly pipeline** (EWMA z-score on per-{SKU, aspect, week} complaint volume, severity-weighted) flagging **\<N\>** quality incidents with **P@10 = \<P\>** on 12 known historical events.
- Engineered **agentic-RAG** with Qdrant hybrid (BM25 + bge-m3) + reranker + self-query filter extraction; **Recall@5 = \<R\>** on 50 hand-built pairs; LangGraph 3-node supervisor producing inline-citation responses.
- Shipped Streamlit dashboard (SKU trend, aspect distribution, emerging issues, chat, alerts, **DQ pass-rate page**); full local repro via `docker compose up`.

> X/Y/N/P/R/Z are filled in at the end of week 5 from actual numbers.

---

## 10. Explicit post-MVP backlog (do NOT do during MVP)

- Multi-locale (DE/JP) data + translation-augmented embeddings.
- Mock SP-API / Shopify / TikTok Shop integration surface.
- Mem0 3-tier memory (MVP uses LangGraph state only).
- Critique node + auto-retry.
- HITL action tools (Jira draft, Slack post, email factory).
- Reranker LoRA fine-tune case study.
- ClickHouse, K8s, Terraform, Helm.
- Cost router across Claude / GPT / Qwen.
- Next.js frontend with AG-UI.
- Langfuse + OTel (MVP uses LangSmith only).

These are documented in README so the architecture story is intact; they are simply **not built** in the MVP.
