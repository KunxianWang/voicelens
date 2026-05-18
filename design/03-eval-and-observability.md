# Evaluation & Observability

The strongest interview surface for a Data/ML hire. Three orthogonal eval surfaces, all CI-integrated.

## 1. Pipeline ML eval (deterministic, offline)

Run per-PR via GitHub Actions on a 10k-review fixture.

| Component | Dataset | Metric | Gate |
|---|---|---|---|
| Language ID | FLORES-200 mini + 2k internal | Accuracy macro | ≥ 0.98 |
| Translation | FLORES devtest sample | COMET-22 | Report; no gate |
| ABSA aspect | 500 reviews, 2 labelers, IAA Cohen's κ | Macro-F1 | ≥ 0.70, no >2pp regression |
| ABSA sentiment | same | Cohen's κ vs. labelers | ≥ 0.65 |
| Topic clustering | weekly snapshot | c_v coherence; human spot-check on 20 clusters | report |
| Anomaly | 12 known historical incidents | P@10, Recall@known | P@10 ≥ 0.6 |

## 2. Retrieval eval

`eval/golden/retrieval.yaml` — 200 (question, gold_review_ids[]) pairs across locales.

Metrics:
- Recall@5, Recall@20
- MRR@10
- NDCG@10
- Filter precision: did self-query produce correct `sku_id` / `locale` / `date_range` filters?

CI gate: nightly run; no >2pp regression on Recall@5.

## 3. End-to-end agent eval

`eval/golden/scenarios.yaml` — 200+ analyst questions, tagged by capability.

Schema:
```yaml
- id: scn-073
  capabilities: [rag, cross-locale, citation]
  question: "Compare overheating complaints on PowerCore 24K between DE and JP last 30 days."
  expectations:
    must_filter:
      sku_model: PowerCore-24K
      locales: [de-DE, ja-JP]
      window_days: 30
    must_cite_at_least: 5
    aspect_codes_mentioned: [battery.overheat]
    forbidden_phrases: ["I cannot", "as an AI"]
  judge_rubric: standard_v3
```

Driven by three layers:

### 3a — Promptfoo (deterministic assertions)
Runs nightly. Checks filter extraction, citation count, forbidden phrases, latency, cost.

### 3b — RAGAS (LLM-automated retrieval metrics)
- Faithfulness
- Answer relevance
- Context precision / recall

### 3c — LLM-as-judge (Shopify Sidekick pattern)
Rubric (versioned `judge_rubric: standard_v3`):
- Correctness (0–5)
- Citation faithfulness (every claim has a real citation) (0–5)
- Completeness vs. plan (0–5)
- Conciseness (0–5)
- Tool-use accuracy (binary)

Judge is Claude Opus 4.7 with prompt cache; calibration set of 50 hand-scored examples kept; we report per-rubric inter-judge κ vs. human (target ≥ 0.55).

### 3d — Trace replay (online → offline)
- 1% of prod traces sampled nightly, replayed against the candidate prompt/model.
- Diff against original; aggregate by capability tag.
- Used as a regression bar before promoting prompts.

## 4. Cost / latency budget

Per-query budget tracked in Langfuse:

| segment | p50 | p95 | p99 |
|---|---|---|---|
| sql-only | 1.2s | 3s | 6s |
| rag-only | 2.5s | 7s | 12s |
| mixed | 3.2s | 9s | 15s |

Alert: rolling-mean p95 > 1.5x budget for 30 min.

Cost: target `$0.012 / query`. Hard cap per-trace at `$0.10` (kills runaway plans).

## 5. Observability stack

- **Langfuse** (LLM-native):
  - Trace tree per request
  - Per-node cost/latency/tokens
  - Prompt versions tagged
  - A/B prompt experiments
- **OpenTelemetry → Grafana / Tempo**:
  - Service-level (FastAPI, Qdrant, Postgres, ClickHouse, vLLM)
  - Histograms: e2e, retrieval, rerank, LLM, sql
- **Sentry** for exceptions
- **Prefect UI** for ETL DAG state

## 6. Drift monitors

- Aspect distribution KL divergence week-over-week per SKU/locale. Sudden divergence triggers re-cluster + ontology review.
- Embedding drift: random sample 1k reviews/week, recompute, monitor distance to historical centroids.
- Prompt eval scores week-over-week per capability — silent degradation tripwire.

## 7. CI/CD eval gates

```
PR pipeline:
  unit + lint
  pipeline ML eval on fixture
  retrieval eval (fast subset, 50 q)
  agent eval (smoke set, 20 scenarios)
  ─────────► merge

Nightly:
  full retrieval eval (200)
  full agent eval (200) including LLM-judge
  trace replay (1% sample)
  cost/latency report
  ─────────► Slack digest + dashboards
```

## 8. What gets shipped in the Resume

- One labelled chart showing Recall@5 / NDCG@10 lifts across the retrieval iterations.
- One LLM-judge dashboard screenshot with per-capability scores.
- A blog-post-style writeup of the reranker fine-tuning case study.
- A 90-second Loom demoing analyst Q&A with streaming + citations + HITL action.
