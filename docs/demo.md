# VoiceLens — Demo Walkthrough

A 5–10 minute walkthrough of the VoiceLens MVP for a recruiter or
interviewer. Everything runs locally; nothing here calls a real LLM or
touches a live seller account.

---

## 1. Prerequisites

- **Docker** — for the Postgres + Qdrant containers.
- **Python 3.11** with the project installed:
  ```bash
  pip install -e ".[dev,retrieval,analytics,dashboard]"
  ```
- A populated database. The demo assumes the pipeline has already been
  run (the repo's working dataset: ~246k loaded reviews, a 1k real
  Claude ABSA batch, 1k Qdrant points, 44 clusters, 23 incidents). If
  you are starting from an empty DB, see *Rebuilding the demo data* at
  the bottom.
- API keys (if you ever re-run ABSA) live in a local `.env` only — it
  is git-ignored and never printed.

---

## 2. Start infra

```bash
make up                # Postgres + Qdrant containers
make init-db           # only if the schema is not yet created
make demo-healthcheck  # confirms Postgres / Qdrant / dashboard are ready
```

`make demo-healthcheck` should end with `--- READY TO DEMO ---`. If any
line says `FAIL`, fix that before going further (most often: Qdrant
container not started, or the 1k batch not yet indexed).

---

## 3. Launch the dashboard

```bash
make dashboard         # streamlit run voicelens/ui/app.py
```

Streamlit prints a local URL (default `http://localhost:8501`). Open it
in a browser. The left sidebar lists the six pages.

---

## 4. Page-by-page script

### Overview
Open here. Point out the headline metrics — total reviews,
ABSA-processed reviews, aspect mentions, clusters, incidents, Qdrant
points — and the four distribution charts (reviews by brand, ABSA
status, aspect distribution, sentiment distribution).

> *Talking point:* "This is the whole pipeline at a glance — ingestion
> through anomaly detection. The review count is the full loaded
> corpus; ABSA so far is a controlled 1k batch, not the full 246k."

### ABSA Distribution
Show the aspect × sentiment matrix, then the severity distribution for
negative mentions. Scroll to the **Reliability spotlight** — reliability
is the dominant negative-quality signal in the 1k batch.

> *Talking point:* "Aspect-based sentiment, not a star-rating average.
> Every negative mention carries a verbatim evidence quote — the
> evidence-verbatim rate on the holdout is 1.0."

### Issue Clusters
Clusters are ranked by severity-weighted size. Pick the top reliability
cluster and expand it to show member reviews and representative quotes.

> *Talking point:* "Negative mentions are clustered per aspect with
> TF-IDF + KMeans. Severity-weighted size means a few severe complaints
> out-rank a pile of minor ones."

### Emerging Incidents
Incidents are ranked by severity score. Switch the granularity filter
between **cluster** and **aspect** to show both views. Expand an
incident to show the deterministic summary, observed vs baseline volume
and z-score.

> *Talking point:* "EWMA baseline plus z-score over weekly volume. The
> summary text is generated deterministically — no LLM in the loop, so
> it is reproducible and cheap."

### Retrieval Search
Default mode is **hybrid / rrf_equal**. Run the three queries below and
show the ranked reviews, scores and evidence quotes.

### Data Quality
Finish here. Show ingest runs, DQ pass/fail counts, failures by reason,
and the ABSA processing-reliability section (coverage rates,
failed/invalid rates).

> *Talking point:* "Data quality is a first-class surface, not an
> afterthought — 97% DQ pass rate, with every failure bucketed by
> reason."

---

## 5. Recommended search queries

Run these on the **Retrieval Search** page (hybrid mode):

1. `product stopped working after a week`
2. `bluetooth keeps disconnecting`
3. `expensive not worth the price`

Each should return reviews whose evidence quotes clearly match the
query intent — a quick, honest illustration of hybrid retrieval without
any LLM answer generation.

---

## 6. What to say in an interview

- **Framing:** "VoiceLens is a data-engineering-first VoC analytics
  platform. The core is the ETL + ML pipeline that turns raw reviews
  into governed, queryable, aspect-labelled facts."
- **Pipeline:** ingestion → data-quality gates → normalize → Postgres →
  ABSA → embed/index → cluster → anomaly detection → dashboard. All
  flows are idempotent and orchestrated with Prefect.
- **Evaluation is real:** ABSA holdout macro-F1 0.71, Cohen's κ 0.93,
  evidence-verbatim rate 1.0; refined retrieval Hit@5 0.96, Recall@20
  0.89, MRR@10 0.85. 350 automated tests, lint clean.
- **Honesty:** the full 245k ABSA pass is **not** done — ABSA ran on a
  controlled 1k batch to keep LLM cost bounded while proving the
  pipeline and the eval harness. The architecture supports scaling it.
- **Scope discipline:** the RAG agent / LangGraph layer is designed
  (see the main README §11) but deliberately not built yet — the
  retrieval page is search-only.

---

## 7. Known limitations

- ABSA coverage is a 1k batch, not the full loaded corpus.
- Cluster-level anomaly signal is sparse on the 1k subset; the
  incidents page exposes an aspect-level fallback granularity.
- Retrieval is search-only — no LLM answer generation (a later
  milestone).
- Single-locale (English), Amazon-only, 7-aspect ontology.
- Local-first: no deployment, no auth, no live marketplace integration.

---

## 8. Rebuilding the demo data (only if starting empty)

This is **not** part of the demo itself — it is how the working dataset
was produced. Do not re-run the ABSA batch unless you intend to spend
LLM budget.

```bash
make ingest-mvp-subset     # load the deterministic MVP review subset
make absa-llm-1k           # 1k real Claude ABSA batch (costs LLM budget)
make embed-v2-1k           # index the 1k batch into Qdrant
make cluster-v2            # cluster negative mentions
make anomaly-v2            # cluster-level incidents
# aspect-level fallback incidents:
python -m voicelens.pipeline.flows.anomaly_flow \
  --aspect-version v2 --provider anthropic --model claude-opus-4.6 \
  --granularity aspect
```

---

## 9. Screenshots

Screenshots are not committed (the dashboard renders client-side, so
automated capture needs a browser driver). To add them for a portfolio
write-up, capture manually:

1. `make dashboard`, open `http://localhost:8501`.
2. For each of the six pages, capture the browser window.
3. Save as `docs/screenshots/<page>.png` (e.g. `overview.png`,
   `absa.png`, `clusters.png`, `incidents.png`, `retrieval.png`,
   `data-quality.png`).

See `docs/screenshots/README.md` for the naming convention.
