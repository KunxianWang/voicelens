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
in a browser. The left sidebar lists the seven pages.

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
query intent — a quick, honest illustration of hybrid retrieval.

---

## 5b. Optional final step — cited answer (M6A beta)

On the **Retrieval Search** page, after running a search, scroll to the
**Answer Generator (Beta)** section:

1. Search for: `product stopped working after a week`.
2. Leave the answer provider on `mock` (offline — no API key needed).
3. Click **Generate cited answer**.
4. A short answer appears above the raw results, with every claim
   marked `[1]`, `[2]` … and an expandable citations list mapping each
   marker to a real retrieved review and its evidence quote.

The same thing from the CLI:

```bash
make ask-voicelens
# or, explicitly:
python scripts/ask_voicelens.py \
  --question "What are the main reliability complaints?" \
  --aspect reliability --sentiment negative --provider mock
```

> *Talking point:* "This is a citation-grounded answer generator — every
> claim is bound to a retrieved review, unsupported citations are
> stripped, and an unanswerable query returns 'insufficient evidence'
> instead of hallucinating."

---

## 5c. Optional final step — agent routing (M6B beta)

Open the **Agent Q&A (Beta)** page. The agent reads a question, a
deterministic LangGraph router picks one of four tools, and the page
shows *which* route it chose. Ask these four in order:

1. `What are the main reliability complaints?`
   → routes to **retrieval_answer** — a cited answer with evidence.
2. `Which aspect has the most negative mentions?`
   → routes to **analytics_summary** — aspect / sentiment statistics.
3. `Which issues spiked recently?`
   → routes to **incident_summary** — top EWMA-detected incidents.
4. `Should I buy Apple stock today?`
   → routes to **insufficient_scope** — a safe refusal, *no* guessed
   answer and no citations.

Each result shows the selected route, extracted filters, citation count
and warnings; the retrieval route additionally attaches a citations
list. The same thing from the CLI:

```bash
python scripts/ask_agent.py --question "Which issues spiked recently?"
```

> *Talking point:* "One question, four different data paths — reviews,
> aggregate stats, the incident table, or an honest refusal — chosen by
> a transparent keyword router. It is a controlled routing workflow,
> deliberately not an autonomous agent: no memory, no actions, one
> routing decision. `make evaluate-agent` scores it on 20 golden
> questions — route accuracy, citation rate, refusal rate."

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
  0.89, MRR@10 0.85. 371 automated tests, lint clean.
- **Honesty:** the full 245k ABSA pass is **not** done — ABSA ran on a
  controlled 1k batch to keep LLM cost bounded while proving the
  pipeline and the eval harness. The architecture supports scaling it.
- **Scope discipline:** the answer generator (M6A) is a single
  retrieve-then-answer pass with citation guardrails — *not* an agent.
  The LangGraph planner / memory / tool routing is designed (main
  README §11) but deliberately not built yet.

---

## 7. Known limitations

- ABSA coverage is a 1k batch, not the full loaded corpus.
- Cluster-level anomaly signal is sparse on the 1k subset; the
  incidents page exposes an aspect-level fallback granularity.
- The answer generator (M6A) is one retrieve-then-answer pass — no
  planner, no memory, no tool routing (those are a later milestone).
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
2. For each of the seven pages, capture the browser window.
3. Save as `docs/screenshots/<page>.png` (e.g. `overview.png`,
   `absa.png`, `clusters.png`, `incidents.png`, `retrieval.png`,
   `agent.png`, `data-quality.png`).

See `docs/screenshots/README.md` for the naming convention.
