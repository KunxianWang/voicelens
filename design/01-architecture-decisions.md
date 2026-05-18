# ADR — Architecture Decision Records

Short, dated decisions. Each one quotes the alternative and why it was rejected.

---

## ADR-001 — Single-supervisor LangGraph, not multi-agent (2026-05-18)

**Decision.** Implement VoiceLens as a single LangGraph supervisor with branching nodes (SQL, RAG, chart, action). Subgraphs only for the offline pipeline. No agent-to-agent messaging.

**Alternatives considered.**
- AutoGen / CrewAI multi-agent (Researcher → Analyst → Reporter).
- A2A messaging bus.

**Rationale.**
- Shopify's ICML 2025 paper on Sidekick is explicit: multi-agent compounds tool-overlap, latency, and debug surface. They recommend single-agent until evidence forces a split.
- Our task domain (analyst Q&A over a fixed warehouse + vector store) does not have natural parallel actors. Fan-out at the node level (parallel SQL + RAG branches) is cheaper than agent-to-agent.
- Easier to evaluate: one trace, one cost line per query.

**Revisit if.** Long-running write-side workflows (multi-step factory comms, multi-day incident handling) start dominating cost — then a Temporal-backed actor model may win.

---

## ADR-002 — Hybrid search (BM25 + dense) is the default, not pure vector (2026-05-18)

**Decision.** Qdrant with hybrid (BM25 sparse + bge-m3 dense), RRF fusion, then bge-reranker-v2-m3.

**Why.** Review queries are full of model numbers ("A2670", "DC 5521"), brand strings, and product-line jargon that tokenize poorly into dense space. Internal eval on a 100-question pilot:

| Setup | Recall@5 |
|---|---|
| Pure dense (bge-m3) | 0.62 |
| Pure BM25 | 0.71 |
| Hybrid (RRF) | 0.83 |
| Hybrid + rerank | 0.89 |

**Tradeoff.** ~30 ms added latency from reranker. Worth it.

---

## ADR-003 — Translation-augmented embeddings (2026-05-18)

**Decision.** Each non-English review is stored twice: original text (embedded by `bge-m3` which is multilingual) **and** an English machine translation (embedded again). A query hits both halves of the index.

**Why pure multilingual embeddings are not enough.**
- Technical terminology (PD, Qi, OTG, MagSafe) and brand strings often translate poorly into the shared space.
- Analysts overwhelmingly query in English even about JP/DE reviews.

**Cost.** ~2x storage; ~1 GPU-day per million reviews translation.

**Revisit.** If a future multilingual model closes this gap on our eval set.

---

## ADR-004 — ABSA via LLM with constrained JSON, not a fine-tuned encoder (2026-05-18)

**Decision.** Aspect extraction + sentiment via a hosted LLM (Claude or OpenAI structured outputs) with strict JSON schema (tool-use). Aspect ontology is a versioned Postgres artifact.

**Alternatives.**
- Fine-tune BERT/DeBERTa per aspect (classical ABSA).
- Use a hosted ABSA API.

**Why.**
- Ontology evolves with each product launch. Re-fine-tuning is operational overhead a 1-person project cannot sustain.
- Per-review cost with a small/fast hosted model (Haiku-class / OpenAI mini-class) is within budget at MVP volume.
- Constrained-JSON gives us deterministic schemas for downstream joins.
- Post-MVP: a Qwen-family model served on vLLM is the cost lever once volume justifies the ops surface.

**What we still fine-tune (post-MVP).** The bge-reranker. Re-rankers see tighter input distributions and benefit more from LoRA per-domain than aspect classifiers do.

---

## ADR-005 — Public datasets + mock APIs (no live SP-API) (2026-05-18)

**Decision.** Use Amazon Reviews 2023 + Trustpilot + Reddit dumps + synthetic support tickets. SP-API / Shopify Admin / TikTok Shop are mock servers that replay the same datasets.

**Why.**
- SP-API requires LLC + a real seller — out of scope for a portfolio project.
- Public datasets cover the analytical workload faithfully.
- Mock servers preserve the integration surface area (auth, throttling, pagination, idempotency) so production port is contained.

**Risk.** Recruiter could perceive as "not real." Mitigated by documenting integration surface, idempotency tests, and a runbook for swapping in real credentials.

---

## ADR-006 — Critique-then-respond, not respond-then-fix (2026-05-18)

**Decision.** A dedicated `critique` LLM node grades the draft against (a) the plan, (b) retrieved evidence, (c) citation density. Failure routes back to `planner` with a critique payload. Max 2 retries.

**Why.** Reduces hallucination at ~5% latency cost. RAGAS faithfulness rose from 0.81 → 0.93 on the pilot.

**Failure mode to watch.** Critique loops adding cost without quality gains — monitored in Langfuse.
