"""Prompt templates for citation-grounded answer generation.

The contract is intentionally strict: the model answers *only* from the
numbered evidence blocks, cites every claim with ``[n]`` markers, and —
when the evidence does not support an answer — emits the
:data:`INSUFFICIENT_EVIDENCE_TOKEN` sentinel instead of guessing.
"""
from __future__ import annotations

from collections.abc import Sequence

from voicelens.rag.citations import Citation

# Sentinel the model must emit (as the first token) when the retrieved
# reviews do not contain enough information to answer. The generator
# detects this and flips the result into an insufficient-evidence state.
INSUFFICIENT_EVIDENCE_TOKEN = "INSUFFICIENT_EVIDENCE"


ANSWER_SYSTEM_PROMPT = f"""\
You are a Voice-of-Customer analyst assistant for a consumer-electronics \
brand. You answer questions strictly from a set of retrieved customer \
reviews supplied in the prompt.

Rules — follow all of them:
1. Use ONLY the evidence in the numbered review blocks. Do not use outside \
knowledge.
2. Cite every claim with bracketed markers that match the block numbers, \
e.g. "Batteries swell after a few months [2][5]." Only cite numbers that \
exist in the evidence.
3. Do NOT invent numbers, percentages, counts, or dates. You may state a \
count only if you can verify it by counting the cited blocks themselves.
4. Where it helps, include a short verbatim quoted snippet from a review.
5. Be concise and analyst-oriented: 2-5 sentences, no preamble, no \
restating the question.
6. If the retrieved reviews do not contain enough information to answer \
the question, respond with EXACTLY this — the token first, then one short \
sentence of explanation:
{INSUFFICIENT_EVIDENCE_TOKEN} <why the evidence is insufficient>
"""


def _format_citation_block(citation: Citation) -> str:
    """Render one numbered evidence block for the user prompt."""
    meta_bits = [f"brand={citation.brand or 'n/a'}", f"asin={citation.asin or 'n/a'}"]
    if citation.rating is not None:
        meta_bits.append(f"rating={citation.rating}")
    if citation.aspect_codes:
        meta_bits.append(f"aspects={','.join(citation.aspect_codes)}")
    body = citation.evidence_quote or citation.text_snippet or "(no text)"
    return f"[{citation.citation_id}] ({'; '.join(meta_bits)})\n{body}"


def build_answer_user_prompt(
    question: str, citations: Sequence[Citation]
) -> str:
    """Compose the user-turn prompt: the question plus numbered evidence."""
    if not citations:
        blocks = "(no reviews were retrieved)"
    else:
        blocks = "\n\n".join(_format_citation_block(c) for c in citations)
    return (
        f"Question: {question.strip()}\n\n"
        f"Retrieved customer reviews (evidence):\n\n{blocks}\n\n"
        "Write the grounded answer now, citing blocks with [n] markers."
    )
