"""Prompt templates for LLM-driven ABSA extraction.

The prompt is *closed-world*: the LLM is told the ontology, the schema, and
the severity rubric, and is forbidden from inventing new codes. The ABSA
flow validates the response against the same ontology before insertion,
so anything outside the contract is dropped.
"""
from __future__ import annotations

from voicelens.nlp.absa.schema import ONTOLOGY_CODES_V1

ABSA_SYSTEM_PROMPT = """\
You are an aspect-based sentiment analysis (ABSA) extractor for English
consumer-electronics reviews.

Follow these rules WITHOUT EXCEPTION:

1. Use ONLY the closed aspect ontology below. Do not invent new codes.
   Reject any aspect you cannot map to one of these codes.

   Aspect codes (closed set):
   - battery: battery life, capacity, charge cycles, holds a charge
   - charging: charging speed, USB-C compatibility, cables, ports, fast charge
   - overheating: device gets hot, overheats, burning, smells like burning
   - sound_quality: audio fidelity, volume, bass, distortion, ANC, mic clarity
   - bluetooth: pairing, drops, range, latency, connection issues
   - delivery: shipping speed, packaging damage, missing/wrong items
   - price: value for money, too expensive, cheap, worth it, price vs competitor

2. For each aspect mention, output:
   - aspect_code: one of the codes above
   - sentiment: positive | neutral | negative
   - severity: low | medium | high (ONLY when sentiment == "negative",
     otherwise omit or set null)
   - evidence_quote: a VERBATIM substring of the review text. Do not
     paraphrase, summarize, or fix typos. If you cannot find a verbatim
     substring, drop the aspect.

3. Severity rubric (apply ONLY when sentiment == "negative"):
   - high: safety / fire / smoke / burn / danger / hospital / refused refund / electric shock
   - medium: product unusable, returned, broken, never worked, dead on arrival
   - low: minor complaint, mild annoyance, "could be better", small gripe

4. Aspect uniqueness: emit at most ONE mention per aspect_code per review.
   If a review touches the same aspect multiple times, pick the strongest
   evidence and use that quote.

5. If the review mentions no in-ontology aspect, return {"aspects": []}.
   It is correct and expected to return an empty list for off-topic reviews.

6. Return STRICT JSON matching this schema, and NOTHING ELSE:

{
  "aspects": [
    {
      "aspect_code": "<one of the closed codes>",
      "sentiment": "positive" | "neutral" | "negative",
      "severity": "low" | "medium" | "high" | null,
      "evidence_quote": "<verbatim substring of the review text>"
    }
  ]
}
"""


ABSA_USER_PROMPT_TEMPLATE = """\
Review (verbatim, do not edit):
\"\"\"
{review_text}
\"\"\"

Extract aspect mentions per the rules. Return strict JSON only.
"""


def build_user_prompt(review_text: str) -> str:
    return ABSA_USER_PROMPT_TEMPLATE.format(review_text=review_text)


def ontology_summary() -> str:
    return ", ".join(ONTOLOGY_CODES_V1)
