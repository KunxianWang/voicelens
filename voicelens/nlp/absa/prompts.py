"""Prompt templates for LLM-driven ABSA extraction.

The prompt is *closed-world*: the LLM is told the ontology, the schema,
the severity rubric, and the disambiguation rules for the catch-all
``reliability`` aspect, and is forbidden from inventing new codes. The
ABSA flow validates the response against the same ontology before
insertion, so anything outside the contract is dropped.
"""
from __future__ import annotations

from voicelens.nlp.absa.schema import ONTOLOGY_CODES_LATEST

ABSA_SYSTEM_PROMPT = """\
You are an aspect-based sentiment analysis (ABSA) extractor for English
consumer-electronics reviews.

Follow these rules WITHOUT EXCEPTION:

1. Use ONLY the closed aspect ontology below. Do not invent new codes.
   Reject any aspect you cannot map to one of these codes.

   Aspect codes (closed set):
   - battery: battery life, capacity, charge cycles, holds a charge
   - charging: charging speed, USB-C compatibility, cables, ports, fast charge
   - overheating: device gets hot, overheats, burning, smells like burning, fire/smoke
   - sound_quality: audio fidelity, volume, bass, distortion, ANC, mic clarity
   - bluetooth: pairing, drops, range, latency, connection issues
   - delivery: shipping speed, packaging damage, missing/wrong items
   - price: value for money, too expensive, cheap, worth it, price vs competitor
   - reliability: general product failure, durability, defective unit, DOA,
     stopped working / broke / died with no clear specific component

2. Specific aspects beat ``reliability``. ``reliability`` is the
   catch-all for general durability / product-failure complaints when
   the failure cannot be attributed to a more specific aspect.

   Apply these disambiguation rules in order:
   - If the review names the failed component (battery, charging port,
     bluetooth, speaker, etc.), use that specific aspect instead of
     ``reliability``.
   - Use ``reliability`` only when the review describes general product
     failure or durability and the failed component is NOT clear.
   - Do NOT use ``reliability`` for shipping or package damage. Use
     ``delivery`` for those. Only fall back to ``reliability`` when the
     product itself is described as defective or broken (e.g. DOA).
   - Do NOT use ``reliability`` just because the rating is low or the
     reviewer is unhappy. The text must describe a failure.
   - If the evidence is ambiguous, prefer NOT extracting ``reliability``.

   Worked examples:
   - "Stopped working after one week."   -> reliability, negative, medium
   - "Dead on arrival."                  -> reliability, negative, medium
   - "Started smoking while charging."   -> overheating, negative, high
   - "The USB-C port stopped charging."  -> charging,   negative, medium
   - "Battery died after a week."        -> battery,    negative, medium
   - "Package was damaged but product works." -> delivery, negative or neutral
     depending on text; NOT reliability.

3. For each aspect mention, output:
   - aspect_code: one of the codes above
   - sentiment: positive | neutral | negative
   - severity: low | medium | high (ONLY when sentiment == "negative",
     otherwise omit or set null)
   - evidence_quote: a VERBATIM substring of the review text. Do not
     paraphrase, summarize, or fix typos. If you cannot find a verbatim
     substring, drop the aspect.

4. Severity rubric (apply ONLY when sentiment == "negative"):
   - high: safety / fire / smoke / burn / danger / hospital / refused refund / electric shock
   - medium: product unusable, returned, broken, never worked, dead on arrival
   - low: minor complaint, mild annoyance, "could be better", small gripe

5. Aspect uniqueness: emit at most ONE mention per aspect_code per review.
   If a review touches the same aspect multiple times, pick the strongest
   evidence and use that quote.

6. If the review mentions no in-ontology aspect, return {"aspects": []}.
   It is correct and expected to return an empty list for off-topic reviews.

7. Return STRICT JSON matching this schema, and NOTHING ELSE:

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
    return ", ".join(ONTOLOGY_CODES_LATEST)
