"""Static text for the M6B agent — route descriptions and canned copy.

The router is deterministic and the tools are mostly non-LLM, so there
is little prompting here. What lives in this module is the user-facing
copy: human-readable route descriptions and the message returned when a
question falls outside the supported scope.
"""
from __future__ import annotations

from voicelens.agent.router import (
    ROUTE_ANALYTICS,
    ROUTE_INCIDENT,
    ROUTE_INSUFFICIENT,
    ROUTE_RETRIEVAL,
)

# One-line description of what each route does — shown in the CLI and
# the dashboard so the routing decision is legible.
ROUTE_DESCRIPTIONS: dict[str, str] = {
    ROUTE_RETRIEVAL: (
        "Retrieve customer reviews and generate a citation-grounded answer."
    ),
    ROUTE_INCIDENT: (
        "Summarise emerging-issue incidents from the anomaly-detection table."
    ),
    ROUTE_ANALYTICS: (
        "Report aggregate ABSA statistics (aspect / sentiment / severity)."
    ),
    ROUTE_INSUFFICIENT: (
        "The question is outside the supported VoC analytics scope."
    ),
}

# Returned verbatim by the insufficient-scope tool. It states what the
# agent *can* do rather than guessing at an answer it cannot ground.
INSUFFICIENT_SCOPE_MESSAGE = (
    "I can't answer that from the VoiceLens data. This assistant only "
    "covers Voice-of-Customer analytics for the indexed reviews — try "
    "asking about customer complaints on an aspect (reliability, "
    "bluetooth, charging, price, delivery, sound quality, overheating, "
    "battery), about emerging issue spikes, or about aspect / sentiment "
    "/ severity statistics."
)
