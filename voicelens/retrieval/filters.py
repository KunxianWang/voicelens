"""Qdrant filter builder.

The indexing flow stores ABSA outputs as parallel arrays on the payload
(``aspect_codes``, ``sentiments``, ``severities``). Qdrant's ``MatchValue``
on a list field acts as "any element equals value", so a single filter
clause cleanly answers "reviews where the bluetooth aspect is present"
or "reviews where any sentiment is negative".

The function is intentionally pure (no Qdrant client involved) so it
can be unit-tested without a server.
"""
from __future__ import annotations

from qdrant_client.http import models as rest


def build_search_filter(
    *,
    brand: str | None = None,
    asin: str | None = None,
    aspect: str | None = None,
    sentiment: str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
    aspect_version: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    absa_status: str | None = None,
) -> rest.Filter | None:
    """Compose a Qdrant ``Filter`` from optional retrieval predicates.

    Returns ``None`` when no predicates are supplied so callers can pass
    the result straight through to ``client.search`` without an explicit
    ``None`` check.
    """
    must: list[rest.FieldCondition] = []

    if brand:
        must.append(_match("brand", brand))
    if asin:
        must.append(_match("asin", asin))
    if aspect:
        # aspect_codes is a list payload; MatchValue on a list = "any equals".
        must.append(_match("aspect_codes", aspect))
    if sentiment:
        must.append(_match("sentiments", sentiment))
    if aspect_version:
        must.append(_match("aspect_version", aspect_version))
    if provider:
        must.append(_match("provider", provider))
    if model_name:
        must.append(_match("model_name", model_name))
    if absa_status:
        must.append(_match("absa_status", absa_status))

    if rating_min is not None or rating_max is not None:
        rng = rest.Range(
            gte=float(rating_min) if rating_min is not None else None,
            lte=float(rating_max) if rating_max is not None else None,
        )
        must.append(rest.FieldCondition(key="rating", range=rng))

    if not must:
        return None
    return rest.Filter(must=must)


def _match(key: str, value: str) -> rest.FieldCondition:
    return rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
