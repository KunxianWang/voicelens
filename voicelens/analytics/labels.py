"""Turn a cluster's TF-IDF keywords + evidence quotes into a short label.

M4A keeps labelling deliberately simple and offline — no LLM call. The
TF-IDF keywords for a cluster already carry the discriminative terms
(``ngram_range=(1, 2)`` means bigrams like ``"stopped working"`` show up
as single keywords), so a readable label is mostly a matter of stitching
the top keywords together without repeating words.

Example outputs: ``"stopped working week"``, ``"bluetooth keeps
disconnecting"``, ``"charging port loose"``, ``"not worth price"``.
"""
from __future__ import annotations

import re

# Words that add no signal to a 3-6 word issue label. Kept small on
# purpose — TF-IDF already removed sklearn's English stop-words, this is
# just the residue that still sneaks into e-commerce review keywords.
_LABEL_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "it", "its", "this", "that", "to", "of", "for", "on", "in", "with",
        "my", "me", "i", "you", "they", "very", "really", "just", "so",
        "would", "could", "get", "got", "one", "also", "even", "product",
        "item", "thing", "amazon",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def build_cluster_label(
    keywords: list[str],
    quotes: list[str],
    *,
    aspect_code: str,
    max_words: int = 6,
) -> str:
    """Compose a short label from cluster ``keywords`` (best) or ``quotes``.

    Keywords are consumed in rank order; their words are flattened with
    duplicates and stop-words dropped, preserving first-seen order so a
    leading bigram like ``"stopped working"`` keeps its phrasing. When no
    usable keyword survives, the first evidence quote is shortened; if
    that is empty too, a generic ``"<aspect> issues"`` label is returned.
    """
    seen: set[str] = set()
    picked: list[str] = []
    for keyword in keywords:
        for word in _words(keyword):
            if word in seen or word in _LABEL_STOPWORDS:
                continue
            seen.add(word)
            picked.append(word)
            if len(picked) >= max_words:
                break
        if len(picked) >= max_words:
            break

    if picked:
        return " ".join(picked)

    for quote in quotes:
        shortened = _shorten_quote(quote, max_words=max_words)
        if shortened:
            return shortened

    return f"{aspect_code.replace('_', ' ')} issues"


def _shorten_quote(quote: str, *, max_words: int) -> str:
    """First ``max_words`` content words of a quote, lower-cased."""
    words = [w for w in _words(quote) if w not in _LABEL_STOPWORDS]
    return " ".join(words[:max_words])
