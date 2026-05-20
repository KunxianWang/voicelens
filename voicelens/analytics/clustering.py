"""Topic clustering over negative ABSA aspect mentions (M4A).

The pipeline is two steps:

1. :func:`build_clustering_records` pulls every *negative* aspect mention
   (under a given ``aspect_version`` / ``provider`` / ``model_name``)
   whose review was successfully ABSA-processed, joined to brand / sku /
   rating metadata.
2. :func:`cluster_records` groups those records **by ``aspect_code``**
   first — an MVP simplification: "charging" complaints and "bluetooth"
   complaints are never mixed — then sub-clusters each aspect's evidence
   quotes with TF-IDF + KMeans (or BERTopic when it is installed).

Heavy models stay optional. ``scikit-learn`` is imported lazily inside
:func:`cluster_records`, so ``import voicelens.analytics`` and the input
builder work without it; BERTopic is used only when importable.
"""
from __future__ import annotations

import importlib.util
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from voicelens.analytics.labels import build_cluster_label
from voicelens.db.models import (
    ABSAReviewStatus,
    AspectMention,
    AspectOntology,
    Brand,
    Review,
    Sku,
)

# Severity -> weight, used so a few "high" complaints can out-rank a
# pile of "low" ones when ranking clusters by impact. Unknown / missing
# severity falls back to ``_DEFAULT_SEVERITY_WEIGHT``.
SEVERITY_WEIGHTS: dict[str, float] = {"high": 3.0, "medium": 2.0, "low": 1.0}
_DEFAULT_SEVERITY_WEIGHT = 1.0

# BERTopic is preferred when present but pulls UMAP + HDBSCAN; detect it
# without importing so the module stays cheap to load.
_HAS_BERTOPIC = importlib.util.find_spec("bertopic") is not None

ALGORITHM_TFIDF_KMEANS = "tfidf_kmeans"
ALGORITHM_BERTOPIC = "bertopic"


def severity_weight(severity: str | None) -> float:
    """Map an ABSA severity label to its ranking weight."""
    if not severity:
        return _DEFAULT_SEVERITY_WEIGHT
    return SEVERITY_WEIGHTS.get(severity.strip().lower(), _DEFAULT_SEVERITY_WEIGHT)


@dataclass
class ClusterRecord:
    """One negative aspect mention, flattened for clustering."""

    review_id: int
    brand: str
    asin: str
    rating: int | None
    posted_at: str | None
    aspect_code: str
    severity: str | None
    evidence_quote: str
    text_raw: str

    @property
    def cluster_text(self) -> str:
        """Text the clusterer sees: the evidence quote, raw review as fallback.

        The evidence quote is the LLM's focused complaint snippet, so it
        is far less noisy than the full review; only when it is empty do
        we fall back to ``text_raw``.
        """
        quote = (self.evidence_quote or "").strip()
        return quote if quote else (self.text_raw or "").strip()


@dataclass
class ClusterResult:
    """One emergent issue cluster within a single aspect."""

    aspect_code: str
    label: str
    algorithm: str
    keywords: list[str]
    members: list[ClusterRecord]
    representative_review_ids: list[int] = field(default_factory=list)
    representative_quotes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def severity_weighted_size(self) -> float:
        return round(sum(severity_weight(m.severity) for m in self.members), 4)

    @property
    def member_review_ids(self) -> list[int]:
        return [m.review_id for m in self.members]


def build_clustering_records(
    session: Session,
    *,
    aspect_version: str = "v2",
    provider: str = "anthropic",
    model_name: str = "claude-opus-4.6",
    limit: int | None = None,
) -> list[ClusterRecord]:
    """Select negative aspect mentions on successfully-processed reviews.

    Only ``sentiment == 'negative'`` mentions are returned, and only for
    reviews carrying an ``ABSAReviewStatus`` of ``success`` under the
    same ``(aspect_version, provider, model_name)`` — failed / invalid
    reviews have unreliable evidence and are excluded.
    """
    stmt = (
        select(
            Review.id,
            Brand.name.label("brand"),
            Sku.asin,
            Review.rating,
            Review.posted_at,
            AspectOntology.code.label("aspect_code"),
            AspectMention.severity,
            AspectMention.evidence_quote,
            Review.text_raw,
        )
        .join(AspectMention, AspectMention.review_id == Review.id)
        .join(AspectOntology, AspectOntology.id == AspectMention.aspect_id)
        .join(Sku, Sku.id == Review.sku_id)
        .join(Brand, Brand.id == Sku.brand_id)
        .join(
            ABSAReviewStatus,
            and_(
                ABSAReviewStatus.review_id == Review.id,
                ABSAReviewStatus.aspect_version == aspect_version,
                ABSAReviewStatus.provider == provider,
                ABSAReviewStatus.model_name == model_name,
                ABSAReviewStatus.status == ABSAReviewStatus.STATUS_SUCCESS,
            ),
        )
        .where(
            and_(
                AspectMention.aspect_version == aspect_version,
                AspectMention.model_name == model_name,
                AspectMention.sentiment == "negative",
            )
        )
        .order_by(AspectOntology.code, Review.id)
    )
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)

    records: list[ClusterRecord] = []
    for row in session.execute(stmt):
        records.append(
            ClusterRecord(
                review_id=int(row.id),
                brand=row.brand or "",
                asin=row.asin or "",
                rating=int(row.rating) if row.rating is not None else None,
                posted_at=row.posted_at.isoformat() if row.posted_at else None,
                aspect_code=row.aspect_code,
                severity=row.severity,
                evidence_quote=row.evidence_quote or "",
                text_raw=row.text_raw or "",
            )
        )
    return records


def cluster_records(
    records: list[ClusterRecord],
    *,
    min_cluster_size: int = 5,
    max_clusters_per_aspect: int = 8,
    keywords_per_cluster: int = 8,
    random_state: int = 42,
    algorithm: str = "auto",
) -> list[ClusterResult]:
    """Cluster ``records`` per aspect; drop aspects/clusters below ``min_cluster_size``.

    An aspect whose total negative-mention count is below
    ``min_cluster_size`` is skipped wholesale — there is not enough
    signal to form even one cluster. Within a large enough aspect, any
    sub-cluster that ends up below ``min_cluster_size`` is also dropped
    (KMeans noise), so every returned cluster has at least
    ``min_cluster_size`` members.

    ``algorithm`` is ``"auto"`` (BERTopic if installed, else TF-IDF +
    KMeans), or one of ``"tfidf_kmeans"`` / ``"bertopic"`` to force one.
    """
    by_aspect: dict[str, list[ClusterRecord]] = defaultdict(list)
    for record in records:
        by_aspect[record.aspect_code].append(record)

    use_bertopic = algorithm == ALGORITHM_BERTOPIC or (
        algorithm == "auto" and _HAS_BERTOPIC
    )

    results: list[ClusterResult] = []
    for aspect_code in sorted(by_aspect):
        group = by_aspect[aspect_code]
        if len(group) < min_cluster_size:
            continue  # too little signal for even one cluster
        if use_bertopic:
            clusters = _cluster_one_aspect_bertopic(
                group,
                min_cluster_size=min_cluster_size,
                keywords_per_cluster=keywords_per_cluster,
            )
        else:
            clusters = _cluster_one_aspect_tfidf(
                group,
                min_cluster_size=min_cluster_size,
                max_clusters=max_clusters_per_aspect,
                keywords_per_cluster=keywords_per_cluster,
                random_state=random_state,
            )
        # Largest clusters first within an aspect — the headline issues.
        clusters.sort(key=lambda c: (-c.size, c.label))
        results.extend(clusters)
    return results


def _representatives(
    members: list[ClusterRecord], *, n: int = 5
) -> tuple[list[int], list[str]]:
    """Pick the highest-severity members as the cluster's exemplars."""
    ordered = sorted(
        members,
        key=lambda m: (-severity_weight(m.severity), m.review_id),
    )
    review_ids: list[int] = []
    quotes: list[str] = []
    seen_quotes: set[str] = set()
    for member in ordered:
        if len(review_ids) < n:
            review_ids.append(member.review_id)
        quote = (member.evidence_quote or "").strip()
        if quote and quote.lower() not in seen_quotes and len(quotes) < n:
            seen_quotes.add(quote.lower())
            quotes.append(quote)
    return review_ids, quotes


def _make_result(
    aspect_code: str,
    algorithm: str,
    members: list[ClusterRecord],
    keywords: list[str],
) -> ClusterResult:
    rep_ids, rep_quotes = _representatives(members)
    label = build_cluster_label(keywords, rep_quotes, aspect_code=aspect_code)
    return ClusterResult(
        aspect_code=aspect_code,
        label=label,
        algorithm=algorithm,
        keywords=keywords,
        members=members,
        representative_review_ids=rep_ids,
        representative_quotes=rep_quotes,
    )


def _cluster_one_aspect_tfidf(
    records: list[ClusterRecord],
    *,
    min_cluster_size: int,
    max_clusters: int,
    keywords_per_cluster: int,
    random_state: int,
) -> list[ClusterResult]:
    """TF-IDF + KMeans sub-clustering of one aspect's negative mentions."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [r.cluster_text for r in records]
    matrix, feature_names = _vectorize(texts, TfidfVectorizer)
    if matrix is None or matrix.shape[1] == 0:
        # No usable vocabulary — emit the aspect as a single cluster.
        return [_make_result(records[0].aspect_code, ALGORITHM_TFIDF_KMEANS, records, [])]

    n_docs = len(records)
    k = max(1, min(max_clusters, n_docs // min_cluster_size))
    k = min(k, n_docs)

    if k <= 1:
        keywords = _top_keywords(matrix, list(range(n_docs)), feature_names, keywords_per_cluster)
        return [_make_result(records[0].aspect_code, ALGORITHM_TFIDF_KMEANS, records, keywords)]

    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(matrix)

    clusters: list[ClusterResult] = []
    for cluster_id in range(k):
        idx = [i for i, lab in enumerate(labels) if lab == cluster_id]
        if len(idx) < min_cluster_size:
            continue  # KMeans noise — too small to be a real issue cluster
        members = [records[i] for i in idx]
        keywords = _top_keywords(matrix, idx, feature_names, keywords_per_cluster)
        clusters.append(
            _make_result(records[0].aspect_code, ALGORITHM_TFIDF_KMEANS, members, keywords)
        )

    if not clusters:
        # Every sub-cluster fell below the floor; keep the aspect whole.
        keywords = _top_keywords(matrix, list(range(n_docs)), feature_names, keywords_per_cluster)
        return [_make_result(records[0].aspect_code, ALGORITHM_TFIDF_KMEANS, records, keywords)]
    return clusters


def _vectorize(texts: list[str], vectorizer_cls):
    """Fit a TF-IDF vectorizer, retrying without stop-words on empty vocab."""
    for use_stopwords in (True, False):
        try:
            vec = vectorizer_cls(
                stop_words="english" if use_stopwords else None,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
            )
            matrix = vec.fit_transform(texts)
            if matrix.shape[1] > 0:
                return matrix, vec.get_feature_names_out()
        except ValueError:
            continue
    return None, []


def _top_keywords(matrix, row_indices, feature_names, top_n) -> list[str]:
    """Top TF-IDF terms by mean weight over the given rows."""
    import numpy as np

    if not row_indices:
        return []
    mean_weights = np.asarray(matrix[row_indices].mean(axis=0)).ravel()
    order = mean_weights.argsort()[::-1]
    keywords: list[str] = []
    for idx in order:
        if mean_weights[idx] <= 0.0:
            break
        keywords.append(str(feature_names[idx]))
        if len(keywords) >= top_n:
            break
    return keywords


def _cluster_one_aspect_bertopic(
    records: list[ClusterRecord],
    *,
    min_cluster_size: int,
    keywords_per_cluster: int,
) -> list[ClusterResult]:
    """BERTopic sub-clustering of one aspect (used only when installed)."""
    from bertopic import BERTopic  # pragma: no cover - optional heavy dep

    texts = [r.cluster_text for r in records]
    model = BERTopic(min_topic_size=min_cluster_size, verbose=False)
    topics, _ = model.fit_transform(texts)

    by_topic: dict[int, list[int]] = defaultdict(list)
    for i, topic_id in enumerate(topics):
        if topic_id != -1:  # -1 is BERTopic's outlier bucket
            by_topic[int(topic_id)].append(i)

    clusters: list[ClusterResult] = []
    for topic_id, idx in by_topic.items():
        if len(idx) < min_cluster_size:
            continue
        members = [records[i] for i in idx]
        topic_terms = model.get_topic(topic_id) or []
        keywords = [str(term) for term, _ in topic_terms[:keywords_per_cluster]]
        clusters.append(
            _make_result(records[0].aspect_code, ALGORITHM_BERTOPIC, members, keywords)
        )

    if not clusters:
        keywords = []
        return [_make_result(records[0].aspect_code, ALGORITHM_BERTOPIC, records, keywords)]
    return clusters
