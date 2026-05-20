from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from voicelens.db.engine import Base


class Brand(Base):
    __tablename__ = "brand"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)

    skus: Mapped[list[Sku]] = relationship("Sku", back_populates="brand")


class Sku(Base):
    __tablename__ = "sku"
    __table_args__ = (UniqueConstraint("brand_id", "asin", name="uq_sku_brand_asin"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brand.id", ondelete="CASCADE"), nullable=False)
    asin: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model_number: Mapped[str | None] = mapped_column(String(128))
    category: Mapped[str | None] = mapped_column(String(64))
    launch_date: Mapped[datetime | None] = mapped_column(DateTime)

    brand: Mapped[Brand] = relationship("Brand", back_populates="skus")
    reviews: Mapped[list[Review]] = relationship("Review", back_populates="sku")


class IngestRun(Base):
    __tablename__ = "ingest_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    dag_run_id: Mapped[str | None] = mapped_column(String(128))

    reviews: Mapped[list[Review]] = relationship("Review", back_populates="ingest_run")
    dq_events: Mapped[list[DQEvent]] = relationship("DQEvent", back_populates="ingest_run")


class Review(Base):
    __tablename__ = "review"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_review_source_sourceid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    sku_id: Mapped[int] = mapped_column(ForeignKey("sku.id"), nullable=False, index=True)
    locale: Mapped[str | None] = mapped_column(String(16))
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    lang_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    posted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    helpful_count: Mapped[int] = mapped_column(Integer, default=0)
    text_raw: Mapped[str] = mapped_column(Text, nullable=False)
    char_len: Mapped[int] = mapped_column(Integer, nullable=False)
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_run.id"), nullable=False)
    ingest_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    sku: Mapped[Sku] = relationship("Sku", back_populates="reviews")
    ingest_run: Mapped[IngestRun] = relationship("IngestRun", back_populates="reviews")
    aspect_mentions: Mapped[list[AspectMention]] = relationship(
        "AspectMention", back_populates="review"
    )


class DQEvent(Base):
    __tablename__ = "dq_event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_run.id"), nullable=False)
    check_name: Mapped[str] = mapped_column(String(64), nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_failed: Mapped[int] = mapped_column(Integer, nullable=False)
    reason_codes: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    ingest_run: Mapped[IngestRun] = relationship("IngestRun", back_populates="dq_events")


class AspectOntology(Base):
    __tablename__ = "aspect_ontology"
    __table_args__ = (UniqueConstraint("version", "code", name="uq_ontology_version_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(16), nullable=False, default="v1")
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    severity_applicable: Mapped[bool] = mapped_column(Boolean, default=True)


class AspectMention(Base):
    """One aspect detected on one review (positive extractions only)."""

    __tablename__ = "aspect_mention"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("review.id"), nullable=False, index=True)
    aspect_id: Mapped[int] = mapped_column(ForeignKey("aspect_ontology.id"), nullable=False)
    sentiment: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[str | None] = mapped_column(String(16))
    evidence_quote: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(64))
    aspect_version: Mapped[str] = mapped_column(String(16), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    review: Mapped[Review] = relationship("Review", back_populates="aspect_mentions")


class ABSAReviewStatus(Base):
    """Per-review ABSA processing record.

    ``aspect_mention`` only records positive extractions, so a review with
    ``{"aspects": []}`` leaves no fingerprint there. This table records
    review-level processing outcome so re-runs can skip reviews we have
    already paid to process — load-bearing for real-LLM cost control.

    The unique constraint ``(review_id, aspect_version, provider,
    model_name)`` lets us run multiple providers (mock + LLM) side-by-side
    and compare their coverage without one clobbering the other.
    """

    __tablename__ = "absa_review_status"
    __table_args__ = (
        UniqueConstraint(
            "review_id", "aspect_version", "provider", "model_name",
            name="uq_absa_status_review_version_provider_model",
        ),
    )

    STATUS_SUCCESS = "success"
    STATUS_NO_MENTIONS = "no_mentions"
    STATUS_INVALID = "invalid"
    STATUS_FAILED = "failed"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("review.id"), nullable=False, index=True)
    aspect_version: Mapped[str] = mapped_column(String(16), nullable=False, default="v1")
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    n_mentions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_errors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_codes_json: Mapped[dict | None] = mapped_column(JSON)
    processed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Cluster(Base):
    """One topic cluster of negative ABSA mentions (M4A).

    Clusters are produced per ``aspect_code`` by ``cluster_flow``: for
    each aspect the negative evidence quotes are grouped with TF-IDF +
    KMeans (or BERTopic when installed). A cluster carries a short
    human-readable ``label``, the TF-IDF ``topic_keywords`` it was
    labelled from, and a few ``representative_review_ids`` / quotes for
    quick inspection.

    ``run_id`` groups every cluster written by a single flow invocation
    so re-runs can be told apart; ``cluster_flow`` clears prior rows for
    the same ``(aspect_version, provider, model_name)`` before inserting
    so stats stay deterministic.
    """

    __tablename__ = "cluster"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    aspect_version: Mapped[str] = mapped_column(String(16), nullable=False, default="v2")
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    aspect_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False, default="tfidf_kmeans")
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    severity_weighted_size: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    topic_keywords: Mapped[list | None] = mapped_column(JSON)
    representative_review_ids: Mapped[list | None] = mapped_column(JSON)
    representative_quotes: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    members: Mapped[list[ReviewCluster]] = relationship(
        "ReviewCluster", back_populates="cluster", cascade="all, delete-orphan"
    )


class ReviewCluster(Base):
    """Membership join: one negative review (under one aspect) in one cluster.

    A review with negative mentions on several aspects is clustered once
    per aspect, so it can appear in several ``review_cluster`` rows — but
    never twice in the same cluster (enforced by the unique constraint).
    """

    __tablename__ = "review_cluster"
    __table_args__ = (
        UniqueConstraint("cluster_id", "review_id", name="uq_review_cluster_cluster_review"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("cluster.id", ondelete="CASCADE"), nullable=False, index=True
    )
    review_id: Mapped[int] = mapped_column(ForeignKey("review.id"), nullable=False, index=True)
    aspect_code: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    cluster: Mapped[Cluster] = relationship("Cluster", back_populates="members")
