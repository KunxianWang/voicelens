"""VoiceLens analytics dashboard (Milestone 5A).

A local-first Streamlit app that visualises the existing pipeline
outputs — ABSA mentions, issue clusters, anomaly incidents, retrieval
search and data-quality stats. It is a *read-only* view over Postgres
and Qdrant: it never calls real LLMs and is not the final RAG layer.
"""
