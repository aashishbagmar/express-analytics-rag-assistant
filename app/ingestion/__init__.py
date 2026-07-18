"""
Document ingestion package.

Purpose:
    Own the offline and API-triggered process for turning technical documents
    into indexed vector-store chunks.

Responsibilities:
    - Load source documents from files or URLs.
    - Split technical content into retrievable chunks.
    - Coordinate embedding, vector upsert, and document registry updates.

Public interfaces:
    - pipeline.IngestionPipeline
"""
