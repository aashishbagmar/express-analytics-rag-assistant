"""
Service adapter package.

Purpose:
    Hold provider- and storage-facing service boundaries used by graph nodes and
    ingestion code.

Responsibilities:
    - Keep graph nodes thin and testable.
    - Isolate LLM, embedding, vector store, registry, and web-search providers.
    - Make provider swaps localized to service adapters.

Public interfaces:
    Service modules expose class stubs for each integration boundary.
"""
