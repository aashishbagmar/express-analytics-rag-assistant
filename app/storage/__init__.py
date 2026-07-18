"""
Storage adapter package.

Purpose:
    Hold persistence boundaries that are not part of the vector-store service
    itself, such as feedback and query metadata stores.

Responsibilities:
    - Keep durable storage concerns out of API routes and graph nodes.
    - Provide small interfaces that can later be backed by SQLite, JSONL, or another store.

Public interfaces:
    - feedback_store.FeedbackStore
"""
