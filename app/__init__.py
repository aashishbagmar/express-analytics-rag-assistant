"""
Application package for the RAG Technical Documentation Assistant.

Purpose:
    Own the FastAPI application, LangGraph workflow, ingestion pipeline,
    service adapters, storage adapters, and shared core configuration.

Responsibilities:
    - Keep web/API code separate from graph orchestration.
    - Keep graph node shells separate from domain services.
    - Expose package-level imports only when they become stable public APIs.

Public interfaces:
    None yet. Submodules expose their own skeleton interfaces.
"""
