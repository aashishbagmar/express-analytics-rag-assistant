"""
Query endpoint skeleton.

Purpose:
    Define the API boundary for submitting user questions to the LangGraph RAG
    workflow.

Responsibilities:
    - Accept validated query requests.
    - Invoke the compiled graph asynchronously.
    - Return answers, citations, retry metadata, confidence, and query_id.

Public interfaces:
    - create_router: returns the query router when implemented.
"""


def create_router():
    """Create the query API router."""
