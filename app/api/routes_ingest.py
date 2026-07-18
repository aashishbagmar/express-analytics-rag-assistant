"""
Document ingestion endpoint skeleton.

Purpose:
    Define the API boundary for ingesting new local files or remote URLs into
    the vector index.

Responsibilities:
    - Accept validated ingestion requests.
    - Delegate load/chunk/embed/store work to the ingestion pipeline.
    - Return document/job status without embedding ingestion logic in routes.

Public interfaces:
    - create_router: returns the ingestion router when implemented.
"""


def create_router():
    """Create the ingestion API router."""
