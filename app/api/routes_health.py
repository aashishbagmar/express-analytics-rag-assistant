"""
Health endpoint skeleton.

Purpose:
    Define the public location for the liveness/readiness route.

Responsibilities:
    - Expose the health router once FastAPI endpoint logic is implemented.
    - Keep health checks minimal and side-effect free.

Public interfaces:
    - create_router: returns the health router when implemented.
"""


def create_router():
    """Create the health API router."""
