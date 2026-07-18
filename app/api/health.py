"""
Health endpoint for local smoke tests and reviewer checks.

Purpose:
    Provide a simple GET /health route that confirms the FastAPI application is
    reachable.

Responsibilities:
    - Return a lightweight liveness response.
    - Avoid touching vector stores, LLM providers, or ingestion services.
    - Keep reviewer smoke testing independent from external API keys.

Public interfaces:
    - router: FastAPI router exposing GET /health.
"""

from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return application liveness status."""
    return {"status": "healthy"}
