"""
FastAPI application entrypoint.

Creates the ASGI app, registers endpoint routers, and installs a small global
exception handler for unexpected failures.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.feedback import router as feedback_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.api.upload import router as upload_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Express Analytics RAG Assistant",
        description="LangGraph-powered technical documentation RAG assistant.",
        version="0.1.0",
    )

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(upload_router)
    app.include_router(documents_router)
    app.include_router(feedback_router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Return a consistent 500 envelope for unexpected exceptions."""
        logger.exception("Unhandled API error for path=%s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    return app


app = create_app()
