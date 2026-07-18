"""
POST /ingest endpoint.

Runs the local ingestion pipeline and incrementally reindexes the requested
corpus path without removing separately uploaded documents.
"""

from __future__ import annotations

import logging
from functools import partial

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.ingestion.pipeline import IngestionPipeline
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.vector_store import VectorStoreService, VectorStoreServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


def get_ingestion_pipeline() -> IngestionPipeline:
    """Dependency provider for the ingestion pipeline."""
    return IngestionPipeline()


def get_vector_store() -> VectorStoreService:
    """Dependency provider for the vector store service."""
    return VectorStoreService()


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest(
    request: IngestRequest | None = Body(default=None),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> IngestResponse:
    """Load/chunk the corpus and incrementally reindex its source documents."""
    ingest_request = request or IngestRequest()

    try:
        chunks = await run_in_threadpool(pipeline.ingest, ingest_request.path)
    except Exception as exc:
        logger.exception("Ingestion pipeline failed for path=%s", ingest_request.path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest corpus.",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No supported documents were found at the ingestion path.",
        )

    try:
        indexed_count = await run_in_threadpool(
            partial(vector_store.replace_documents_for_sources, chunks)
        )
    except VectorStoreServiceError as exc:
        logger.exception("Vector index update failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update vector index.",
        ) from exc

    source_ids = {
        str(chunk.metadata.get("source_id"))
        for chunk in chunks
        if chunk.metadata and chunk.metadata.get("source_id")
    }
    logger.info(
        "Ingested %d source documents into %d indexed chunks",
        len(source_ids),
        indexed_count,
    )

    return IngestResponse(
        documents_processed=len(source_ids),
        chunks_created=indexed_count,
        status="success",
    )
