"""
GET /documents endpoint.

Lists source documents currently represented in the persistent vector index.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.schemas.documents import DocumentSummary
from app.services.vector_store import VectorStoreService, VectorStoreServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


def get_vector_store() -> VectorStoreService:
    """Dependency provider for the vector store service."""
    return VectorStoreService()


@router.get(
    "/documents",
    response_model=list[DocumentSummary],
    status_code=status.HTTP_200_OK,
)
async def list_documents(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> list[DocumentSummary]:
    """Return summaries for indexed source documents."""
    try:
        documents = await run_in_threadpool(vector_store.list_documents)
    except VectorStoreServiceError as exc:
        logger.exception("Failed to list indexed documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list indexed documents.",
        ) from exc

    return [
        DocumentSummary(
            source_id=str(document.get("source_id") or "unknown"),
            source_title=document.get("source_title"),
            filename=document.get("filename"),
            file_type=document.get("file_type"),
            pages=document.get("pages"),
            chunks=document.get("chunk_count"),
        )
        for document in documents
    ]
