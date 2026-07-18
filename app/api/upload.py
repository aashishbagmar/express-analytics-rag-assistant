"""
POST /upload endpoint.

Accepts multipart file uploads, saves them under data/uploads/, and ingests
them through the existing load-chunk-embed-index pipeline.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.ingestion.pipeline import IngestionPipeline
from app.schemas.upload import UploadResponse
from app.services.vector_store import VectorStoreService, VectorStoreServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])

UPLOAD_DIRECTORY = Path("data/uploads")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def get_ingestion_pipeline() -> IngestionPipeline:
    """Dependency provider for the ingestion pipeline."""
    return IngestionPipeline()


def get_vector_store() -> VectorStoreService:
    """Dependency provider for the vector store service."""
    return VectorStoreService()


def _validate_upload_filename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a filename.",
        )

    safe_name = Path(filename).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded filename is invalid.",
        )

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported file type. Supported uploads: "
                "pdf, docx, txt, md."
            ),
        )
    return safe_name


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> UploadResponse:
    """Save an uploaded file and ingest it into the vector index."""
    filename = _validate_upload_filename(file.filename)
    destination = UPLOAD_DIRECTORY / filename

    if destination.exists():
        logger.info("Upload skipped because file already exists: %s", destination)
        return UploadResponse(
            status="already_exists",
            filename=filename,
            message="A file with this name already exists in uploads. Skipping ingestion.",
        )

    try:
        content = await file.read()
    except OSError as exc:
        logger.exception("Failed to read uploaded file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        ) from exc

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File exceeds the 10 MB upload limit.",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    started_at = time.perf_counter()
    try:
        UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
        await run_in_threadpool(partial(destination.write_bytes, content))

        chunks = await run_in_threadpool(pipeline.ingest_file, destination)
        if not chunks:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No supported content could be extracted from the uploaded file.",
            )

        indexed_count = await run_in_threadpool(partial(vector_store.add_documents, chunks))
    except HTTPException:
        raise
    except VectorStoreServiceError as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Vector indexing failed for uploaded file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index uploaded document.",
        ) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Upload ingestion failed for file %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded document.",
        ) from exc

    processing_time = round(time.perf_counter() - started_at, 3)
    logger.info(
        "Uploaded and indexed %s into %d chunks in %.3fs",
        filename,
        indexed_count,
        processing_time,
    )

    return UploadResponse(
        status="success",
        filename=filename,
        chunks_created=indexed_count,
        processing_time_seconds=processing_time,
        message="Upload successful",
    )
