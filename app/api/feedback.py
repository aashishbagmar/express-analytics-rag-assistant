"""
POST /feedback endpoint.

Persists user feedback for generated answers through the storage layer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.storage.feedback_store import FeedbackStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


def get_feedback_store() -> FeedbackStore:
    """Dependency provider for feedback persistence."""
    return FeedbackStore()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_feedback(
    request: FeedbackRequest,
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackResponse:
    """Persist answer feedback for a query_id."""
    try:
        await run_in_threadpool(
            lambda: store.save_feedback(
                query_id=request.query_id,
                rating=request.rating,
                comment=request.comment,
            )
        )
    except Exception as exc:
        logger.exception("Failed to persist feedback for query_id=%s", request.query_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist feedback.",
        ) from exc

    return FeedbackResponse(query_id=request.query_id, status="success")
