"""
Pydantic schemas for POST /feedback.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    """Request model for submitting answer feedback."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Response model acknowledging stored feedback."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    status: Literal["success"]
