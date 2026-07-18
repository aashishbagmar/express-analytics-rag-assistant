"""
Pydantic schemas for POST /upload.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    """Response model for a single uploaded and ingested document."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "already_exists", "error"]
    filename: str
    chunks_created: int | None = Field(default=None, ge=0)
    processing_time_seconds: float | None = Field(default=None, ge=0)
    message: str | None = None
