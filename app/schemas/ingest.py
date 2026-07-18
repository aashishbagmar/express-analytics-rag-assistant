"""
Pydantic schemas for POST /ingest.

The current API supports ingesting a local corpus path, matching the
assignment's allowance for local/standalone ingestion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    """Request model for ingesting a local corpus path."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(default="data/corpus", min_length=1)


class IngestResponse(BaseModel):
    """Response model for ingestion status and indexed document metadata."""

    model_config = ConfigDict(extra="forbid")

    documents_processed: int = Field(..., ge=0)
    chunks_created: int = Field(..., ge=0)
    status: Literal["success"]
