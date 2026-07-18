"""
Pydantic schemas for GET /documents.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DocumentSummary(BaseModel):
    """Response model for one indexed document's visible metadata."""

    model_config = ConfigDict(extra="ignore")

    source_id: str
    source_title: str | None = None
    filename: str | None = None
    file_type: str | None = None
    pages: int | None = None
    chunks: int | None = None
