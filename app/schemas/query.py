"""
Pydantic schemas for POST /query.

These models define the public HTTP contract and intentionally stay separate
from internal GraphState TypedDicts so graph implementation details can evolve
without changing the API surface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Request model for submitting a natural-language question."""

    question: str = Field(..., min_length=1, description="User question to answer.")


class CitationResponse(BaseModel):
    """Source citation returned with a grounded answer."""

    source_id: str
    source_title: str
    chunk_index: int


class QueryResponse(BaseModel):
    """Response model for returning a grounded answer with sources."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    retry_count: int = Field(..., ge=0)
    fallback_used: bool
