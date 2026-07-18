"""
POST /query endpoint.

Accepts a natural-language question, invokes the compiled LangGraph workflow,
and returns a grounded answer with citations and retry metadata.
"""

from __future__ import annotations

import logging
from functools import lru_cache, partial
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from langgraph.graph.state import CompiledStateGraph

from app.graph.builder import build_graph
from app.schemas.query import CitationResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@lru_cache(maxsize=1)
def get_compiled_graph() -> CompiledStateGraph:
    """Return the process-wide compiled graph instance."""
    return build_graph()


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query(
    request: QueryRequest,
    graph: CompiledStateGraph = Depends(get_compiled_graph),
) -> QueryResponse:
    """Run the RAG graph for one user question."""
    query_id = str(uuid4())
    initial_state = _build_initial_state(query_id=query_id, question=request.question)

    try:
        result = await run_in_threadpool(
            partial(graph.invoke, initial_state, config={"recursion_limit": 25})
        )
    except Exception as exc:
        logger.exception("Graph invocation failed for query_id=%s", query_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query.",
        ) from exc

    answer = result.get("answer")
    confidence = result.get("confidence")
    if not isinstance(answer, str) or not answer.strip() or confidence not in {
        "high",
        "medium",
        "low",
    }:
        logger.error("Graph returned incomplete query response for query_id=%s", query_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Graph returned an incomplete response.",
        )

    citations = [
        CitationResponse(
            source_id=str(citation.get("source_id") or "unknown"),
            source_title=str(citation.get("source_title") or "Untitled source"),
            chunk_index=int(citation.get("chunk_index") or 0),
        )
        for citation in result.get("citations", []) or []
        if isinstance(citation, dict)
    ]

    return QueryResponse(
        query_id=str(result.get("query_id") or query_id),
        answer=answer,
        citations=citations,
        confidence=confidence,
        retry_count=int(result.get("retry_count") or 0),
        fallback_used=bool(result.get("fallback_used")),
    )


def _build_initial_state(query_id: str, question: str) -> dict[str, object]:
    """Create the initial GraphState-compatible payload for graph invocation."""
    return {
        "query_id": query_id,
        "question": question,
        "rewritten_query": question,
        "query_type": "conceptual",
        "previous_queries": [],
        "seen_chunk_ids": [],
        "retrieved_docs": [],
        "graded_docs": [],
        "grading_results": [],
        "retry_count": 0,
        "answer": None,
        "citations": [],
        "is_grounded": None,
        "regen_count": 0,
        "fallback_used": False,
        "web_search_docs": None,
        "error": None,
        "confidence": None,
    }
