"""
Retrieval node.

Purpose:
    Retrieve relevant chunks from the persistent Chroma vector store using the
    analyzed/re-written query from GraphState.

Responsibilities:
    - Read ``rewritten_query`` from GraphState.
    - Delegate vector search to ``VectorStoreService.similarity_search``.
    - Return only the state update owned by this node: ``retrieved_docs``.
    - Keep business logic inside VectorStoreService, not in the graph node.

Public interfaces:
    - retrieve_documents
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from app.graph.state import GraphState
from app.services.vector_store import VectorStoreService, VectorStoreServiceError

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


class RetrievedDocument(TypedDict):
    """Retrieved chunk shape consumed by downstream grading."""

    content: str
    metadata: dict[str, Any]
    score: float


class RetrievalUpdate(TypedDict):
    """Partial GraphState update emitted by the retrieval node."""

    retrieved_docs: list[RetrievedDocument]


class RetrievalNodeError(RuntimeError):
    """Raised when the retrieval node cannot safely retrieve documents."""


def retrieve_documents(
    state: GraphState,
    *,
    vector_store: VectorStoreService | None = None,
    k: int = DEFAULT_TOP_K,
) -> RetrievalUpdate:
    """
    Retrieve candidate document chunks for the current rewritten_query.

    Args:
        state: Current graph state. Must contain ``rewritten_query``.
        vector_store: Optional injected vector-store service, primarily useful
            for tests. If omitted, the default persistent VectorStoreService is
            used.
        k: Number of top chunks to retrieve.

    Returns:
        A partial state update containing only ``retrieved_docs``.
    """
    query = _extract_rewritten_query(state)
    _validate_k(k)

    store = vector_store or VectorStoreService()

    try:
        results = store.similarity_search(query=query, k=k)
    except VectorStoreServiceError:
        logger.exception("Vector store retrieval failed")
        raise
    except Exception as exc:
        logger.exception("Unexpected retrieval failure")
        raise RetrievalNodeError("Unexpected retrieval failure") from exc

    retrieved_docs = [_normalize_result(result) for result in results]
    logger.info(
        "Retrieved %d documents for rewritten query; k=%d",
        len(retrieved_docs),
        k,
    )
    return {"retrieved_docs": retrieved_docs}


def _extract_rewritten_query(state: GraphState) -> str:
    """Read and validate the rewritten retrieval query from GraphState."""
    raw_query = state.get("rewritten_query")
    if not isinstance(raw_query, str):
        raise TypeError("GraphState.rewritten_query must be a string")

    query = raw_query.strip()
    if not query:
        raise ValueError("GraphState.rewritten_query must be a non-empty string")
    return query


def _validate_k(k: int) -> None:
    """Validate the requested number of retrieval results."""
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _normalize_result(result: dict[str, Any]) -> RetrievedDocument:
    """
    Normalize VectorStoreService output into the retrieved_docs state shape.

    VectorStoreService is expected to return:
        {"content": str, "metadata": dict, "score": float}
    """
    content = result.get("content")
    metadata = result.get("metadata")
    score = result.get("score")

    if not isinstance(content, str):
        raise RetrievalNodeError("Retrieved result content must be a string")
    if not isinstance(metadata, dict):
        raise RetrievalNodeError("Retrieved result metadata must be a dictionary")
    if not isinstance(score, (int, float)):
        raise RetrievalNodeError("Retrieved result score must be numeric")

    return {
        "content": content,
        "metadata": dict(metadata),
        "score": float(score),
    }
