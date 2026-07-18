from __future__ import annotations

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_document() -> Document:
    return Document(
        page_content=(
            "# Request Bodies\n\n"
            "FastAPI validates request bodies using Pydantic models and type hints.\n\n"
            "## Nested Models\n\n"
            "Pydantic models can be nested and validated recursively."
        ),
        metadata={
            "source_id": "request-models",
            "source_title": "request_models",
            "source_path": "data/corpus/request_models.md",
        },
    )


@pytest.fixture
def retrieved_doc() -> dict[str, object]:
    return {
        "content": "FastAPI validates request bodies using Pydantic models.",
        "metadata": {
            "source_id": "request-models",
            "source_title": "request_models",
            "chunk_index": 0,
        },
        "score": 0.82,
    }


@pytest.fixture
def base_graph_state() -> dict[str, object]:
    return {
        "query_id": "test-query",
        "question": "How does FastAPI validate request bodies?",
        "rewritten_query": "How does FastAPI validate request bodies?",
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


class FakeEmbeddingService:
    """Small deterministic embedding service for vector-store tests."""

    def model_id(self) -> str:
        return "fake-embedding-model"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("fastapi")),
            float(lowered.count("validat")),
            float(lowered.count("dependency")),
            float(len(lowered.split())),
        ]
