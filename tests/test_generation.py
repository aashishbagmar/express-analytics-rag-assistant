from __future__ import annotations

import pytest

from app.graph.nodes.generation import GenerationNodeError, generate_answer


class FakeLLM:
    def generate_answer(self, question: str, context: str):
        return {"answer": "FastAPI validates request bodies with Pydantic."}


class FailingLLM:
    def generate_answer(self, question: str, context: str):
        raise RuntimeError("offline")


def test_generation_uses_llm_answer_and_structural_citations(retrieved_doc):
    state = {
        "question": "How does FastAPI validate request bodies?",
        "graded_docs": [retrieved_doc],
        "retry_count": 0,
        "fallback_used": False,
    }

    update = generate_answer(state, llm_service=FakeLLM())

    assert update["answer"] == "FastAPI validates request bodies with Pydantic."
    assert update["citations"] == [
        {"source_id": "request-models", "source_title": "request_models", "chunk_index": 0}
    ]
    assert update["confidence"] == "medium"


def test_generation_falls_back_to_extractive_answer(retrieved_doc):
    state = {
        "question": "How does FastAPI validate request bodies?",
        "graded_docs": [retrieved_doc],
        "retry_count": 0,
        "fallback_used": False,
    }

    update = generate_answer(state, llm_service=FailingLLM())

    assert "Based on the available documentation" in update["answer"]
    assert "FastAPI validates request bodies" in update["answer"]


def test_generation_confidence_high_with_multiple_docs_first_attempt(retrieved_doc):
    second = dict(retrieved_doc)
    second["metadata"] = {"source_id": "request-models", "source_title": "request_models", "chunk_index": 1}
    state = {
        "question": "How does FastAPI validate request bodies?",
        "graded_docs": [retrieved_doc, second],
        "retry_count": 0,
        "fallback_used": False,
    }

    update = generate_answer(state, llm_service=FakeLLM())

    assert update["confidence"] == "high"


def test_generation_raises_on_empty_graded_docs():
    with pytest.raises(GenerationNodeError):
        generate_answer(
            {
                "question": "q",
                "graded_docs": [],
                "retry_count": 0,
                "fallback_used": False,
            },
            llm_service=FakeLLM(),
        )
