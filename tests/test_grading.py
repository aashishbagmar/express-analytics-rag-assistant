from __future__ import annotations

from app.graph.nodes.grading import grade_documents
from app.services.llm import LLMServiceError


class LLMRelevant:
    def grade_relevance(self, question: str, chunk: str):
        return {
            "verdict": "relevant",
            "confidence": 0.88,
            "reason": "The chunk directly answers the question.",
        }


class FailingLLM:
    def grade_relevance(self, question: str, chunk: str):
        raise LLMServiceError("offline")


def test_grading_llm_path_keeps_relevant_doc(retrieved_doc):
    update = grade_documents(
        {
            "question": "How does FastAPI validate request bodies?",
            "retrieved_docs": [retrieved_doc],
        },
        llm_service=LLMRelevant(),
    )

    assert len(update["graded_docs"]) == 1
    assert update["grading_results"][0]["verdict"] == "relevant"
    assert update["grading_results"][0]["confidence"] == 0.88


def test_grading_heuristic_fallback_rejects_single_generic_overlap():
    state = {
        "question": "How do I configure OAuth2 scopes in FastAPI?",
        "retrieved_docs": [
            {
                "content": "FastAPI validates request bodies using Pydantic models.",
                "metadata": {"source_id": "request", "chunk_index": 0},
                "score": 0.50,
            }
        ],
    }

    update = grade_documents(state, llm_service=FailingLLM())

    assert update["graded_docs"] == []
    assert update["grading_results"][0]["verdict"] == "irrelevant"
    assert "insufficient lexical support" in update["grading_results"][0]["reason"]


def test_grading_heuristic_fallback_accepts_specific_overlap():
    state = {
        "question": "How do I configure OAuth2 scopes in FastAPI?",
        "retrieved_docs": [
            {
                "content": "FastAPI OAuth2 scopes can restrict access to endpoints.",
                "metadata": {"source_id": "security", "chunk_index": 0},
                "score": 0.50,
            }
        ],
    }

    update = grade_documents(state, llm_service=FailingLLM())

    assert len(update["graded_docs"]) == 1
    assert update["grading_results"][0]["verdict"] == "relevant"
