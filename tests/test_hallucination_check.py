from __future__ import annotations

import pytest

from app.graph.nodes.hallucination_check import HallucinationCheckError, check_grounding


def test_hallucination_check_marks_grounded_answer_true():
    update = check_grounding(
        {
            "question": "How does FastAPI validate request bodies?",
            "answer": "FastAPI validates request bodies using Pydantic models. [1]",
            "graded_docs": [
                {
                    "content": "FastAPI validates request bodies using Pydantic models.",
                    "metadata": {},
                    "score": 0.9,
                }
            ],
        }
    )

    assert update == {"is_grounded": True}


def test_hallucination_check_rejects_unsupported_concepts():
    update = check_grounding(
        {
            "question": "How does FastAPI validate request bodies?",
            "answer": "FastAPI uses blockchain signatures and quantum encryption.",
            "graded_docs": [
                {
                    "content": "FastAPI validates request bodies using Pydantic models.",
                    "metadata": {},
                    "score": 0.9,
                }
            ],
        }
    )

    assert update == {"is_grounded": False}


def test_hallucination_check_fails_closed_on_empty_answer():
    update = check_grounding({"question": "q", "answer": None, "graded_docs": [{"content": "context"}]})

    assert update == {"is_grounded": False}


def test_hallucination_check_raises_on_malformed_doc():
    with pytest.raises(HallucinationCheckError):
        check_grounding({"question": "q", "answer": "answer", "graded_docs": [{"content": 123}]})
