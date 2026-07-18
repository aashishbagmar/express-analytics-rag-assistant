from __future__ import annotations

from app.graph.nodes.fallback import handle_fallback


def test_fallback_returns_safe_low_confidence_response():
    update = handle_fallback({"error": None})

    assert update == {
        "answer": "I could not find enough relevant information in the indexed documentation to answer this question.",
        "fallback_used": True,
        "confidence": "low",
    }


def test_fallback_does_not_leak_internal_error():
    update = handle_fallback({"error": "chromadb connection refused at localhost"})

    assert "chromadb" not in update["answer"].lower()
    assert update["fallback_used"] is True


def test_fallback_handles_malformed_state_without_raising():
    update = handle_fallback(object())  # type: ignore[arg-type]

    assert update["confidence"] == "low"
