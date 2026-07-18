from __future__ import annotations

from typing import get_type_hints

from app.graph.state import GraphState


def test_graph_state_contains_required_fields():
    hints = get_type_hints(GraphState, include_extras=True)

    required_fields = {
        "query_id",
        "question",
        "rewritten_query",
        "query_type",
        "previous_queries",
        "retrieved_docs",
        "seen_chunk_ids",
        "graded_docs",
        "grading_results",
        "retry_count",
        "answer",
        "citations",
        "is_grounded",
        "regen_count",
        "fallback_used",
        "web_search_docs",
        "error",
        "confidence",
    }

    assert required_fields <= set(hints)


def test_append_fields_have_reducers():
    hints = get_type_hints(GraphState, include_extras=True)

    for field in ["previous_queries", "seen_chunk_ids", "grading_results"]:
        assert hasattr(hints[field], "__metadata__")


def test_retry_limits_are_not_state_fields():
    hints = get_type_hints(GraphState, include_extras=True)

    assert "max_retries" not in hints
    assert "max_regen" not in hints
