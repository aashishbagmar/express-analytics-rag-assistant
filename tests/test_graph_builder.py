from __future__ import annotations

import app.graph.builder as builder_mod


def test_graph_builder_compiles():
    graph = builder_mod.build_graph()

    assert graph is not None


def test_graph_builder_contains_expected_edges():
    graph = builder_mod.build_graph()
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("__start__", "query_analysis") in edges
    assert ("query_analysis", "retrieval") in edges
    assert ("retrieval", "grading") in edges
    assert ("transform_query", "retrieval") in edges
    assert ("generation", "__end__") in edges
    assert ("fallback", "__end__") in edges


def test_graph_happy_path(monkeypatch, base_graph_state, retrieved_doc):
    monkeypatch.setattr(
        builder_mod,
        "analyze_query",
        lambda state: {"rewritten_query": state["question"], "query_type": "conceptual", "previous_queries": [state["question"]]},
    )
    monkeypatch.setattr(builder_mod, "retrieve_documents", lambda state: {"retrieved_docs": [retrieved_doc]})
    monkeypatch.setattr(
        builder_mod,
        "grade_documents",
        lambda state: {"graded_docs": state["retrieved_docs"], "grading_results": []},
    )
    monkeypatch.setattr(
        builder_mod,
        "generate_answer",
        lambda state: {
            "answer": "grounded answer",
            "citations": [{"source_id": "request-models", "source_title": "request_models", "chunk_index": 0}],
            "confidence": "high",
        },
    )

    result = builder_mod.build_graph().invoke(base_graph_state, config={"recursion_limit": 25})

    assert result["answer"] == "grounded answer"
    assert result["fallback_used"] is False
    assert result["retry_count"] == 0


def test_graph_retry_then_happy_path(monkeypatch, base_graph_state, retrieved_doc):
    calls = {"retrieval": 0}

    monkeypatch.setattr(
        builder_mod,
        "analyze_query",
        lambda state: {"rewritten_query": state["question"], "query_type": "conceptual", "previous_queries": [state["question"]]},
    )

    def fake_retrieve(state):
        calls["retrieval"] += 1
        return {"retrieved_docs": [] if calls["retrieval"] == 1 else [retrieved_doc]}

    monkeypatch.setattr(builder_mod, "retrieve_documents", fake_retrieve)
    monkeypatch.setattr(
        builder_mod,
        "grade_documents",
        lambda state: {
            "graded_docs": state["retrieved_docs"],
            "grading_results": [],
        },
    )
    monkeypatch.setattr(
        builder_mod,
        "transform_query",
        lambda state: {
            "rewritten_query": "retry query",
            "previous_queries": [state["rewritten_query"]],
            "retry_count": state["retry_count"] + 1,
        },
    )
    monkeypatch.setattr(
        builder_mod,
        "generate_answer",
        lambda state: {"answer": "after retry", "citations": [], "confidence": "medium"},
    )

    result = builder_mod.build_graph().invoke(base_graph_state, config={"recursion_limit": 25})

    assert result["answer"] == "after retry"
    assert result["retry_count"] == 1
    assert calls["retrieval"] == 2


def test_graph_fallback_path(monkeypatch, base_graph_state):
    monkeypatch.setattr(
        builder_mod,
        "analyze_query",
        lambda state: {"rewritten_query": state["question"], "query_type": "conceptual", "previous_queries": [state["question"]]},
    )
    monkeypatch.setattr(builder_mod, "retrieve_documents", lambda state: {"retrieved_docs": []})
    monkeypatch.setattr(builder_mod, "grade_documents", lambda state: {"graded_docs": [], "grading_results": []})
    monkeypatch.setattr(
        builder_mod,
        "transform_query",
        lambda state: {
            "rewritten_query": f"retry {state['retry_count']}",
            "previous_queries": [state["rewritten_query"]],
            "retry_count": state["retry_count"] + 1,
        },
    )
    monkeypatch.setattr(
        builder_mod,
        "handle_fallback",
        lambda state: {"answer": "fallback", "fallback_used": True, "confidence": "low"},
    )

    result = builder_mod.build_graph().invoke(base_graph_state, config={"recursion_limit": 25})

    assert result["answer"] == "fallback"
    assert result["fallback_used"] is True
    assert result["retry_count"] == 2
