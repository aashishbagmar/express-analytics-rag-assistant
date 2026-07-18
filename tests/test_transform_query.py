from __future__ import annotations

from app.graph.nodes.transform_query import transform_query


def test_transform_query_increments_retry_count():
    update = transform_query(
        {
            "question": "What is dependency injection?",
            "rewritten_query": "Explain dependency injection in FastAPI.",
            "previous_queries": [],
            "retry_count": 0,
        }
    )

    assert update["retry_count"] == 1


def test_transform_query_generates_unique_rewrites():
    state = {
        "question": "How do I configure OAuth2 scopes in FastAPI?",
        "rewritten_query": "How do I configure OAuth2 scopes in FastAPI?",
        "previous_queries": ["How do I configure OAuth2 scopes in FastAPI?"],
        "retry_count": 0,
    }

    history = list(state["previous_queries"])
    for _ in range(2):
        update = transform_query(state)
        history += update["previous_queries"]
        state = {
            "question": state["question"],
            "rewritten_query": update["rewritten_query"],
            "previous_queries": history,
            "retry_count": update["retry_count"],
        }

    sequence = history + [state["rewritten_query"]]
    assert len({query.strip().lower() for query in sequence}) == len(sequence)


def test_transform_query_preserves_previous_queries_reducer_behavior():
    update = transform_query(
        {
            "question": "How do I configure OAuth2 scopes in FastAPI?",
            "rewritten_query": "How do I configure OAuth2 scopes in FastAPI?",
            "previous_queries": ["How do I configure OAuth2 scopes in FastAPI?"],
            "retry_count": 0,
        }
    )

    # Current query already exists in accumulated history, so the node returns
    # an empty delta rather than the full history or a duplicate.
    assert update["previous_queries"] == []
