from __future__ import annotations

from app.graph.nodes.query_analysis import analyze_query


def test_query_analysis_classifies_conceptual_question():
    update = analyze_query({"question": "What is dependency injection?"})

    assert update["query_type"] == "conceptual"
    assert update["rewritten_query"] == "Explain the concept of dependency injection in FastAPI."
    assert update["previous_queries"] == ["What is dependency injection?"]


def test_query_analysis_classifies_how_to_with_existing_context():
    update = analyze_query({"question": "How do I configure routes in FastAPI?"})

    assert update["query_type"] == "how_to"
    assert update["rewritten_query"] == "How do I configure routes in FastAPI?"


def test_query_analysis_classifies_troubleshooting():
    update = analyze_query({"question": "FastAPI returns a validation error, how do I fix it?"})

    assert update["query_type"] == "troubleshooting"


def test_query_analysis_rewrites_request_body_validation_naturally():
    update = analyze_query({"question": "How are request bodies validated?"})

    assert update["rewritten_query"] == "How does FastAPI validate request bodies using Pydantic models?"
