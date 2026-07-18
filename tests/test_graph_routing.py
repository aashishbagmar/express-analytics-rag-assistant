from __future__ import annotations

from app.graph.router import decide_generation_path, decide_hallucination_path


def test_generation_router_handles_relevant_documents():
    assert decide_generation_path({"graded_docs": [{"content": "x"}], "retry_count": 0}, max_retries=2) == "generate"


def test_generation_router_routes_to_retry_under_cap():
    assert decide_generation_path({"graded_docs": [], "retry_count": 1}, max_retries=2) == "transform_query"


def test_generation_router_respects_retry_cap():
    assert decide_generation_path({"graded_docs": [], "retry_count": 2}, max_retries=2) == "fallback"


def test_generation_router_error_routes_to_fallback():
    assert decide_generation_path({"graded_docs": [{"content": "x"}], "retry_count": 0, "error": "boom"}, max_retries=2) == "fallback"


def test_hallucination_router_grounded_ends():
    assert decide_hallucination_path({"is_grounded": True, "regen_count": 0}, max_regen=1) == "end"


def test_hallucination_router_regenerates_under_cap():
    assert decide_hallucination_path({"is_grounded": False, "regen_count": 0}, max_regen=1) == "generate"


def test_hallucination_router_fallback_when_exhausted():
    assert decide_hallucination_path({"is_grounded": False, "regen_count": 1}, max_regen=1) == "fallback"
