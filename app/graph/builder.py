"""
LangGraph builder.

Purpose:
    Assemble the compiled LangGraph ``StateGraph`` for the self-corrective
    RAG workflow: wiring nodes, the grading conditional edge, the
    retry loop, and the fallback termination path.

Responsibilities:
    - Register graph nodes in the required workflow order.
    - Wire the post-grading conditional routing through the pure router
      function ``decide_generation_path`` (app/graph/router.py).
    - Close the retry loop: ``grading -> transform_query -> retrieval ->
      grading`` until ``retry_count`` reaches ``max_retries``.
    - Terminate on ``generation`` (happy path) or ``fallback`` (insufficient
      context / retries exhausted / error).
    - Compile and expose the graph for the API layer via ``build_graph()``.
    - Keep graph construction free of node business logic - every node here
      is imported from ``app/graph/nodes/*``, never redefined inline.

Scope for this phase:
    Per the current implementation plan, this builder wires exactly the
    nodes listed in the assignment for this phase: ``query_analysis``,
    ``retrieval``, ``grading``, ``transform_query``, ``generation``,
    ``fallback``. The bonus hallucination-check loop
    (``check_grounding`` / ``decide_hallucination_path``) is intentionally
    NOT wired here - ``generation`` edges straight to ``END`` for now. That
    loop is a self-contained future extension: inserting it only requires
    replacing the single ``generation -> END`` edge with
    ``generation -> {conditional: decide_hallucination_path}``; no other
    part of this graph changes.

Public interfaces:
    - build_graph
    - GraphNodeName
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes.fallback import handle_fallback
from app.graph.nodes.generation import generate_answer
from app.graph.nodes.grading import grade_documents
from app.graph.nodes.query_analysis import analyze_query
from app.graph.nodes.retrieval import retrieve_documents
from app.graph.nodes.transform_query import transform_query
from app.graph.router import decide_generation_path
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

GraphNodeName = Literal[
    "query_analysis",
    "retrieval",
    "grading",
    "transform_query",
    "generation",
    "fallback",
]

# Node names, defined once so the node-registration and edge-wiring sections
# below cannot drift from each other via a typo in a string literal.
_QUERY_ANALYSIS: GraphNodeName = "query_analysis"
_RETRIEVAL: GraphNodeName = "retrieval"
_GRADING: GraphNodeName = "grading"
_TRANSFORM_QUERY: GraphNodeName = "transform_query"
_GENERATION: GraphNodeName = "generation"
_FALLBACK: GraphNodeName = "fallback"

# Maps decide_generation_path's route labels to the node each one enters.
# Kept local to build_graph's conditional edge so router.py never needs to
# know concrete LangGraph node names (it only returns semantic labels).
_GENERATION_ROUTE_MAP: dict[str, GraphNodeName] = {
    "generate": _GENERATION,
    "transform_query": _TRANSFORM_QUERY,
    "fallback": _FALLBACK,
}


def build_graph() -> CompiledStateGraph:
    """
    Build and compile the LangGraph StateGraph for the RAG workflow.

    Graph shape:

        START -> query_analysis -> retrieval -> grading
                                                    |
                                    (decide_generation_path)
                                     /       |        \\
                                generation  transform_query  fallback
                                    |             |              |
                                   END        retrieval          END
                                          (closes retry loop)

    The retry loop (``grading -> transform_query -> retrieval -> grading``)
    is bounded by ``retry_count`` vs. ``max_retries``, both read/compared
    only inside ``decide_generation_path`` - never inside a node - so the
    loop's termination logic stays a single, pure, testable predicate.

    Returns:
        A compiled LangGraph graph ready for ``.invoke()``/``.ainvoke()``.
        Callers should pass an explicit ``recursion_limit`` in the invoke
        config as a second, execution-level safety net against the retry
        loop (per ARCHITECTURE.md's retry-mechanism design), independent of
        the ``retry_count``/``max_retries`` application-level bound.
    """
    graph = StateGraph(GraphState)

    graph.add_node(_QUERY_ANALYSIS, analyze_query)
    graph.add_node(_RETRIEVAL, retrieve_documents)
    graph.add_node(_GRADING, grade_documents)
    graph.add_node(_TRANSFORM_QUERY, transform_query)
    graph.add_node(_GENERATION, generate_answer)
    graph.add_node(_FALLBACK, handle_fallback)

    graph.add_edge(START, _QUERY_ANALYSIS)
    graph.add_edge(_QUERY_ANALYSIS, _RETRIEVAL)
    graph.add_edge(_RETRIEVAL, _GRADING)

    graph.add_conditional_edges(
        _GRADING,
        decide_generation_path,
        _GENERATION_ROUTE_MAP,
    )

    # Closes the self-corrective retry loop: a rewritten query always flows
    # back through retrieval before being graded again.
    graph.add_edge(_TRANSFORM_QUERY, _RETRIEVAL)

    graph.add_edge(_GENERATION, END)
    graph.add_edge(_FALLBACK, END)

    compiled = graph.compile()
    logger.info(
        "Compiled RAG StateGraph with nodes=%s",
        [_QUERY_ANALYSIS, _RETRIEVAL, _GRADING, _TRANSFORM_QUERY, _GENERATION, _FALLBACK],
    )
    return compiled
