"""
LangGraph orchestration package.

Purpose:
    Contain state, graph assembly, routing predicates, and node boundaries for
    the self-corrective RAG workflow.

Responsibilities:
    - Keep workflow control flow out of FastAPI routes.
    - Keep node orchestration separate from service adapters.
    - Expose a compiled graph only through builder interfaces.

Public interfaces:
    - builder.build_graph
    - state.GraphState
"""
