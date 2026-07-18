"""
LangGraph routing predicates.

Purpose:
    Define pure conditional-edge functions for routing graph state after
    document grading and optional hallucination checking.

Responsibilities:
    - Keep routing decisions side-effect free (no state mutation, no writes).
    - Read state and run configuration only.
    - Return stable route labels consumed by graph builder mappings via
      ``StateGraph.add_conditional_edges``.

Design notes:
    - Both functions are pure: same inputs always produce the same output,
      no I/O, no logging side effects beyond a read-only log line, and no
      mutation of the ``state`` mapping passed in.
    - ``retry_count``/``regen_count`` live in ``GraphState`` (see
      app/graph/state.py), but their caps (``max_retries``/``max_regen``) are
      deliberately run-time configuration, not persisted state (per
      ARCHITECTURE.md Section 6: "Config vs. state"). Both router functions
      accept the cap as an optional keyword argument so the graph builder can
      pass the value it read from ``graph.ainvoke(..., config=...)``; if
      omitted, the process-wide default from ``Settings`` is used so the
      functions remain usable/testable standalone.
    - **Error short-circuit**: both functions check ``state.get("error")``
      first and route straight to ``"fallback"`` if a prior node recorded an
      error. This is required by ARCHITECTURE.md's error-handling design
      ("a non-None error routes straight to fallback") and is additive to,
      not a replacement for, the exact branch rules the assignment specifies
      for the non-error case.
    - Routers only ever read ``retry_count``/``regen_count``; incrementing
      them is the responsibility of ``transform_query`` and generate's
      regeneration branch respectively, never the router itself, keeping
      routers pure per the assignment's explicit requirement.

Public interfaces:
    - decide_generation_path
    - decide_hallucination_path
    - RouteLabel
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from app.core.config import get_settings
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

RouteLabel = Literal["generate", "transform_query", "fallback", "end"]


def decide_generation_path(
    state: GraphState,
    *,
    max_retries: Optional[int] = None,
) -> RouteLabel:
    """
    Route the graph after document grading.

    Branches (evaluated in order):
        1. ``state["error"]`` is set -> ``"fallback"`` (a prior node failed;
           no point continuing the happy path).
        2. ``len(state["graded_docs"]) > 0`` -> ``"generate"`` (at least one
           relevant chunk was found).
        3. ``state["retry_count"] < max_retries`` -> ``"transform_query"``
           (no relevant chunks yet, but retry budget remains).
        4. Otherwise -> ``"fallback"`` (retry budget exhausted).

    Args:
        state: Current graph state. Reads ``error``, ``graded_docs``, and
            ``retry_count`` only; never mutated.
        max_retries: Run-config retry cap. Defaults to
            ``Settings.max_retries`` when omitted.

    Returns:
        One of ``"generate"``, ``"transform_query"``, or ``"fallback"``.
    """
    retry_limit = max_retries if max_retries is not None else get_settings().max_retries

    if state.get("error"):
        logger.info("decide_generation_path: routing to fallback due to state error")
        return "fallback"

    graded_docs = state.get("graded_docs") or []
    if len(graded_docs) > 0:
        logger.info(
            "decide_generation_path: routing to generate (%d relevant docs)",
            len(graded_docs),
        )
        return "generate"

    retry_count = state.get("retry_count", 0)
    if retry_count < retry_limit:
        logger.info(
            "decide_generation_path: routing to transform_query (retry_count=%d < max_retries=%d)",
            retry_count,
            retry_limit,
        )
        return "transform_query"

    logger.info(
        "decide_generation_path: routing to fallback (retry_count=%d >= max_retries=%d, no relevant docs)",
        retry_count,
        retry_limit,
    )
    return "fallback"


def decide_hallucination_path(
    state: GraphState,
    *,
    max_regen: Optional[int] = None,
) -> RouteLabel:
    """
    Route the graph after the (bonus) hallucination/groundedness check.

    Branches (evaluated in order):
        1. ``state["error"]`` is set -> ``"fallback"``.
        2. ``state["is_grounded"]`` is truthy -> ``"end"`` (answer is
           supported by context; terminate the graph).
        3. ``state["regen_count"] < max_regen`` -> ``"generate"`` (answer
           unsupported, but regeneration budget remains).
        4. Otherwise -> ``"fallback"`` (regeneration budget exhausted).

    Args:
        state: Current graph state. Reads ``error``, ``is_grounded``, and
            ``regen_count`` only; never mutated.
        max_regen: Run-config regeneration cap. Defaults to
            ``Settings.max_regen`` when omitted.

    Returns:
        One of ``"end"``, ``"generate"``, or ``"fallback"``.
    """
    regen_limit = max_regen if max_regen is not None else get_settings().max_regen

    if state.get("error"):
        logger.info("decide_hallucination_path: routing to fallback due to state error")
        return "fallback"

    if state.get("is_grounded"):
        logger.info("decide_hallucination_path: routing to end (answer is grounded)")
        return "end"

    regen_count = state.get("regen_count", 0)
    if regen_count < regen_limit:
        logger.info(
            "decide_hallucination_path: routing to generate (regen_count=%d < max_regen=%d)",
            regen_count,
            regen_limit,
        )
        return "generate"

    logger.info(
        "decide_hallucination_path: routing to fallback (regen_count=%d >= max_regen=%d, answer ungrounded)",
        regen_count,
        regen_limit,
    )
    return "fallback"
