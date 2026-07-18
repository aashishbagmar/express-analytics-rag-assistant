"""
Fallback node.

Purpose:
    Handle exhausted retries, empty/insufficient retrieval, and unrecoverable
    node errors by producing a safe, low-confidence terminal response instead
    of letting the graph crash or return an ungrounded answer.

Responsibilities:
    - Read ``error`` from GraphState (diagnostics only - never shown to the
      end user).
    - Return only the state updates owned by this node: ``answer``,
      ``fallback_used``, ``confidence``.
    - Never raise: this is the graph's last line of defense, so it must
      itself be maximally defensive and always produce a usable response.

Design notes:
    - Terminal node: per app/graph/builder.py, ``fallback -> END``. This node
      always returns a complete, user-safe response - it never delegates
      further or re-raises, because there is no further recovery path after
      it in the graph.
    - User-facing text is a single, constant, honest message. It deliberately
      does not echo the internal ``error`` string (which may contain
      provider/stack-trace details) back to the caller - that stays in logs
      only, per this node's requirements, to avoid leaking internal
      implementation details in the API response.
    - Always sets ``fallback_used = True`` and ``confidence = "low"`` per the
      spec, regardless of *why* fallback was entered (exhausted retries,
      empty index, or a caught error) - from the client's perspective, all
      of these are equally "we could not produce a confident answer."
    - Defensive-by-construction: even if ``state`` is missing keys or
      malformed, this node degrades to the same safe response rather than
      raising, since there is nowhere left for an exception to route to.

Public interfaces:
    - handle_fallback
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from app.graph.state import GraphState

logger = logging.getLogger(__name__)

Confidence = Literal["high", "medium", "low"]

_FALLBACK_ANSWER = (
    "I could not find enough relevant information in the indexed "
    "documentation to answer this question."
)


class FallbackUpdate(TypedDict):
    """Partial GraphState update emitted by the fallback node."""

    answer: str
    fallback_used: bool
    confidence: Confidence


def handle_fallback(state: GraphState) -> FallbackUpdate:
    """
    Produce the graph's terminal "insufficient context" response.

    Entered whenever ``decide_generation_path``/``decide_hallucination_path``
    (app/graph/router.py) route here: retries exhausted with no relevant
    docs, an empty index, a regeneration budget exhausted on an ungrounded
    answer, or a prior node recording ``state["error"]``.

    Args:
        state: Current graph state. Only ``error`` is read, and only for
            logging - it never affects the returned answer text.

    Returns:
        A partial state update containing only ``answer``,
        ``fallback_used`` (always ``True``), and ``confidence`` (always
        ``"low"``).
    """
    try:
        error = _extract_error(state)
        if error:
            logger.warning("Fallback triggered with a recorded state error: %s", error)
        else:
            logger.info("Fallback triggered with no recorded state error (exhausted retries or empty context)")

        return {
            "answer": _FALLBACK_ANSWER,
            "fallback_used": True,
            "confidence": "low",
        }
    except Exception:
        # This node must never raise - it is the graph's terminal safety
        # net, so any unexpected failure here still degrades to the same
        # safe, constant response rather than propagating.
        logger.exception("Unexpected failure inside fallback node; returning safe default response")
        return {
            "answer": _FALLBACK_ANSWER,
            "fallback_used": True,
            "confidence": "low",
        }


def _extract_error(state: GraphState) -> str | None:
    """Best-effort, non-raising read of GraphState.error for diagnostics."""
    try:
        raw_error = state.get("error")
    except Exception:
        return None
    return raw_error if isinstance(raw_error, str) and raw_error.strip() else None
