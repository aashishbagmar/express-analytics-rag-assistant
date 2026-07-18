"""
Hallucination check node (bonus).

Purpose:
    Self-RAG-style groundedness verification: determine whether the
    generated ``answer`` is actually supported by ``graded_docs`` (the
    context generation was supposed to be grounded on), before the graph
    terminates or loops back to regenerate.

Responsibilities:
    - Read ``question``, ``answer``, and ``graded_docs`` from GraphState.
    - Deterministically compare the answer's substantive terms against the
      union of graded_docs content.
    - Return only the state update owned by this node: ``is_grounded``.

Design notes:
    - No LLM required: groundedness is judged with a lexical
      containment/overlap check rather than an LLM grader, per this node's
      explicit "no LLM required" instruction (unlike ``grading.py``, which
      the assignment explicitly requires to use an LLM). Determinism here
      matters more than nuance - this is a cheap, reproducible circuit
      breaker against obvious hallucination, not a substitute for human
      review.
    - Method: extract the answer's substantive (non-stopword, non-numeric,
      length > 2) terms, extract the same term set from the concatenation of
      all graded_docs content, and compute what fraction of the answer's
      terms are actually present in that source-context term set
      (`support_ratio`). The answer is grounded iff `support_ratio` meets
      ``_MIN_SUPPORT_RATIO``, i.e. the answer does not introduce a
      significant fraction of vocabulary that never appeared anywhere in the
      supplied context.
    - Why term-set overlap and not substring/sentence matching: the
      extractive generator (``generation.py``'s fallback path) copies chunk
      text verbatim, which would trivially score as perfectly grounded under
      any method; term-overlap remains meaningful once a real LLM-authored,
      paraphrased answer is plugged in later, since paraphrases reuse the
      same substantive vocabulary even when sentence structure changes.
    - Citation markers (e.g. ``[1]``) and markdown syntax are stripped before
      comparison so formatting artifacts never affect the verdict.
    - Empty ``answer`` or empty ``graded_docs`` is treated as "not grounded"
      (fails closed) rather than raising, since an ungrounded verdict is the
      safe default that routes to regeneration/fallback rather than crashing
      the graph on this optional, bonus check.

Public interfaces:
    - check_grounding
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from app.graph.state import GraphState

logger = logging.getLogger(__name__)

# An answer is considered grounded when at least this fraction of its
# substantive terms are attested somewhere in the graded_docs context.
# Set below 1.0 to tolerate minor connective/summarizing vocabulary (e.g.
# "therefore", "overall") that legitimately does not appear verbatim in the
# source chunks without treating that as hallucination.
_MIN_SUPPORT_RATIO = 0.75

# Answers shorter than this many substantive terms are judged by requiring
# the maximum overlap ratio, since a short answer has no room for the
# statistical smoothing that "75% of terms" provides on a long answer -
# every term should be traceable to context.
_MIN_TERMS_FOR_RATIO_CHECK = 4

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}


class HallucinationCheckUpdate(TypedDict):
    """Partial GraphState update emitted by the hallucination check node."""

    is_grounded: bool


class HallucinationCheckError(RuntimeError):
    """Raised when the hallucination check node cannot safely evaluate groundedness."""


def check_grounding(state: GraphState) -> HallucinationCheckUpdate:
    """
    Deterministically check whether ``answer`` is grounded in ``graded_docs``.

    Args:
        state: Current graph state. Reads ``question`` (for logging context
            only), ``answer``, and ``graded_docs``.

    Returns:
        A partial state update containing only ``is_grounded``.
    """
    try:
        answer = _extract_answer(state)
        graded_docs = _extract_graded_docs(state)

        if not answer:
            logger.warning("No answer text available to ground; treating as not grounded")
            return {"is_grounded": False}

        if not graded_docs:
            logger.warning("No graded_docs available to ground the answer against; treating as not grounded")
            return {"is_grounded": False}

        context_terms = _build_context_terms(graded_docs)
        answer_terms = _extract_terms(answer)

        is_grounded, support_ratio, unsupported_terms = _evaluate_groundedness(
            answer_terms,
            context_terms,
        )

        logger.info(
            "Hallucination check: is_grounded=%s support_ratio=%.2f unsupported_term_count=%d",
            is_grounded,
            support_ratio,
            len(unsupported_terms),
        )
        if not is_grounded and unsupported_terms:
            logger.info("Sample unsupported terms: %s", sorted(unsupported_terms)[:10])

        return {"is_grounded": is_grounded}
    except HallucinationCheckError:
        raise
    except Exception as exc:
        logger.exception("Hallucination check failed")
        raise HallucinationCheckError("Unexpected failure while checking groundedness") from exc


def _extract_answer(state: GraphState) -> str:
    """Read and normalize the generated answer from GraphState."""
    raw_answer = state.get("answer")
    if raw_answer is None:
        return ""
    if not isinstance(raw_answer, str):
        raise TypeError("GraphState.answer must be a string or None")
    return raw_answer.strip()


def _extract_graded_docs(state: GraphState) -> list[dict[str, Any]]:
    """Read and validate graded_docs from GraphState."""
    raw_docs = state.get("graded_docs")
    if raw_docs is None:
        return []
    if not isinstance(raw_docs, list):
        raise TypeError("GraphState.graded_docs must be a list")

    documents: list[dict[str, Any]] = []
    for index, raw_document in enumerate(raw_docs):
        if not isinstance(raw_document, dict):
            raise HallucinationCheckError(f"graded_docs[{index}] must be a dictionary")
        content = raw_document.get("content")
        if not isinstance(content, str):
            raise HallucinationCheckError(f"graded_docs[{index}].content must be a string")
        documents.append(raw_document)
    return documents


def _build_context_terms(graded_docs: list[dict[str, Any]]) -> set[str]:
    """Build the union of substantive terms across all graded_docs content."""
    context_terms: set[str] = set()
    for document in graded_docs:
        context_terms |= _extract_terms(document["content"])
    return context_terms


def _evaluate_groundedness(
    answer_terms: set[str],
    context_terms: set[str],
) -> tuple[bool, float, set[str]]:
    """
    Compute the support ratio of answer_terms within context_terms and the
    resulting groundedness verdict.

    Returns:
        (is_grounded, support_ratio, unsupported_terms)
    """
    if not answer_terms:
        # An answer with no substantive terms at all (e.g. only stopwords)
        # cannot introduce unsupported concepts, so it is vacuously grounded.
        return True, 1.0, set()

    supported_terms = answer_terms & context_terms
    unsupported_terms = answer_terms - context_terms
    support_ratio = len(supported_terms) / len(answer_terms)

    if len(answer_terms) < _MIN_TERMS_FOR_RATIO_CHECK:
        is_grounded = len(unsupported_terms) == 0
    else:
        is_grounded = support_ratio >= _MIN_SUPPORT_RATIO

    return is_grounded, support_ratio, unsupported_terms


def _extract_terms(text: str) -> set[str]:
    """Extract a normalized, stopword-filtered, citation-stripped term set."""
    without_citations = re.sub(r"\[\d+\]", " ", text)
    without_code_fences = re.sub(r"```.*?```", " ", without_citations, flags=re.DOTALL)
    tokens = re.findall(r"[a-z0-9_]+", without_code_fences.lower())
    return {
        token
        for token in tokens
        if token not in _STOPWORDS and len(token) > 2 and not token.isdigit()
    }
