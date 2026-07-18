"""
Transform query node.

Purpose:
    Rewrite the retrieval query after document grading finds no relevant
    chunks, closing the self-corrective retry loop
    (grading -> transform_query -> retrieval -> grading).

Responsibilities:
    - Read ``rewritten_query`` and ``previous_queries`` from GraphState.
    - Increment ``retry_count``.
    - Append the just-superseded query to ``previous_queries`` only if that
      exact normalized query is not already present in history.
    - Produce a genuinely different ``rewritten_query`` for the next
      retrieval pass, never repeating a query already tried this run.
    - Return only the state updates owned by this node: ``rewritten_query``,
      ``previous_queries``, ``retry_count``.

Design notes:
    - Deterministic reformulation strategies: rather than calling an LLM
      (which the assignment does not require for this node - only grading
      is explicitly required to use an LLM), this node cycles through a
      fixed, ordered list of reformulation strategies (broaden scope,
      extract key entities, switch phrasing, add explicit "documentation
      search" framing, strip framework qualifiers). Each strategy is applied
      to the *original* ``question`` (not the previous rewrite), so
      reformulations diversify rather than compounding drift across
      attempts.
    - Never-repeat guarantee: the node normalizes queries with lowercase +
      trim-whitespace comparison, then checks both append history and next
      rewrite candidates against ``previous_queries`` plus ``current_query``.
      If a strategy's output was already tried, the node advances to the
      next strategy in the cycle. If every strategy has been exhausted, it
      falls back to a guaranteed-unique "attempt N" suffix so the loop can
      never stall on a duplicate query even under a pathological number of
      retries.
    - Reducer semantics respected: this node's return value REPLACES
      ``rewritten_query`` and ``retry_count`` (both are plain "replace"
      fields per state.py) and returns either a single-element list or an
      empty list for ``previous_queries``, relying on the graph's
      ``operator.add`` reducer to append only new history - never returning
      the full accumulated history itself, which would double-count under
      that reducer.

Public interfaces:
    - transform_query
"""

from __future__ import annotations

import logging
import re
from typing import Callable, TypedDict

from app.graph.state import GraphState

logger = logging.getLogger(__name__)


class TransformQueryUpdate(TypedDict):
    """Partial GraphState update emitted by the transform_query node."""

    rewritten_query: str
    previous_queries: list[str]
    retry_count: int


class TransformQueryError(RuntimeError):
    """Raised when the transform_query node cannot safely reformulate the query."""


def transform_query(state: GraphState) -> TransformQueryUpdate:
    """
    Reformulate the retrieval query and advance retry state.

    Args:
        state: Current graph state. Must contain ``question`` and
            ``rewritten_query``. Reads ``previous_queries`` and
            ``retry_count`` to compute the next attempt.

    Returns:
        A partial state update containing only ``rewritten_query``,
        ``previous_queries`` (a single new entry, appended by the graph's
        reducer), and ``retry_count`` (incremented by one).
    """
    try:
        question = _extract_question(state)
        current_query = _extract_current_query(state)
        previous_queries = _extract_previous_queries(state)
        retry_count = state.get("retry_count", 0)

        previous_query_keys = {_normalize(q) for q in previous_queries}
        current_query_key = _normalize(current_query)
        tried_queries = set(previous_query_keys)
        tried_queries.add(current_query_key)

        next_query = _next_distinct_reformulation(question, retry_count, tried_queries)
        previous_queries_update = [] if current_query_key in previous_query_keys else [current_query]

        logger.info(
            "Transformed query for retry attempt %d: %r -> %r; appended_history=%s",
            retry_count + 1,
            current_query,
            next_query,
            bool(previous_queries_update),
        )

        return {
            "rewritten_query": next_query,
            "previous_queries": previous_queries_update,
            "retry_count": retry_count + 1,
        }
    except TransformQueryError:
        raise
    except Exception as exc:
        logger.exception("Query transformation failed")
        raise TransformQueryError("Unexpected failure while transforming query") from exc


def _extract_question(state: GraphState) -> str:
    """Read and validate the original user question from GraphState."""
    raw_question = state.get("question")
    if not isinstance(raw_question, str):
        raise TypeError("GraphState.question must be a string")

    question = raw_question.strip()
    if not question:
        raise ValueError("GraphState.question must be a non-empty string")
    return question


def _extract_current_query(state: GraphState) -> str:
    """Read and validate the current rewritten_query from GraphState."""
    raw_query = state.get("rewritten_query")
    if not isinstance(raw_query, str):
        raise TypeError("GraphState.rewritten_query must be a string")

    query = raw_query.strip()
    if not query:
        raise ValueError("GraphState.rewritten_query must be a non-empty string")
    return query


def _extract_previous_queries(state: GraphState) -> list[str]:
    """Read and validate previous_queries from GraphState."""
    raw_previous = state.get("previous_queries")
    if raw_previous is None:
        return []
    if not isinstance(raw_previous, list):
        raise TypeError("GraphState.previous_queries must be a list")
    return [str(entry) for entry in raw_previous]


def _normalize(query: str) -> str:
    """Normalize a query with lowercase + trim for duplicate detection."""
    return query.strip().lower()


# --------------------------------------------------------------------------- #
# Reformulation strategies
#
# Each strategy derives a candidate rewritten_query from the ORIGINAL
# question (not the previous rewrite), so successive retries diversify
# rather than compounding drift from earlier rewrites.
# --------------------------------------------------------------------------- #


def _broaden_scope(question: str) -> str:
    """Strategy 1: broaden the question to its general subject area."""
    core = _strip_question_words(question)
    return f"Give a broad overview of {core} and related concepts."


def _extract_key_entities(question: str) -> str:
    """Strategy 2: reformulate around the question's key nouns/entities."""
    entities = _key_entities(question)
    if entities:
        return f"Documentation about: {', '.join(entities)}."
    return f"Documentation related to: {question}"


def _switch_to_keyword_phrasing(question: str) -> str:
    """Strategy 3: switch from natural language to a keyword-style query."""
    entities = _key_entities(question)
    keywords = entities if entities else _strip_question_words(question).split()
    return " ".join(keywords) if keywords else question


def _add_documentation_search_framing(question: str) -> str:
    """Strategy 4: explicitly frame the query as a documentation lookup."""
    return f"Search the technical documentation for information about: {question}"


def _strip_framework_qualifiers(question: str) -> str:
    """Strategy 5: drop framework-specific qualifiers to widen the search."""
    lowered = question.lower()
    stripped = re.sub(
        r"\b(fastapi|pydantic|langchain|langgraph|starlette|python)\b",
        "",
        lowered,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()
    core = _strip_question_words(stripped) if stripped else _strip_question_words(question)
    return f"What is {core}?" if core else question


_STRATEGIES: list[Callable[[str], str]] = [
    _broaden_scope,
    _extract_key_entities,
    _switch_to_keyword_phrasing,
    _add_documentation_search_framing,
    _strip_framework_qualifiers,
]

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "explain",
    "for",
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
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
}


def _strip_question_words(question: str) -> str:
    """Strip common interrogative/stopword prefixes to isolate the core subject."""
    stripped = question.strip().rstrip("?.!")
    stripped = re.sub(
        r"(?i)^(what\s+is|what\s+are|how\s+do\s+i|how\s+can\s+i|how\s+to|explain|define)\s+",
        "",
        stripped,
    ).strip()
    return stripped or question.strip().rstrip("?.!")


def _key_entities(question: str) -> list[str]:
    """Extract a small set of non-stopword tokens as the question's key entities."""
    tokens = re.findall(r"[A-Za-z0-9_]+", question)
    entities = [token for token in tokens if token.lower() not in _STOPWORDS and len(token) > 2]
    # Preserve first-seen order while de-duplicating case-insensitively.
    seen: set[str] = set()
    unique_entities = []
    for entity in entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    return unique_entities


def _next_distinct_reformulation(
    question: str,
    retry_count: int,
    tried_queries: set[str],
) -> str:
    """
    Produce the next rewritten_query that has not already been tried.

    Cycles through ``_STRATEGIES`` starting at the index matching the
    current retry attempt, so consecutive retries use different strategies
    rather than always starting from strategy 1. If every strategy has
    already produced a previously-tried query (e.g. many retries on a very
    short question), falls back to an attempt-numbered variant that is
    guaranteed unique.
    """
    strategy_count = len(_STRATEGIES)
    for offset in range(strategy_count):
        strategy_index = (retry_count + offset) % strategy_count
        candidate = _STRATEGIES[strategy_index](question)
        if _normalize(candidate) not in tried_queries:
            return candidate

    logger.warning(
        "All %d reformulation strategies were already tried; using a "
        "guaranteed-unique fallback for retry attempt %d",
        strategy_count,
        retry_count + 1,
    )
    return f"{_add_documentation_search_framing(question)} (attempt {retry_count + 1})"
