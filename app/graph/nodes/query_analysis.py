"""
Query analysis node.

Purpose:
    Prepare the user's raw question for retrieval without calling an LLM.

Responsibilities:
    - Normalize and lightly rewrite the user question for vector search.
    - Expand ambiguous questions with natural, conservative retrieval context.
    - Classify the query into one of the supported query types:
      conceptual, how_to, troubleshooting, api_reference.
    - Return only GraphState-compatible updates for fields owned by this node.

Public interfaces:
    - analyze_query
"""

from __future__ import annotations

import logging
import re
from typing import Literal, TypedDict

from app.graph.state import GraphState

logger = logging.getLogger(__name__)

QueryType = Literal["conceptual", "how_to", "troubleshooting", "api_reference"]


class QueryAnalysisUpdate(TypedDict):
    """Partial GraphState update emitted by the query analysis node."""

    rewritten_query: str
    query_type: QueryType
    previous_queries: list[str]


_TROUBLESHOOTING_TERMS = {
    "bug",
    "crash",
    "error",
    "exception",
    "fail",
    "failed",
    "failing",
    "fix",
    "issue",
    "not working",
    "problem",
    "stack trace",
    "traceback",
}

_HOW_TO_TERMS = {
    "add",
    "build",
    "configure",
    "create",
    "enable",
    "implement",
    "install",
    "make",
    "run",
    "set up",
    "setup",
    "use",
}

_API_REFERENCE_TERMS = {
    "api",
    "argument",
    "attribute",
    "class",
    "endpoint",
    "function",
    "method",
    "parameter",
    "property",
    "return",
    "signature",
}

_CONCEPTUAL_TERMS = {
    "compare",
    "concept",
    "define",
    "definition",
    "difference",
    "explain",
    "overview",
    "what is",
    "why",
}


def analyze_query(state: GraphState) -> QueryAnalysisUpdate:
    """
    Analyze the user question and prepare retrieval-oriented state updates.

    The implementation is intentionally deterministic and conservative. It
    preserves the user's core intent while making short, natural rewrites that
    read like documentation search queries rather than keyword bags.
    """
    try:
        question = _extract_question(state)
        query_type = _classify_query(question)
        rewritten_query = _rewrite_query(question, query_type)

        logger.info(
            "Analyzed query; query_type=%s original_length=%d rewritten_length=%d",
            query_type,
            len(question),
            len(rewritten_query),
        )

        return {
            "rewritten_query": rewritten_query,
            "query_type": query_type,
            "previous_queries": [question],
        }
    except Exception:
        logger.exception("Query analysis failed")
        raise


def _extract_question(state: GraphState) -> str:
    """Read and normalize the original user question from GraphState."""
    raw_question = state.get("question")
    if not isinstance(raw_question, str):
        raise TypeError("GraphState.question must be a string")

    question = _normalize_whitespace(raw_question)
    if not question:
        raise ValueError("GraphState.question must be a non-empty string")
    return question


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving user wording."""
    return re.sub(r"\s+", " ", text).strip()


def _classify_query(question: str) -> QueryType:
    """Classify a normalized question into the supported query types."""
    lowered = question.lower()

    if _contains_any(lowered, _TROUBLESHOOTING_TERMS):
        return "troubleshooting"

    if _looks_like_api_reference(question, lowered):
        return "api_reference"

    if _looks_like_how_to(lowered):
        return "how_to"

    if _contains_any(lowered, _CONCEPTUAL_TERMS):
        return "conceptual"

    # Default to conceptual because unknown questions usually ask for meaning
    # or explanation, and conceptual retrieval should remain broad.
    return "conceptual"


def _looks_like_api_reference(question: str, lowered: str) -> bool:
    """Return whether the question appears to ask for API/reference details."""
    if _contains_any(lowered, _API_REFERENCE_TERMS):
        return True

    # Common technical symbols often indicate code/API lookup intent.
    return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(|\b[A-Za-z_][A-Za-z0-9_]*\.", question))


def _looks_like_how_to(lowered: str) -> bool:
    """Return whether the question asks for steps or usage."""
    if lowered.startswith(("how do i ", "how can i ", "how to ")):
        return True
    return _contains_any(lowered, _HOW_TO_TERMS)


def _contains_any(text: str, terms: set[str]) -> bool:
    """Return whether any whole-word-ish term appears in text."""
    for term in terms:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def _rewrite_query(question: str, query_type: QueryType) -> str:
    """
    Conservatively rewrite the question for retrieval.

    Rewrites are natural-language search queries. They add framework context
    only when the user did not already name a framework/library, which keeps
    broad questions such as "What is dependency injection?" retrievable from
    FastAPI docs without turning every query into generic keyword stuffing.
    """
    lowered = question.lower()

    if _mentions_request_body_validation(lowered):
        return "How does FastAPI validate request bodies using Pydantic models?"

    if query_type == "conceptual":
        concept = _extract_concept(question)
        if concept:
            return f"Explain the concept of {concept}{_context_suffix(question)}."
        return f"Explain this concept{_context_suffix(question)}: {question}"

    if query_type == "how_to":
        if _has_domain_context(question):
            return question
        action = _strip_question_prefix(question)
        return f"How does FastAPI {action}?"

    if query_type == "troubleshooting":
        if _has_domain_context(question):
            return question
        return f"How do you troubleshoot this FastAPI issue: {question}"

    if query_type == "api_reference":
        if _has_domain_context(question):
            return question
        return f"What does the FastAPI documentation say about this API reference: {question}"

    return question


def _mentions_request_body_validation(lowered: str) -> bool:
    """Return whether the question asks about FastAPI request body validation."""
    has_validation = "validat" in lowered
    has_body_context = "request body" in lowered or "request bodies" in lowered
    return has_validation and (
        has_body_context or "body" in lowered or "pydantic" in lowered
    )


def _extract_concept(question: str) -> str:
    """Extract the core concept from common conceptual question forms."""
    stripped = question.strip().rstrip("?.!")
    patterns = [
        r"(?i)^what\s+is\s+(.+)$",
        r"(?i)^what\s+are\s+(.+)$",
        r"(?i)^explain\s+(.+)$",
        r"(?i)^define\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stripped)
        if match:
            return match.group(1).strip()
    return stripped


def _strip_question_prefix(question: str) -> str:
    """Convert common how-to phrasings into a natural action phrase."""
    stripped = question.strip().rstrip("?.!")
    replacements = [
        (r"(?i)^how\s+do\s+i\s+", ""),
        (r"(?i)^how\s+can\s+i\s+", ""),
        (r"(?i)^how\s+to\s+", ""),
    ]
    action = stripped
    for pattern, replacement in replacements:
        action = re.sub(pattern, replacement, action).strip()
    return action[:1].lower() + action[1:] if action else stripped


def _context_suffix(question: str) -> str:
    """Return a framework context suffix only when the question lacks one."""
    if _has_domain_context(question):
        return ""
    return " in FastAPI"


def _has_domain_context(question: str) -> bool:
    """Return whether the user already named a likely framework/library context."""
    lowered = question.lower()
    known_context_terms = {
        "fastapi",
        "pydantic",
        "langchain",
        "langgraph",
        "python",
        "starlette",
    }
    return _contains_any(lowered, known_context_terms)
