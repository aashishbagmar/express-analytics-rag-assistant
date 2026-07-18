"""
Generation node.

Purpose:
    Generate the final grounded answer, with structured source citations,
    from the chunks the grading node judged relevant.

Responsibilities:
    - Read ``question`` and ``graded_docs`` from GraphState.
    - Delegate answer generation to ``LLMService.generate_answer``.
    - Fail over to a deterministic extractive answer (built only from
      ``graded_docs`` content) whenever the LLM path is unavailable, so this
      node never silently produces an answer that isn't grounded in the
      supplied context.
    - Derive ``citations`` structurally from ``graded_docs`` metadata rather
      than trusting free-text citations from the LLM.
    - Derive ``confidence`` deterministically from ``graded_docs`` count,
      ``retry_count``, and ``fallback_used``.
    - Return only the state updates owned by this node: ``answer``,
      ``citations``, and ``confidence``.

Design notes:
    - Grounding guarantee: the prompt/extractive context passed to the
      answer generator is built *exclusively* from ``graded_docs`` content -
      no other state field ever contributes text to the answer - so the
      answer can never reference material outside the supplied context.
    - LLM contract: this node calls
      ``LLMService.generate_answer(question=question, context=context)``,
      expecting either a plain string or a ``{"answer": str}`` mapping back.
      ``LLMService.generate_answer`` is still a stub (raises
      ``NotImplementedError``) as of this phase; this node treats that
      exactly like any other LLM failure (``LLMServiceError``, timeout,
      malformed response) and transparently falls over to the deterministic
      extractive generator below, the same fail-safe pattern used by
      ``graph/nodes/grading.py``. Once ``LLMService.generate_answer`` gains a
      real implementation matching this signature, the LLM path activates
      with no changes needed here.
    - Deterministic extractive fallback: concatenates the highest-scoring
      graded chunks (each attributed inline as ``[n]``) into a short grounded
      answer. It is intentionally simple and reproducible rather than
      "smart," since its only job is to guarantee a context-grounded answer
      is always returned even with no LLM available.
    - ``graded_docs`` empty is treated as a programming error, not a normal
      routing outcome: ``decide_generation_path`` (app/graph/router.py) is
      responsible for never reaching this node with zero relevant chunks
      (that case routes to ``transform_query``/``fallback`` instead), so
      reaching here with an empty list means the graph was wired incorrectly.
      Hence ``GenerationNodeError`` is raised rather than silently degrading.

Public interfaces:
    - generate_answer
    - GenerationNodeError
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, TypedDict

from app.core.config import get_settings
from app.graph.state import Citation, GraphState
from app.services.llm import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

Confidence = Literal["high", "medium", "low"]

# Below this count of graded_docs, a "high" confidence answer is never
# warranted even on a first-attempt, non-fallback generation - a single
# supporting chunk is thinner grounding than several corroborating ones.
_MIN_GRADED_DOCS_FOR_HIGH_CONFIDENCE = 2

# Cap on how many graded chunks feed the extractive fallback, keeping the
# fallback answer readable rather than dumping every retrieved chunk.
_MAX_FALLBACK_CONTEXT_CHUNKS = 5

_UNKNOWN_SOURCE_ID = "unknown"
_UNKNOWN_SOURCE_TITLE = "Untitled source"


class GradedDocumentInput(TypedDict):
    """Expected shape of each entry in GraphState.graded_docs."""

    content: str
    metadata: dict[str, Any]
    score: float


class GenerationUpdate(TypedDict):
    """Partial GraphState update emitted by the generation node."""

    answer: str
    citations: list[Citation]
    confidence: Confidence


class GenerationNodeError(RuntimeError):
    """Raised when the generation node cannot safely produce a grounded answer."""


def generate_answer(
    state: GraphState,
    *,
    llm_service: Optional[LLMService] = None,
) -> GenerationUpdate:
    """
    Generate a grounded answer and citations from graded_docs.

    Args:
        state: Current graph state. Must contain ``question`` and a
            non-empty ``graded_docs``.
        llm_service: Optional injected LLM service, primarily for tests. If
            omitted, a default ``LLMService`` is constructed.

    Returns:
        A partial state update containing only ``answer``, ``citations``,
        and ``confidence``.

    Raises:
        GenerationNodeError: if ``graded_docs`` is empty, state is malformed,
            or answer generation fails unexpectedly.
    """
    try:
        question = _extract_question(state)
        graded_docs = _extract_graded_docs(state)

        if not graded_docs:
            raise GenerationNodeError(
                "generate_answer requires at least one graded_docs entry; "
                "the router should route empty graded_docs to transform_query/fallback"
            )

        service = llm_service if llm_service is not None else LLMService(get_settings())

        answer, used_fallback_generator = _generate_answer_text(question, graded_docs, service)
        citations = _build_citations(graded_docs)
        confidence = _compute_confidence(state, graded_docs)

        logger.info(
            "Generated answer from %d graded docs; llm_used=%s confidence=%s",
            len(graded_docs),
            not used_fallback_generator,
            confidence,
        )
        return {"answer": answer, "citations": citations, "confidence": confidence}
    except GenerationNodeError:
        raise
    except Exception as exc:
        logger.exception("Answer generation failed")
        raise GenerationNodeError("Unexpected failure while generating an answer") from exc


def _extract_question(state: GraphState) -> str:
    """Read and validate the original user question from GraphState."""
    raw_question = state.get("question")
    if not isinstance(raw_question, str):
        raise TypeError("GraphState.question must be a string")

    question = raw_question.strip()
    if not question:
        raise ValueError("GraphState.question must be a non-empty string")
    return question


def _extract_graded_docs(state: GraphState) -> list[GradedDocumentInput]:
    """Read and validate graded_docs from GraphState."""
    raw_docs = state.get("graded_docs")
    if not isinstance(raw_docs, list):
        raise TypeError("GraphState.graded_docs must be a list")

    documents: list[GradedDocumentInput] = []
    for index, raw_document in enumerate(raw_docs):
        documents.append(_validate_document(raw_document, index))
    return documents


def _validate_document(raw_document: Any, index: int) -> GradedDocumentInput:
    """Validate a single graded_docs entry against the expected shape."""
    if not isinstance(raw_document, dict):
        raise GenerationNodeError(f"graded_docs[{index}] must be a dictionary")

    content = raw_document.get("content")
    metadata = raw_document.get("metadata")
    score = raw_document.get("score")

    if not isinstance(content, str) or not content.strip():
        raise GenerationNodeError(f"graded_docs[{index}].content must be a non-empty string")
    if not isinstance(metadata, dict):
        raise GenerationNodeError(f"graded_docs[{index}].metadata must be a dictionary")
    if not isinstance(score, (int, float)):
        raise GenerationNodeError(f"graded_docs[{index}].score must be numeric")

    return {"content": content, "metadata": dict(metadata), "score": float(score)}


def _generate_answer_text(
    question: str,
    graded_docs: list[GradedDocumentInput],
    service: LLMService,
) -> tuple[str, bool]:
    """
    Produce the answer text, preferring the LLM and falling back safely.

    Returns:
        A tuple of (answer_text, used_fallback_generator).
    """
    context = _build_context(graded_docs)

    try:
        raw_result = service.generate_answer(question=question, context=context)  # type: ignore[call-arg]
    except (LLMServiceError, NotImplementedError, TypeError) as exc:
        logger.warning(
            "LLM answer generation unavailable (%s); using extractive fallback",
            type(exc).__name__,
        )
        return _extractive_fallback_answer(graded_docs), True
    except Exception:
        logger.exception("Unexpected error calling LLM generator; using extractive fallback")
        return _extractive_fallback_answer(graded_docs), True

    answer = _normalize_llm_answer(raw_result)
    if answer is None:
        logger.warning("LLM answer generation returned an unusable response; using extractive fallback")
        return _extractive_fallback_answer(graded_docs), True
    return answer, False


def _normalize_llm_answer(raw_result: Any) -> Optional[str]:
    """Coerce the LLM service's return value into a non-empty answer string."""
    if isinstance(raw_result, str):
        candidate = raw_result.strip()
    elif isinstance(raw_result, dict):
        candidate_raw = raw_result.get("answer")
        candidate = candidate_raw.strip() if isinstance(candidate_raw, str) else ""
    else:
        candidate = ""

    return candidate or None


def _build_context(graded_docs: list[GradedDocumentInput]) -> str:
    """Build the LLM prompt context strictly from graded_docs content."""
    sections = []
    for index, document in enumerate(graded_docs, start=1):
        sections.append(f"[{index}] {document['content'].strip()}")
    return "\n\n".join(sections)


def _extractive_fallback_answer(graded_docs: list[GradedDocumentInput]) -> str:
    """
    Deterministically build a context-grounded answer without an LLM.

    Ranks graded_docs by score (highest first, stable on ties by original
    order) and concatenates their content with inline ``[n]`` attributions,
    so the answer is provably built only from supplied context.
    """
    ranked = sorted(
        enumerate(graded_docs),
        key=lambda pair: (-pair[1]["score"], pair[0]),
    )
    top_docs = ranked[:_MAX_FALLBACK_CONTEXT_CHUNKS]

    excerpts = [f"{document['content'].strip()} [{original_index + 1}]" for original_index, document in top_docs]

    return (
        "Based on the available documentation:\n\n" + "\n\n".join(excerpts)
    )


def _build_citations(graded_docs: list[GradedDocumentInput]) -> list[Citation]:
    """Derive structural citations from graded_docs metadata, de-duplicated."""
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for document in graded_docs:
        metadata = document["metadata"]
        source_id = metadata.get("source_id")
        source_id = source_id if isinstance(source_id, str) and source_id else _UNKNOWN_SOURCE_ID

        source_title = metadata.get("source_title")
        source_title = (
            source_title if isinstance(source_title, str) and source_title else _UNKNOWN_SOURCE_TITLE
        )

        chunk_index = metadata.get("chunk_index")
        chunk_index = chunk_index if isinstance(chunk_index, int) else 0

        dedup_key = (source_id, chunk_index)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        citations.append(
            {
                "source_id": source_id,
                "source_title": source_title,
                "chunk_index": chunk_index,
            }
        )

    return citations


def _compute_confidence(state: GraphState, graded_docs: list[GradedDocumentInput]) -> Confidence:
    """
    Deterministically derive confidence from graded_docs count, retry_count,
    and fallback_used.

    Rules (evaluated in order):
        1. fallback_used truthy -> "low" (bonus fallback/web-search path).
        2. retry_count == 0 AND len(graded_docs) >= threshold -> "high"
           (first-attempt generation with solid corroborating context).
        3. Otherwise -> "medium" (required a retry, or thin context).
    """
    fallback_used = bool(state.get("fallback_used"))
    if fallback_used:
        return "low"

    retry_count = state.get("retry_count", 0)
    if retry_count == 0 and len(graded_docs) >= _MIN_GRADED_DOCS_FOR_HIGH_CONFIDENCE:
        return "high"

    return "medium"
