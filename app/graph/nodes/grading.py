"""
Document grading node.

Purpose:
    Evaluate whether retrieved chunks are actually relevant to the user's
    question. This is the self-corrective component of the RAG pipeline: its
    output drives the conditional edge that decides whether to proceed to
    generation, rewrite the query and retry, or fall back.

Responsibilities:
    - Read ``question`` and ``retrieved_docs`` from GraphState.
    - Grade each retrieved chunk as "relevant" or "irrelevant" by asking an
      LLM (``LLMService.grade_relevance``, per chunk) for a structured
      verdict, as required by the assignment.
    - Fail over to a deterministic heuristic grader (vector similarity score
      combined with lexical overlap) whenever the LLM is disabled, times out,
      or returns an unusable response, so grading - and therefore the graph -
      never hard-fails because of an LLM/provider outage.
    - Return only the state updates owned by this node: ``graded_docs`` and
      ``grading_results``.

Design notes:
    - Primary grader: ``LLMService.grade_relevance(question, chunk)`` is
      called once per retrieved chunk and asked to return structured JSON
      (``{"verdict", "confidence", "reason"}``) grounded in both the question
      and the chunk content, matching the assignment's explicit "use an LLM
      to grade each retrieved chunk" requirement.
    - Fallback grader: the original deterministic heuristic (similarity-score
      bands + keyword overlap) is kept verbatim as ``_heuristic_grade_one``.
      It is used per-chunk, transparently, whenever the LLM path raises
      ``LLMServiceError`` for that chunk, or when
      ``settings.llm_grading_enabled`` is False. Each grading_results entry
      records which grader produced it via the ``reason`` text, so a reviewer
      can see exactly how many chunks fell back and why.
    - Determinism: the LLM is called with ``temperature=0.0`` for maximal
      reproducibility, but genuine LLM non-determinism is an accepted
      trade-off of satisfying the assignment's explicit requirement; the
      heuristic fallback remains fully deterministic and is what test suites
      should exercise by injecting an ``llm_service`` that always raises.
    - ``grading_results`` uses the exact shape requested by this node's
      contract: ``{chunk_id, verdict, confidence, reason}``. This is a
      grading-node-local structure and is intentionally simpler than
      ``GraphState``'s ``GradeResult`` (which also tracks ``attempt_number``
      for cross-run traceability) -- the graph builder is responsible for
      reconciling/annotating attempt numbers when appending to state, which is
      out of scope for this node.

Public interfaces:
    - grade_documents
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Literal, Optional, TypedDict

from app.core.config import get_settings
from app.graph.state import GraphState
from app.services.llm import LLMService, LLMServiceError

logger = logging.getLogger(__name__)

Verdict = Literal["relevant", "irrelevant"]

# A relevant verdict requires reasonable vector similarity. Below this floor,
# no amount of lexical overlap can rescue a chunk -- it is almost certainly an
# unrelated part of the corpus that happened to embed nearby.
_MIN_SIMILARITY_FOR_RELEVANCE = 0.35

# Above this similarity, a chunk is considered a strong semantic candidate,
# but the heuristic fallback still requires lexical support. This avoids
# accepting generic same-corpus matches just because everything mentions
# "FastAPI".
_HIGH_SIMILARITY_THRESHOLD = 0.60

# Heuristic fallback relevance requires both enough overlapping terms and a
# sufficient fraction of the question's substantive terms to appear in the
# chunk. This is intentionally stricter than the LLM path and prevents a lone
# shared framework token (for example, "FastAPI") from marking a chunk relevant.
_MIN_OVERLAP_TERMS_FOR_RELEVANCE = 2
_MIN_OVERLAP_RATIO_FOR_RELEVANCE = 0.40

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
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


class GradingResult(TypedDict):
    """Structured grading verdict for a single retrieved chunk."""

    chunk_id: str
    verdict: Verdict
    confidence: float
    reason: str


class GradedDocument(TypedDict):
    """A retrieved document that was graded relevant, in its original shape."""

    content: str
    metadata: dict[str, Any]
    score: float


class GradingUpdate(TypedDict):
    """Partial GraphState update emitted by the grading node."""

    graded_docs: list[GradedDocument]
    grading_results: list[GradingResult]


class GradingNodeError(RuntimeError):
    """Raised when the grading node cannot safely grade retrieved documents."""


def grade_documents(
    state: GraphState,
    *,
    llm_service: Optional[LLMService] = None,
) -> GradingUpdate:
    """
    Grade retrieved document chunks and separate relevant from irrelevant context.

    Each chunk is graded by an LLM (``LLMService.grade_relevance``). If the
    LLM call fails, times out, or LLM grading is disabled via settings, that
    chunk transparently falls back to a deterministic heuristic grader so
    this node never crashes the graph over a provider outage.

    Args:
        state: Current graph state. Must contain ``question`` and
            ``retrieved_docs``.
        llm_service: Optional injected LLM service, primarily for tests
            (e.g. a fake that raises to exercise the fallback path). If
            omitted, a default ``LLMService`` is constructed.

    Returns:
        A partial state update containing only ``graded_docs`` (the relevant
        subset, in their original retrieved shape) and ``grading_results``
        (one structured verdict per retrieved chunk, relevant and irrelevant
        alike, for full traceability and future hallucination checking).
    """
    try:
        question = _extract_question(state)
        retrieved_docs = _extract_retrieved_docs(state)

        if not retrieved_docs:
            logger.info("No retrieved documents to grade")
            return {"graded_docs": [], "grading_results": []}

        settings = get_settings()
        service = llm_service if llm_service is not None else LLMService(settings)
        use_llm = settings.llm_grading_enabled

        question_terms = _extract_terms(question)

        graded_docs: list[GradedDocument] = []
        grading_results: list[GradingResult] = []
        llm_failures = 0

        for document in retrieved_docs:
            result, is_relevant, used_fallback = _grade_one(
                document,
                question,
                question_terms,
                service=service,
                use_llm=use_llm,
            )
            grading_results.append(result)
            if is_relevant:
                graded_docs.append(document)
            if used_fallback:
                llm_failures += 1

        logger.info(
            "Graded %d retrieved documents; relevant=%d irrelevant=%d heuristic_fallbacks=%d",
            len(retrieved_docs),
            len(graded_docs),
            len(retrieved_docs) - len(graded_docs),
            llm_failures,
        )
        return {"graded_docs": graded_docs, "grading_results": grading_results}
    except GradingNodeError:
        raise
    except Exception as exc:
        logger.exception("Document grading failed")
        raise GradingNodeError("Unexpected failure while grading documents") from exc


def _extract_question(state: GraphState) -> str:
    """Read and validate the original user question from GraphState."""
    raw_question = state.get("question")
    if not isinstance(raw_question, str):
        raise TypeError("GraphState.question must be a string")

    question = raw_question.strip()
    if not question:
        raise ValueError("GraphState.question must be a non-empty string")
    return question


def _extract_retrieved_docs(state: GraphState) -> list[GradedDocument]:
    """Read and validate retrieved_docs from GraphState."""
    raw_docs = state.get("retrieved_docs")
    if raw_docs is None:
        raise TypeError("GraphState.retrieved_docs must be a list")
    if not isinstance(raw_docs, list):
        raise TypeError("GraphState.retrieved_docs must be a list")

    documents: list[GradedDocument] = []
    for index, raw_document in enumerate(raw_docs):
        documents.append(_validate_document(raw_document, index))
    return documents


def _validate_document(raw_document: Any, index: int) -> GradedDocument:
    """Validate a single retrieved_docs entry against the expected shape."""
    if not isinstance(raw_document, dict):
        raise GradingNodeError(f"retrieved_docs[{index}] must be a dictionary")

    content = raw_document.get("content")
    metadata = raw_document.get("metadata")
    score = raw_document.get("score")

    if not isinstance(content, str):
        raise GradingNodeError(f"retrieved_docs[{index}].content must be a string")
    if not isinstance(metadata, dict):
        raise GradingNodeError(f"retrieved_docs[{index}].metadata must be a dictionary")
    if not isinstance(score, (int, float)):
        raise GradingNodeError(f"retrieved_docs[{index}].score must be numeric")

    return {"content": content, "metadata": dict(metadata), "score": float(score)}


def _grade_one(
    document: GradedDocument,
    question: str,
    question_terms: set[str],
    *,
    service: LLMService,
    use_llm: bool,
) -> tuple[GradingResult, bool, bool]:
    """
    Grade a single retrieved document, preferring the LLM grader.

    Returns:
        A tuple of (result, is_relevant, used_fallback) where used_fallback
        indicates the heuristic grader was used instead of the LLM.
    """
    chunk_id = _resolve_chunk_id(document)

    if use_llm:
        llm_result = _llm_grade_one(chunk_id, question, document["content"], service)
        if llm_result is not None:
            return llm_result[0], llm_result[1], False

    result, is_relevant = _heuristic_grade_one(chunk_id, document, question_terms)
    return result, is_relevant, True


def _llm_grade_one(
    chunk_id: str,
    question: str,
    content: str,
    service: LLMService,
) -> Optional[tuple[GradingResult, bool]]:
    """
    Grade one chunk via the LLM. Returns None (signalling fallback) on any
    LLM failure so the caller can seamlessly use the heuristic grader.
    """
    try:
        verdict_payload = service.grade_relevance(question=question, chunk=content)
    except LLMServiceError:
        logger.warning("LLM grading failed for chunk_id=%s; using heuristic fallback", chunk_id)
        return None
    except Exception:
        logger.exception(
            "Unexpected error calling LLM grader for chunk_id=%s; using heuristic fallback",
            chunk_id,
        )
        return None

    is_relevant = verdict_payload["verdict"] == "relevant"
    result: GradingResult = {
        "chunk_id": chunk_id,
        "verdict": verdict_payload["verdict"],
        "confidence": verdict_payload["confidence"],
        "reason": verdict_payload["reason"],
    }
    return result, is_relevant


def _heuristic_grade_one(
    chunk_id: str,
    document: GradedDocument,
    question_terms: set[str],
) -> tuple[GradingResult, bool]:
    """
    Deterministic fallback grader (vector similarity + keyword overlap).

    Used only when the LLM grader is disabled or fails for a given chunk.
    """
    content_terms = _extract_terms(document["content"])
    overlap_terms = sorted(question_terms & content_terms)
    score = document["score"]

    overlap_ratio = _overlap_ratio(overlap_terms, question_terms)

    is_relevant, confidence, reason = _decide_relevance(
        score=score,
        overlap_terms=overlap_terms,
        overlap_ratio=overlap_ratio,
    )
    reason = f"[heuristic fallback] {reason}"

    result: GradingResult = {
        "chunk_id": chunk_id,
        "verdict": "relevant" if is_relevant else "irrelevant",
        "confidence": confidence,
        "reason": reason,
    }
    return result, is_relevant


def _decide_relevance(
    score: float,
    overlap_terms: list[str],
    overlap_ratio: float,
) -> tuple[bool, float, str]:
    """
    Deterministically decide relevance from similarity score and term overlap.

    The decision is intentionally simple and explainable:
      - Below the minimum similarity floor: always irrelevant.
      - At or above the high-similarity threshold: relevant only if there is
        enough lexical support from the question terms.
      - In the mid-similarity band: relevant only if the chunk shares both at
        least two non-trivial terms and enough of the question's substantive
        term set.
    """
    clamped_score = max(0.0, min(1.0, score))

    if clamped_score < _MIN_SIMILARITY_FOR_RELEVANCE:
        confidence = round(1.0 - clamped_score, 4)
        reason = (
            f"Similarity score {clamped_score:.2f} is below the relevance "
            f"floor of {_MIN_SIMILARITY_FOR_RELEVANCE:.2f}."
        )
        return False, confidence, reason

    has_strong_overlap = (
        len(overlap_terms) >= _MIN_OVERLAP_TERMS_FOR_RELEVANCE
        and overlap_ratio >= _MIN_OVERLAP_RATIO_FOR_RELEVANCE
    )

    if clamped_score >= _HIGH_SIMILARITY_THRESHOLD and has_strong_overlap:
        confidence = round(clamped_score, 4)
        reason = (
            f"Similarity score {clamped_score:.2f} meets the high-relevance "
            f"threshold and lexical support is strong "
            f"({len(overlap_terms)} overlapping terms, overlap_ratio={overlap_ratio:.2f}). "
            f"Overlapping terms: {', '.join(overlap_terms)}."
        )
        return True, confidence, reason

    if clamped_score >= _HIGH_SIMILARITY_THRESHOLD:
        confidence = round(1.0 - overlap_ratio, 4)
        reason = (
            f"Similarity score {clamped_score:.2f} is high, but lexical support "
            f"is too weak ({len(overlap_terms)} overlapping terms, "
            f"overlap_ratio={overlap_ratio:.2f}; required at least "
            f"{_MIN_OVERLAP_TERMS_FOR_RELEVANCE} terms and "
            f"{_MIN_OVERLAP_RATIO_FOR_RELEVANCE:.2f} ratio)."
        )
        return False, confidence, reason

    if has_strong_overlap:
        confidence = round(min(1.0, (clamped_score + 0.5) / 1.5), 4)
        reason = (
            f"Moderate similarity score {clamped_score:.2f} supported by "
            f"{len(overlap_terms)} overlapping terms "
            f"(overlap_ratio={overlap_ratio:.2f}): {', '.join(overlap_terms)}."
        )
        return True, confidence, reason

    confidence = round(1.0 - clamped_score, 4)
    reason = (
        f"Moderate similarity score {clamped_score:.2f} but insufficient "
        f"lexical support ({len(overlap_terms)} overlapping terms, "
        f"overlap_ratio={overlap_ratio:.2f}; required at least "
        f"{_MIN_OVERLAP_TERMS_FOR_RELEVANCE} terms and "
        f"{_MIN_OVERLAP_RATIO_FOR_RELEVANCE:.2f} ratio)."
    )
    return False, confidence, reason


def _overlap_ratio(overlap_terms: list[str], question_terms: set[str]) -> float:
    """Return fraction of substantive question terms supported by a chunk."""
    if not question_terms:
        return 0.0
    return len(overlap_terms) / len(question_terms)


def _resolve_chunk_id(document: GradedDocument) -> str:
    """
    Resolve a stable chunk_id for a graded document.

    Prefers metadata identifiers produced by ingestion/vector-store layers
    (``source_id`` + ``chunk_index``) and falls back to a content hash so
    grading never fails just because upstream metadata is incomplete.
    """
    metadata = document["metadata"]
    source_id = metadata.get("source_id")
    chunk_index = metadata.get("chunk_index")

    if isinstance(source_id, str) and source_id and chunk_index is not None:
        return f"{source_id}:{chunk_index}"

    if isinstance(source_id, str) and source_id:
        return source_id

    content_hash = hashlib.sha1(document["content"].encode("utf-8")).hexdigest()[:16]
    logger.warning(
        "retrieved_docs entry missing source_id metadata; using content hash as chunk_id"
    )
    return f"unknown:{content_hash}"


def _extract_terms(text: str) -> set[str]:
    """Extract a normalized, stopword-filtered term set from text."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}
