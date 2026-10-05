"""
LLM service boundary (Ollama-backed, provider-agnostic interface).

Purpose:
    Provide a provider-agnostic boundary for generation, query rewriting,
    relevance grading, and optional groundedness checking.

Responsibilities:
    - Hide the concrete LLM provider's client/HTTP details behind plain
      Python methods that graph nodes call into.
    - Centralize model selection, timeout, and structured-output concerns.
    - Expose high-level operations used by graph nodes.

Current provider:
    Local Ollama server (no API key required). Calls are made over Ollama's
    REST API (``/api/chat``) using ``format="json"`` for structured output,
    so swapping to a hosted provider later only requires changing
    ``_call_chat`` - callers of ``grade_relevance``/``generate_answer``/etc.
    are unaffected.

Public interfaces:
    - LLMService
    - LLMServiceError
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when an LLM call fails or returns an unusable response."""


_GRADE_RELEVANCE_SYSTEM_PROMPT = (
    "You are a strict relevance grader for a retrieval-augmented generation "
    "system. Given a user QUESTION and a retrieved document CHUNK, decide "
    "whether the chunk contains information that helps answer the question. "
    "Respond ONLY with a JSON object of the exact shape: "
    '{"verdict": "relevant" | "irrelevant", "confidence": <float 0.0-1.0>, '
    '"reason": "<one short sentence>"}. '
    "Be conservative: mark a chunk irrelevant unless it clearly relates to "
    "the question's subject matter."
)

_GRADE_RELEVANCE_BATCH_SYSTEM_PROMPT = (
    "You are a strict relevance grader for a retrieval-augmented generation "
    "system. Given a user QUESTION and multiple retrieved CHUNKS (each with "
    "a chunk_id), decide whether each chunk helps answer the question. "
    "Respond ONLY with a JSON object of the exact shape: "
    '{"results": [{"chunk_id": "<id>", "verdict": "relevant" | "irrelevant", '
    '"confidence": <float 0.0-1.0>, "reason": "<one short sentence>"}, ...]}. '
    "Include exactly one result per chunk_id provided, in any order. "
    "Be conservative: mark a chunk irrelevant unless it clearly relates to "
    "the question's subject matter."
)

_GENERATE_ANSWER_SYSTEM_PROMPT = (
    "You are a retrieval-augmented AI assistant.\n\n"
    "Use ONLY the supplied context.\n\n"
    "Do not invent facts.\n\n"
    "If the answer is not supported by the context, explicitly say that the "
    "context does not contain enough information.\n\n"
    "Provide concise synthesized answers.\n\n"
    "If the user asks for bullets, return bullets."
)


class LLMService:
    """Provider-agnostic LLM client boundary, currently backed by Ollama."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Create an LLM service.

        Args:
            settings: Optional injected Settings, primarily for tests. If
                omitted, the process-wide cached Settings are used.
        """
        self._settings = settings or get_settings()

    def rewrite_query(self) -> None:
        """Rewrite or expand a user question for retrieval. Not yet implemented."""
        raise NotImplementedError("rewrite_query is implemented by query_analysis for now")

    def classify_query(self) -> None:
        """Classify a user question into the workflow query_type labels. Not yet implemented."""
        raise NotImplementedError("classify_query is implemented by query_analysis for now")

    def grade_relevance(self, question: str, chunk: str) -> dict[str, Any]:
        """
        Grade a single (question, chunk) pair for relevance using the LLM.

        Args:
            question: The user's original question.
            chunk: The retrieved document chunk's text content.

        Returns:
            A dict of the shape ``{"verdict": "relevant" | "irrelevant",
            "confidence": float, "reason": str}``.

        Raises:
            LLMServiceError: if the LLM call fails, times out, or returns a
                response that cannot be parsed into the expected shape.
                Callers (graph/nodes/grading.py) are expected to catch this
                and fail over to the deterministic heuristic grader.
        """
        user_prompt = (
            f"QUESTION:\n{question}\n\nCHUNK:\n{chunk}\n\n"
            "Return only the JSON verdict object described in your instructions."
        )
        raw_content = self._call_chat(
            system_prompt=_GRADE_RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return self._parse_grade_response(raw_content)

    def grade_relevance_batch(
        self,
        question: str,
        items: list[tuple[str, str]],
    ) -> dict[str, dict[str, Any]]:
        """
        Grade multiple (chunk_id, content) pairs in a single LLM call.

        Args:
            question: The user's original question.
            items: ``(chunk_id, chunk_text)`` pairs to grade.

        Returns:
            Mapping of ``chunk_id`` to
            ``{"verdict", "confidence", "reason"}`` for each input id.

        Raises:
            LLMServiceError: if the LLM call fails or the response cannot be
                parsed into the expected batched shape.
        """
        if not items:
            return {}

        if len(items) == 1:
            chunk_id, chunk = items[0]
            return {chunk_id: self.grade_relevance(question, chunk)}

        chunks_block = "\n\n".join(
            f'CHUNK_ID: "{chunk_id}"\n{content.strip()}' for chunk_id, content in items
        )
        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"CHUNKS ({len(items)} total):\n{chunks_block}\n\n"
            "Return only the JSON results object described in your instructions."
        )
        raw_content = self._call_chat(
            system_prompt=_GRADE_RELEVANCE_BATCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return self._parse_batch_grade_response(raw_content, expected_ids=[cid for cid, _ in items])

    def generate_answer(self, question: str, context: str) -> dict[str, str]:
        """
        Generate a grounded answer from retrieved context using the LLM.

        Args:
            question: The user's original question.
            context: Concatenated text from graded document chunks.

        Returns:
            A dict of the shape ``{"answer": "<non-empty string>"}``.

        Raises:
            LLMServiceError: if the LLM call fails, times out, or returns a
                response that cannot be parsed into the expected shape.
                Callers (graph/nodes/generation.py) are expected to catch this
                and fail over to the deterministic extractive generator.
        """
        user_prompt = (
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
            "Return ONLY JSON:\n\n"
            '{\n  "answer": "..."\n}'
        )
        raw_content = self._call_chat(
            system_prompt=_GENERATE_ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return self._parse_answer_response(raw_content)

    def check_grounding(self) -> None:
        """Check whether an answer is supported by context. Not yet implemented."""
        raise NotImplementedError("check_grounding is implemented by the hallucination_check node")

    def _call_chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the local Ollama chat endpoint with JSON-mode structured output.

        Returns the raw string content of the model's reply.
        """
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self._settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }

        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=self._settings.llm_request_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Ollama chat call failed: %s", exc)
            raise LLMServiceError("Ollama chat call failed") from exc

        try:
            body = response.json()
            content = body["message"]["content"]
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Ollama chat response had an unexpected shape")
            raise LLMServiceError("Ollama chat response had an unexpected shape") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError("Ollama chat response content was empty")
        return content

    def _parse_grade_response(self, raw_content: str) -> dict[str, Any]:
        """Parse and validate the JSON grading payload returned by the LLM."""
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            logger.warning("LLM grading response was not valid JSON: %r", raw_content)
            raise LLMServiceError("LLM grading response was not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise LLMServiceError("LLM grading response must be a JSON object")

        verdict = parsed.get("verdict")
        confidence = parsed.get("confidence")
        reason = parsed.get("reason")

        if verdict not in ("relevant", "irrelevant"):
            raise LLMServiceError(f"LLM grading response had invalid verdict: {verdict!r}")
        if not isinstance(confidence, (int, float)):
            raise LLMServiceError("LLM grading response had a non-numeric confidence")
        if not isinstance(reason, str) or not reason.strip():
            raise LLMServiceError("LLM grading response had an empty reason")

        return {
            "verdict": verdict,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "reason": reason.strip(),
        }

    def _parse_batch_grade_response(
        self,
        raw_content: str,
        *,
        expected_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Parse and validate a batched grading payload returned by the LLM."""
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            logger.warning("LLM batch grading response was not valid JSON: %r", raw_content)
            raise LLMServiceError("LLM batch grading response was not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise LLMServiceError("LLM batch grading response must be a JSON object")

        raw_results = parsed.get("results")
        if not isinstance(raw_results, list):
            raise LLMServiceError("LLM batch grading response missing results list")

        by_id: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(raw_results):
            if not isinstance(entry, dict):
                raise LLMServiceError(f"LLM batch grading results[{index}] must be an object")
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise LLMServiceError(f"LLM batch grading results[{index}] missing chunk_id")
            verdict = entry.get("verdict")
            confidence = entry.get("confidence")
            reason = entry.get("reason")
            if verdict not in ("relevant", "irrelevant"):
                raise LLMServiceError(
                    f"LLM batch grading results[{index}] had invalid verdict: {verdict!r}"
                )
            if not isinstance(confidence, (int, float)):
                raise LLMServiceError(
                    f"LLM batch grading results[{index}] had non-numeric confidence"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise LLMServiceError(f"LLM batch grading results[{index}] had empty reason")
            by_id[chunk_id.strip()] = {
                "verdict": verdict,
                "confidence": max(0.0, min(1.0, float(confidence))),
                "reason": reason.strip(),
            }

        missing = [chunk_id for chunk_id in expected_ids if chunk_id not in by_id]
        if missing:
            raise LLMServiceError(
                f"LLM batch grading response missing chunk_ids: {missing[:5]}"
            )
        return by_id

    def _parse_answer_response(self, raw_content: str) -> dict[str, str]:
        """Parse and validate the JSON answer payload returned by the LLM."""
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            logger.warning("LLM answer response was not valid JSON: %r", raw_content)
            raise LLMServiceError("LLM answer response was not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise LLMServiceError("LLM answer response must be a JSON object")

        answer = parsed.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise LLMServiceError("LLM answer response had an empty answer")

        return {"answer": answer.strip()}
