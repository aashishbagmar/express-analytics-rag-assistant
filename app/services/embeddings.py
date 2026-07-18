"""
Embedding service for local sentence-transformer embeddings.

Purpose:
    Provide one production-quality boundary for embedding document chunks and
    user queries without leaking provider details into ingestion, retrieval, or
    graph code.

Responsibilities:
    - Use sentence-transformers with the default ``all-MiniLM-L6-v2`` model.
    - Load the model lazily on first embedding call, not at module import time.
    - Share one model instance per model name across service instances.
    - Expose a stable model identifier for index compatibility checks.
    - Avoid vector-store, retrieval, LangGraph, FastAPI, LLM, grading, or
      generation logic.

Public interfaces:
    - EmbeddingService.embed_documents
    - EmbeddingService.embed_query
    - EmbeddingService.model_id
"""

from __future__ import annotations

import logging
import threading
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when embeddings cannot be generated safely."""


class EmbeddingService:
    """SentenceTransformer-backed embedding service with lazy shared loading."""

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    _model_lock: ClassVar[threading.Lock] = threading.Lock()
    _models: ClassVar[dict[str, Any]] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Create an embedding service.

        Args:
            model_name: SentenceTransformer model name. Defaults to
                ``all-MiniLM-L6-v2`` as required by the assignment.
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self._model_name = model_name.strip()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed document chunk texts.

        Returns one embedding per input text. Empty input returns an empty list
        without loading the model.
        """
        self._validate_texts(texts)
        if not texts:
            logger.info("No document texts supplied for embedding")
            return []

        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return self._to_float_matrix(embeddings)
        except Exception as exc:
            logger.exception("Failed to embed %d document texts", len(texts))
            raise EmbeddingServiceError("Failed to embed document texts") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed one user query string."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            model = self._get_model()
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return self._to_float_vector(embedding)
        except Exception as exc:
            logger.exception("Failed to embed query text")
            raise EmbeddingServiceError("Failed to embed query text") from exc

    def model_id(self) -> str:
        """Return the embedding model identifier used for compatibility checks."""
        return self._model_name

    def _get_model(self) -> Any:
        """Return the shared model instance, loading it once per model name."""
        cached_model = self._models.get(self._model_name)
        if cached_model is not None:
            return cached_model

        with self._model_lock:
            cached_model = self._models.get(self._model_name)
            if cached_model is not None:
                return cached_model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                logger.exception("sentence-transformers is not installed")
                raise EmbeddingServiceError(
                    "sentence-transformers is required for EmbeddingService"
                ) from exc

            try:
                logger.info("Loading embedding model: %s", self._model_name)
                model = SentenceTransformer(self._model_name)
            except Exception as exc:
                logger.exception("Failed to load embedding model: %s", self._model_name)
                raise EmbeddingServiceError(
                    f"Failed to load embedding model: {self._model_name}"
                ) from exc

            self._models[self._model_name] = model
            logger.info("Embedding model loaded: %s", self._model_name)
            return model

    def _validate_texts(self, texts: list[str]) -> None:
        """Validate document embedding inputs before model execution."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        invalid_indexes = [
            index for index, text in enumerate(texts) if not isinstance(text, str)
        ]
        if invalid_indexes:
            raise TypeError(
                f"texts must contain only strings; invalid indexes={invalid_indexes}"
            )

    def _to_float_matrix(self, embeddings: Any) -> list[list[float]]:
        """Convert SentenceTransformer output into a JSON-serializable matrix."""
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [[float(value) for value in vector] for vector in embeddings]

    def _to_float_vector(self, embedding: Any) -> list[float]:
        """Convert SentenceTransformer output into a JSON-serializable vector."""
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        return [float(value) for value in embedding]
