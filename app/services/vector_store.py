"""
Persistent ChromaDB vector-store service.

Purpose:
    Encapsulate local ChromaDB indexing and similarity search behind a stable
    service interface for future ingestion endpoints, retrieval nodes, and
    documents endpoints.

Responsibilities:
    - Persist embedded LangChain Documents into Chroma.
    - Preserve source/chunk metadata as faithfully as Chroma allows.
    - Delegate embedding generation to ``EmbeddingService``.
    - Return chunk content, restored metadata, and similarity scores.
    - Avoid LangGraph, retrieval-node, grading, generation, and FastAPI logic.

Public interfaces:
    - VectorStoreService.build_index
    - VectorStoreService.add_documents
    - VectorStoreService.replace_documents_for_sources
    - VectorStoreService.similarity_search
    - VectorStoreService.list_documents
    - VectorStoreService.document_count
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document

from app.services.embeddings import EmbeddingService, EmbeddingServiceError

logger = logging.getLogger(__name__)


class VectorStoreServiceError(RuntimeError):
    """Raised when Chroma vector-store operations fail safely."""


class VectorStoreService:
    """Persistent ChromaDB-backed vector store for LangChain Documents."""

    DEFAULT_PERSIST_DIRECTORY = Path("app/storage/chroma_db")
    DEFAULT_COLLECTION_NAME = "technical_docs"

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        """
        Create a persistent Chroma vector-store service.

        Args:
            persist_directory: Directory where Chroma persists its local index.
            collection_name: Chroma collection name.
            embedding_service: Embedding boundary used for documents and queries.
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name.strip()
        self.embedding_service = embedding_service or EmbeddingService()
        self._client: Any | None = None
        self._collection: Any | None = None

    def build_index(self, documents: list[Document]) -> int:
        """
        Rebuild the collection from scratch and index the provided documents.

        Returns:
            Number of documents/chunks indexed.
        """
        self._validate_documents(documents)

        try:
            client = self._get_client()
            try:
                client.delete_collection(name=self.collection_name)
                logger.info("Deleted existing Chroma collection: %s", self.collection_name)
            except Exception:
                logger.info(
                    "No existing Chroma collection to delete: %s",
                    self.collection_name,
                )

            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"embedding_model": self.embedding_service.model_id()},
            )
            logger.info("Recreated Chroma collection: %s", self.collection_name)
            return self.add_documents(documents)
        except VectorStoreServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to build Chroma index")
            raise VectorStoreServiceError("Failed to build vector index") from exc

    def add_documents(self, documents: list[Document]) -> int:
        """
        Add or update LangChain Documents in the persistent Chroma collection.

        Returns:
            Number of documents/chunks indexed.
        """
        self._validate_documents(documents)
        if not documents:
            logger.info("No documents supplied for vector-store indexing")
            return 0

        try:
            collection = self._get_collection()
            texts = [document.page_content for document in documents]
            embeddings = self.embedding_service.embed_documents(texts)
            ids = [self._document_id(document) for document in documents]
            metadatas = [self._sanitize_metadata(document.metadata) for document in documents]

            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.info(
                "Indexed %d documents/chunks into Chroma collection %s",
                len(documents),
                self.collection_name,
            )
            return len(documents)
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to add documents to Chroma collection")
            raise VectorStoreServiceError("Failed to add documents to vector store") from exc

    def replace_documents_for_sources(self, documents: list[Document]) -> int:
        """
        Replace indexed chunks for the source documents present in ``documents``.

        Any existing chunks whose ``source_id`` appears in the incoming batch
        are removed first, then the new chunks are upserted. Chunks belonging
        to other source documents are left untouched. This supports incremental
        corpus re-ingestion without wiping separately uploaded documents.
        """
        self._validate_documents(documents)
        if not documents:
            logger.info("No documents supplied for source-scoped reindex")
            return 0

        source_ids = self._extract_source_ids(documents)
        try:
            self.delete_chunks_for_source_ids(source_ids)
        except VectorStoreServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to delete existing chunks for source reindex")
            raise VectorStoreServiceError("Failed to replace source documents") from exc

        return self.add_documents(documents)

    def delete_chunks_for_source_ids(self, source_ids: list[str]) -> int:
        """Delete all indexed chunks whose metadata ``source_id`` is in ``source_ids``."""
        normalized_ids = [source_id for source_id in dict.fromkeys(source_ids) if source_id]
        if not normalized_ids:
            return 0

        try:
            collection = self._get_collection()
            if self.document_count() == 0:
                return 0

            existing = collection.get(include=["metadatas"])
            ids_to_delete: list[str] = []
            for chunk_id, raw_metadata in zip(
                existing.get("ids") or [],
                existing.get("metadatas") or [],
            ):
                metadata = self._restore_metadata(raw_metadata or {})
                if str(metadata.get("source_id") or "") in normalized_ids:
                    ids_to_delete.append(chunk_id)

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(
                    "Deleted %d existing chunks for source_ids=%s",
                    len(ids_to_delete),
                    normalized_ids,
                )
            return len(ids_to_delete)
        except Exception as exc:
            logger.exception("Failed to delete chunks for source_ids=%s", normalized_ids)
            raise VectorStoreServiceError("Failed to delete chunks for source documents") from exc

    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Search indexed chunks by semantic similarity.

        Returns:
            A list of dictionaries with:
                - content: chunk text
                - metadata: restored chunk/source metadata
                - score: normalized score derived from Chroma distance
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        try:
            collection = self._get_collection()
            if self.document_count() == 0:
                logger.warning("Similarity search requested on an empty Chroma collection")
                return []

            query_embedding = self.embedding_service.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0] or []
            metadatas = results.get("metadatas", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []

            matches: list[dict[str, Any]] = []
            for content, metadata, distance in zip(documents, metadatas, distances):
                matches.append(
                    {
                        "content": content,
                        "metadata": self._restore_metadata(metadata or {}),
                        "score": self._distance_to_similarity(distance),
                    }
                )

            logger.info(
                "Similarity search returned %d results from collection %s",
                len(matches),
                self.collection_name,
            )
            return matches
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to run similarity search")
            raise VectorStoreServiceError("Failed to run similarity search") from exc

    def list_documents(self) -> list[dict[str, Any]]:
        """
        Return unique source-document summaries from indexed chunk metadata.

        This supports the future ``GET /documents`` endpoint without exposing
        raw Chroma collection internals.
        """
        try:
            collection = self._get_collection()
            raw = collection.get(include=["metadatas"])
            metadatas = raw.get("metadatas") or []

            summaries: dict[str, dict[str, Any]] = {}
            for raw_metadata in metadatas:
                metadata = self._restore_metadata(raw_metadata or {})
                source_id = str(metadata.get("source_id") or "unknown")

                summary = summaries.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "source_title": metadata.get("source_title"),
                        "filename": metadata.get("filename"),
                        "file_type": metadata.get("file_type"),
                        "pages": metadata.get("page_count"),
                        "source_path": metadata.get("source_path"),
                        "chunk_count": 0,
                    },
                )
                if metadata.get("filename") and not summary.get("filename"):
                    summary["filename"] = metadata.get("filename")
                if metadata.get("file_type") and not summary.get("file_type"):
                    summary["file_type"] = metadata.get("file_type")
                if metadata.get("page_count") is not None and summary.get("pages") is None:
                    summary["pages"] = metadata.get("page_count")
                if metadata.get("source_title") and not summary.get("source_title"):
                    summary["source_title"] = metadata.get("source_title")
                summary["chunk_count"] += 1

            return sorted(
                summaries.values(),
                key=lambda item: (str(item.get("source_title")), str(item.get("source_id"))),
            )
        except Exception as exc:
            logger.exception("Failed to list indexed documents")
            raise VectorStoreServiceError("Failed to list indexed documents") from exc

    def document_count(self) -> int:
        """Return the number of indexed chunks in the Chroma collection."""
        try:
            return int(self._get_collection().count())
        except Exception as exc:
            logger.exception("Failed to count indexed documents")
            raise VectorStoreServiceError("Failed to count indexed documents") from exc

    def _get_client(self) -> Any:
        """Return a persistent Chroma client, creating storage lazily."""
        if self._client is not None:
            return self._client

        try:
            import chromadb
        except ImportError as exc:
            logger.exception("chromadb is not installed")
            raise VectorStoreServiceError("chromadb is required for VectorStoreService") from exc

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            logger.info("Initialized Chroma persistent client at %s", self.persist_directory)
            return self._client
        except Exception as exc:
            logger.exception("Failed to initialize Chroma persistent client")
            raise VectorStoreServiceError("Failed to initialize Chroma client") from exc

    def _get_collection(self) -> Any:
        """Return the configured Chroma collection, creating it lazily."""
        if self._collection is not None:
            return self._collection

        try:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name,
                metadata={"embedding_model": self.embedding_service.model_id()},
            )
            return self._collection
        except Exception as exc:
            logger.exception("Failed to get or create Chroma collection")
            raise VectorStoreServiceError("Failed to get or create Chroma collection") from exc

    def _validate_documents(self, documents: list[Document]) -> None:
        """Validate LangChain Document inputs before indexing."""
        if not isinstance(documents, list):
            raise TypeError("documents must be a list of LangChain Document objects")

        invalid_indexes: list[int] = []
        for index, document in enumerate(documents):
            if not isinstance(document, Document) or not document.page_content.strip():
                invalid_indexes.append(index)

        if invalid_indexes:
            raise ValueError(
                "documents must contain non-empty LangChain Documents; "
                f"invalid indexes={invalid_indexes}"
            )

    def _document_id(self, document: Document) -> str:
        """Create a deterministic chunk id for idempotent Chroma upserts."""
        metadata = document.metadata or {}
        source_id = str(metadata.get("source_id") or "unknown-source")
        chunk_index = str(metadata.get("chunk_index") or 0)
        content_hash = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()[:16]
        return f"{source_id}:{chunk_index}:{content_hash}"

    def _extract_source_ids(self, documents: list[Document]) -> list[str]:
        """Return unique source_ids from a document batch in first-seen order."""
        source_ids: list[str] = []
        seen: set[str] = set()
        for document in documents:
            metadata = document.metadata or {}
            source_id = str(metadata.get("source_id") or "").strip()
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            source_ids.append(source_id)
        return source_ids

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """
        Convert metadata to Chroma-compatible scalar values.

        Chroma metadata accepts only str/int/float/bool values. To preserve
        richer metadata faithfully, None values and JSON-serializable structures
        are marked so they can be restored when read back.
        """
        sanitized: dict[str, str | int | float | bool] = {}
        none_keys: list[str] = []
        json_keys: list[str] = []

        for key, value in (metadata or {}).items():
            safe_key = str(key)
            if value is None:
                sanitized[safe_key] = ""
                none_keys.append(safe_key)
            elif isinstance(value, (str, int, float, bool)):
                sanitized[safe_key] = value
            else:
                sanitized[safe_key] = json.dumps(value, ensure_ascii=True)
                json_keys.append(safe_key)

        if none_keys:
            sanitized["_none_metadata_keys"] = json.dumps(none_keys, ensure_ascii=True)
        if json_keys:
            sanitized["_json_metadata_keys"] = json.dumps(json_keys, ensure_ascii=True)
        sanitized["_embedding_model"] = self.embedding_service.model_id()
        return sanitized

    def _restore_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Restore metadata values encoded for Chroma compatibility."""
        restored = dict(metadata or {})

        none_keys = self._load_metadata_key_list(restored.pop("_none_metadata_keys", "[]"))
        json_keys = self._load_metadata_key_list(restored.pop("_json_metadata_keys", "[]"))

        for key in json_keys:
            if key in restored:
                try:
                    restored[key] = json.loads(restored[key])
                except (TypeError, json.JSONDecodeError):
                    logger.warning("Could not JSON-restore metadata key: %s", key)

        for key in none_keys:
            restored[key] = None

        return restored

    def _load_metadata_key_list(self, raw_value: Any) -> list[str]:
        """Decode internal metadata key lists safely."""
        if not raw_value:
            return []
        try:
            decoded = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(decoded, list):
            return []
        return [str(item) for item in decoded]

    def _distance_to_similarity(self, distance: Any) -> float:
        """
        Convert a Chroma distance value into a stable similarity score.

        Chroma distance semantics depend on collection configuration. This
        monotonic transformation keeps the score bounded in (0, 1] while
        preserving ordering: smaller distance means larger similarity.
        """
        if distance is None:
            return 0.0
        numeric_distance = max(float(distance), 0.0)
        return 1.0 / (1.0 + numeric_distance)
