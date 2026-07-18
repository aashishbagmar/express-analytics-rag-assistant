"""
Ingestion pipeline for the load-and-chunk phase only.

Purpose:
    Orchestrate the assignment-required first half of ingestion:
    load local documents -> split into chunks -> return LangChain Documents.

Responsibilities:
    - Validate input paths.
    - Load Markdown, text, and HTML documents from disk.
    - Split loaded documents into metadata-rich chunks.
    - Skip unsupported/bad files safely with logging.
    - Avoid embeddings, vector stores, FastAPI, LangGraph, LLM calls, grading,
      retrieval, and generation.

Public interfaces:
    - IngestionPipeline.ingest
    - IngestionPipeline.ingest_file
    - IngestionPipeline.ingest_directory
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.chunking import DocumentChunker
from app.ingestion.loaders import DocumentLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Coordinate local document loading and chunking."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        """Create a pipeline with injectable loader/chunker dependencies."""
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()

    def ingest(self, path: str | Path) -> list[Document]:
        """
        Load and chunk a file or directory.

        Example:
            pipeline = IngestionPipeline()
            chunks = pipeline.ingest("data/corpus")
        """
        input_path = Path(path)
        if not input_path.exists():
            logger.warning("Ingestion path does not exist: %s", input_path)
            return []

        if input_path.is_file():
            return self.ingest_file(input_path)
        if input_path.is_dir():
            return self.ingest_directory(input_path)

        logger.warning("Ingestion path is neither file nor directory: %s", input_path)
        return []

    def ingest_file(self, path: str | Path) -> list[Document]:
        """Load and chunk one local file."""
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("Ingestion file does not exist: %s", file_path)
            return []
        if not file_path.is_file():
            logger.warning("Ingestion path is not a file: %s", file_path)
            return []

        documents = self.loader.load_file(file_path)
        if not documents:
            return []

        chunks = self.chunker.split(documents)
        logger.info("Ingested file %s into %d chunks", file_path, len(chunks))
        return chunks

    def ingest_directory(self, path: str | Path) -> list[Document]:
        """Load and chunk supported files from a local directory recursively."""
        directory = Path(path)
        if not directory.exists():
            logger.warning("Ingestion directory does not exist: %s", directory)
            return []
        if not directory.is_dir():
            logger.warning("Ingestion path is not a directory: %s", directory)
            return []

        documents = self.loader.load_directory(directory)
        if not documents:
            return []

        chunks = self.chunker.split(documents)
        logger.info("Ingested directory %s into %d chunks", directory, len(chunks))
        return chunks

    def ingest_corpus(self, path: str | Path = "data/corpus") -> list[Document]:
        """Load and chunk the default local corpus directory."""
        return self.ingest_directory(path)
