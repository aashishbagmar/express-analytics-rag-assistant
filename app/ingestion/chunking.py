"""
Chunking strategy for the ingestion-only phase.

Purpose:
    Split loaded LangChain ``Document`` objects into retrieval-friendly chunks
    while preserving source metadata for citations.

Responsibilities:
    - Use RecursiveCharacterTextSplitter with chunk_size=800 and overlap=100.
    - Preserve all metadata from loaded documents.
    - Add ``chunk_index`` and ``section_heading`` metadata to each chunk.
    - Avoid embeddings, vector stores, FastAPI, LangGraph, and LLM calls.

Public interfaces:
    - DocumentChunker.split
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - compatibility with older LangChain installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SectionHeading:
    """A heading and its character offset in a source document."""

    offset: int
    text: str


class DocumentChunker:
    """Split source documents into metadata-rich chunks."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        """Create a technical-document chunker with assignment-required defaults."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(self, documents: Iterable[Document]) -> list[Document]:
        """
        Split loaded documents into chunks.

        Bad or empty documents are skipped with warnings so one malformed source
        never crashes the ingestion run.
        """
        chunked_documents: list[Document] = []

        for document in documents:
            try:
                chunked_documents.extend(self._split_one(document))
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.warning(
                    "Skipping document during chunking; source_path=%s error=%s",
                    document.metadata.get("source_path"),
                    exc,
                )

        logger.info("Created %d chunks from loaded documents", len(chunked_documents))
        return chunked_documents

    def _split_one(self, document: Document) -> list[Document]:
        """Split one source document and attach per-chunk metadata."""
        if not document.page_content.strip():
            logger.warning(
                "Skipping empty document during chunking; source_path=%s",
                document.metadata.get("source_path"),
            )
            return []

        headings = list(self._extract_markdown_headings(document.page_content))
        raw_chunks = self._splitter.split_documents([document])
        chunks: list[Document] = []

        for index, chunk in enumerate(raw_chunks):
            start_index = self._resolve_start_index(document.page_content, chunk)
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = index
            metadata["section_heading"] = self._section_heading_for_offset(
                headings=headings,
                offset=start_index,
            )

            chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        logger.info(
            "Chunked document source_id=%s into %d chunks",
            document.metadata.get("source_id"),
            len(chunks),
        )
        return chunks

    def _extract_markdown_headings(self, text: str) -> Iterable[SectionHeading]:
        """Extract Markdown-style headings from source text."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
        for match in heading_pattern.finditer(text):
            yield SectionHeading(offset=match.start(), text=match.group(2).strip())

    def _resolve_start_index(self, source_text: str, chunk: Document) -> int:
        """
        Resolve a chunk's start offset in the original document.

        RecursiveCharacterTextSplitter is configured with add_start_index=True,
        but this fallback keeps section-heading inference stable if a LangChain
        version omits that metadata.
        """
        raw_start = chunk.metadata.get("start_index")
        if isinstance(raw_start, int):
            return raw_start

        found = source_text.find(chunk.page_content)
        if found >= 0:
            return found

        logger.debug("Could not determine start_index for chunk; defaulting to 0")
        return 0

    def _section_heading_for_offset(
        self,
        headings: list[SectionHeading],
        offset: int,
    ) -> Optional[str]:
        """Return the nearest preceding section heading, or None if unknown."""
        section_heading: Optional[str] = None
        for heading in headings:
            if heading.offset > offset:
                break
            section_heading = heading.text
        return section_heading
