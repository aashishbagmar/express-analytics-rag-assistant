"""
Document loaders for the ingestion-only phase.

Purpose:
    Load supported local technical documentation files into LangChain
    ``Document`` objects while preserving citation-ready source metadata.

Responsibilities:
    - Support Markdown (``.md``), text (``.txt``), HTML (``.html``/``.htm``),
      PDF (``.pdf``), and Word (``.docx``).
    - Ignore unsupported files gracefully.
    - Log warnings instead of crashing ingestion on one bad file.
    - Avoid chunking, embeddings, vector stores, FastAPI, LangGraph, and LLM calls.

Public interfaces:
    - DocumentLoader.load_file
    - DocumentLoader.load_directory
"""

from __future__ import annotations

import hashlib
import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import fitz
from docx import Document as DocxDocument
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """Small standard-library HTML-to-text extractor for local documentation."""

    _BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        """Create an extractor with an internal text buffer."""
        super().__init__()
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track ignored tags and add spacing before block elements."""
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Track ignored tags and add spacing after block elements."""
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Collect visible text data."""
        if self._ignored_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)
            self._parts.append(" ")

    def text(self) -> str:
        """Return normalized visible text."""
        lines = (" ".join(part.split()) for part in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


class DocumentLoader:
    """Load supported local documentation files as LangChain Documents."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".pdf", ".docx"}

    def load_file(self, path: str | Path) -> list[Document]:
        """
        Load one supported file from disk.

        Unsupported, missing, unreadable, or empty files are skipped with a
        warning and return an empty list so one bad file never fails ingestion.
        """
        file_path = Path(path)

        if not file_path.exists():
            logger.warning("Skipping missing file: %s", file_path)
            return []
        if not file_path.is_file():
            logger.warning("Skipping non-file path: %s", file_path)
            return []
        if not self.is_supported(file_path):
            logger.warning("Skipping unsupported file type: %s", file_path)
            return []

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                content, extra_metadata = self._load_pdf(file_path)
            elif suffix == ".docx":
                content, extra_metadata = self._load_docx(file_path)
            else:
                content = self._load_text(file_path)
                extra_metadata = {}
        except OSError as exc:
            logger.warning("Could not read file %s: %s", file_path, exc)
            return []

        if not content.strip():
            logger.warning("Skipping empty document after normalization: %s", file_path)
            return []

        metadata = self._build_metadata(file_path)
        metadata.update(extra_metadata)
        if "file_type" not in metadata:
            metadata["file_type"] = file_path.suffix.lower().lstrip(".")
        logger.info("Loaded document: %s", file_path)
        return [Document(page_content=content, metadata=metadata)]

    def load_directory(self, path: str | Path) -> list[Document]:
        """
        Load all supported files below a directory, recursively.

        Unsupported files are ignored with a warning from ``load_file``.
        Results are returned in deterministic path order for reproducible tests.
        """
        directory = Path(path)
        if not directory.exists():
            logger.warning("Ingestion path does not exist: %s", directory)
            return []
        if not directory.is_dir():
            logger.warning("Ingestion path is not a directory: %s", directory)
            return []

        documents: list[Document] = []
        for file_path in self._iter_files(directory):
            documents.extend(self.load_file(file_path))

        logger.info("Loaded %d documents from directory: %s", len(documents), directory)
        return documents

    def is_supported(self, path: str | Path) -> bool:
        """Return whether the file extension is supported by this loader."""
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _iter_files(self, directory: Path) -> Iterable[Path]:
        """Yield files recursively in deterministic order."""
        return sorted(path for path in directory.rglob("*") if path.is_file())

    def _load_text(self, file_path: Path) -> str:
        """Read and normalize a text-based documentation file."""
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = file_path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as exc:
                raise OSError(f"could not decode file: {exc}") from exc
        return self._normalize_content(raw_text, file_path.suffix.lower())

    def _load_pdf(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Extract text and page metadata from a PDF file."""
        try:
            with fitz.open(file_path) as pdf_doc:
                page_count = pdf_doc.page_count
                parts: list[str] = []
                for page in pdf_doc:
                    page_text = page.get_text().strip()
                    if page_text:
                        parts.append(page_text)
        except (OSError, RuntimeError, ValueError) as exc:
            raise OSError(f"could not read PDF: {exc}") from exc

        content = "\n\n".join(parts)
        return content, {"page_count": page_count, "file_type": "pdf"}

    def _load_docx(self, file_path: Path) -> tuple[str, dict[str, str]]:
        """Extract paragraph text from a Word document in document order."""
        try:
            docx_doc = DocxDocument(str(file_path))
            paragraphs = [paragraph.text.strip() for paragraph in docx_doc.paragraphs if paragraph.text.strip()]
        except (OSError, ValueError) as exc:
            raise OSError(f"could not read DOCX: {exc}") from exc

        content = "\n".join(paragraphs)
        return content, {"file_type": "docx"}

    def _normalize_content(self, text: str, suffix: str) -> str:
        """Normalize raw file contents for downstream chunking."""
        if suffix in {".html", ".htm"}:
            extractor = _HTMLTextExtractor()
            extractor.feed(text)
            return extractor.text()
        return text.strip()

    def _build_metadata(self, file_path: Path) -> dict[str, Any]:
        """Build source metadata required by the retrieval/generation schema."""
        resolved_path = file_path.resolve()
        source_path = str(resolved_path)
        source_id = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:16]

        return {
            "source_id": source_id,
            "source_title": file_path.stem,
            "source_path": source_path,
            "filename": file_path.name,
        }
