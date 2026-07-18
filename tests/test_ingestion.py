from __future__ import annotations

import fitz
from docx import Document as DocxDocument
from langchain_core.documents import Document

from app.ingestion.chunking import DocumentChunker
from app.ingestion.loaders import DocumentLoader
from app.ingestion.pipeline import IngestionPipeline


def _write_sample_pdf(path) -> None:
    pdf = fitz.open()
    page_one = pdf.new_page()
    page_one.insert_text((72, 72), "PDF page one content.")
    page_two = pdf.new_page()
    page_two.insert_text((72, 72), "PDF page two content.")
    pdf.save(path)
    pdf.close()


def _write_sample_docx(path) -> None:
    docx = DocxDocument()
    docx.add_paragraph("First DOCX paragraph.")
    docx.add_paragraph("")
    docx.add_paragraph("Second DOCX paragraph.")
    docx.save(path)


def test_loader_loads_markdown_with_source_metadata(tmp_path):
    path = tmp_path / "request_models.md"
    path.write_text("# Request Models\nFastAPI validates bodies.", encoding="utf-8")

    docs = DocumentLoader().load_file(path)

    assert len(docs) == 1
    assert docs[0].page_content.startswith("# Request Models")
    assert docs[0].metadata["source_title"] == "request_models"
    assert docs[0].metadata["source_path"] == str(path.resolve())
    assert docs[0].metadata["source_id"]


def test_loader_converts_html_and_skips_unsupported_files(tmp_path):
    html_path = tmp_path / "intro.html"
    html_path.write_text(
        "<html><body><h1>FastAPI</h1><script>hide()</script><p>Visible text.</p></body></html>",
        encoding="utf-8",
    )
    unsupported = tmp_path / "notes.pdf"
    unsupported.write_text("not supported", encoding="utf-8")

    loader = DocumentLoader()

    html_docs = loader.load_file(html_path)
    unsupported_docs = loader.load_file(unsupported)

    assert html_docs[0].page_content == "FastAPI\nVisible text."
    assert unsupported_docs == []


def test_load_pdf_returns_document(tmp_path):
    path = tmp_path / "assignment.pdf"
    _write_sample_pdf(path)

    docs = DocumentLoader().load_file(path)

    assert len(docs) == 1
    assert "PDF page one content." in docs[0].page_content
    assert "PDF page two content." in docs[0].page_content


def test_load_docx_returns_document(tmp_path):
    path = tmp_path / "spec.docx"
    _write_sample_docx(path)

    docs = DocumentLoader().load_file(path)

    assert len(docs) == 1
    assert docs[0].page_content == "First DOCX paragraph.\nSecond DOCX paragraph."


def test_pdf_metadata_contains_page_count(tmp_path):
    path = tmp_path / "assignment.pdf"
    _write_sample_pdf(path)

    docs = DocumentLoader().load_file(path)

    assert docs[0].metadata["page_count"] == 2
    assert docs[0].metadata["file_type"] == "pdf"
    assert docs[0].metadata["filename"] == "assignment.pdf"
    assert docs[0].metadata["source_title"] == "assignment"
    assert docs[0].metadata["source_id"]


def test_docx_metadata_contains_file_type(tmp_path):
    path = tmp_path / "spec.docx"
    _write_sample_docx(path)

    docs = DocumentLoader().load_file(path)

    assert docs[0].metadata["file_type"] == "docx"
    assert docs[0].metadata["filename"] == "spec.docx"
    assert docs[0].metadata["source_title"] == "spec"
    assert docs[0].metadata["source_path"] == str(path.resolve())


def test_unsupported_file_still_skipped(tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"not a supported document")

    docs = DocumentLoader().load_file(path)

    assert docs == []


def test_chunking_preserves_metadata_and_adds_chunk_fields(sample_document):
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)

    chunks = chunker.split([sample_document])

    assert chunks
    assert chunks[0].metadata["source_id"] == "request-models"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["section_heading"] == "Request Bodies"
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_chunking_skips_empty_documents():
    doc = Document(page_content="   ", metadata={"source_path": "empty.md"})

    chunks = DocumentChunker().split([doc])

    assert chunks == []


def test_pipeline_loads_and_chunks_directory(tmp_path):
    (tmp_path / "a.md").write_text("# A\nFastAPI validates data.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Plain text documentation.", encoding="utf-8")
    (tmp_path / "ignore.bin").write_text("ignored", encoding="utf-8")

    chunks = IngestionPipeline().ingest(tmp_path)

    assert chunks
    assert {chunk.metadata["source_title"] for chunk in chunks} == {"a", "b"}
    assert all("source_id" in chunk.metadata for chunk in chunks)
