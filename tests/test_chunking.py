from __future__ import annotations

from app.ingestion.chunking import DocumentChunker


def test_chunking_defaults_match_assignment():
    chunker = DocumentChunker()

    assert chunker.chunk_size == 800
    assert chunker.chunk_overlap == 100


def test_chunking_preserves_metadata(sample_document):
    chunks = DocumentChunker(chunk_size=100, chunk_overlap=10).split([sample_document])

    assert chunks
    assert chunks[0].metadata["source_id"] == sample_document.metadata["source_id"]
    assert chunks[0].metadata["source_title"] == sample_document.metadata["source_title"]
