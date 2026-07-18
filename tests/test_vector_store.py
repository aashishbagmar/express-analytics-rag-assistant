from __future__ import annotations

from langchain_core.documents import Document

from app.services.vector_store import VectorStoreService
from tests.conftest import FakeEmbeddingService


def test_vector_store_build_index_and_document_count(tmp_path):
    docs = [
        Document(
            page_content="FastAPI validates request bodies with Pydantic.",
            metadata={"source_id": "request", "source_title": "Request Models", "chunk_index": 0},
        ),
        Document(
            page_content="FastAPI dependency injection uses Depends.",
            metadata={"source_id": "deps", "source_title": "Dependencies", "chunk_index": 0},
        ),
    ]
    store = VectorStoreService(
        persist_directory=tmp_path / "chroma",
        collection_name="test_docs",
        embedding_service=FakeEmbeddingService(),
    )

    indexed = store.build_index(docs)

    assert indexed == 2
    assert store.document_count() == 2


def test_vector_store_similarity_search_returns_content_metadata_and_score(tmp_path):
    docs = [
        Document(
            page_content="FastAPI validates request bodies with Pydantic.",
            metadata={"source_id": "request", "source_title": "Request Models", "chunk_index": 0},
        ),
        Document(
            page_content="FastAPI dependency injection uses Depends.",
            metadata={"source_id": "deps", "source_title": "Dependencies", "chunk_index": 0},
        ),
    ]
    store = VectorStoreService(
        persist_directory=tmp_path / "chroma",
        collection_name="test_docs",
        embedding_service=FakeEmbeddingService(),
    )
    store.build_index(docs)

    results = store.similarity_search("How does FastAPI validate bodies?", k=1)

    assert len(results) == 1
    assert set(results[0]) == {"content", "metadata", "score"}
    assert results[0]["metadata"]["source_id"] in {"request", "deps"}
    assert isinstance(results[0]["score"], float)


def test_vector_store_list_documents_deduplicates_sources(tmp_path):
    docs = [
        Document(
            page_content="chunk one",
            metadata={"source_id": "same", "source_title": "Same Doc", "chunk_index": 0},
        ),
        Document(
            page_content="chunk two",
            metadata={"source_id": "same", "source_title": "Same Doc", "chunk_index": 1},
        ),
    ]
    store = VectorStoreService(
        persist_directory=tmp_path / "chroma",
        collection_name="test_docs",
        embedding_service=FakeEmbeddingService(),
    )
    store.build_index(docs)

    summaries = store.list_documents()

    assert summaries == [
        {
            "source_id": "same",
            "source_title": "Same Doc",
            "filename": None,
            "file_type": None,
            "pages": None,
            "source_path": None,
            "chunk_count": 2,
        }
    ]


def test_vector_store_replace_documents_for_sources_preserves_other_sources(tmp_path):
    store = VectorStoreService(
        persist_directory=tmp_path / "chroma",
        collection_name="test_docs",
        embedding_service=FakeEmbeddingService(),
    )
    upload_docs = [
        Document(
            page_content="Uploaded intern assignment requirements are listed here.",
            metadata={
                "source_id": "uploaded-assignment",
                "source_title": "uploaded_assignment",
                "chunk_index": 0,
            },
        )
    ]
    corpus_docs = [
        Document(
            page_content="FastAPI validates request bodies with Pydantic.",
            metadata={"source_id": "corpus-request", "source_title": "request_models", "chunk_index": 0},
        )
    ]

    store.add_documents(upload_docs)
    store.replace_documents_for_sources(corpus_docs)
    assert store.document_count() == 2

    updated_corpus_docs = [
        Document(
            page_content="FastAPI validates request bodies using Pydantic models and type hints.",
            metadata={"source_id": "corpus-request", "source_title": "request_models", "chunk_index": 0},
        )
    ]
    store.replace_documents_for_sources(updated_corpus_docs)

    assert store.document_count() == 2
    summaries = store.list_documents()
    assert {summary["source_id"] for summary in summaries} == {"uploaded-assignment", "corpus-request"}

    results = store.similarity_search("intern assignment requirements", k=1)
    assert results
    assert results[0]["metadata"]["source_id"] == "uploaded-assignment"
