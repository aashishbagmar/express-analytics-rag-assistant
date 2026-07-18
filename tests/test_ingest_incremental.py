from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import documents as documents_api
from app.api import ingest as ingest_api
from app.api import upload as upload_api
from app.ingestion.pipeline import IngestionPipeline
from app.main import app
from app.services.vector_store import VectorStoreService
from tests.conftest import FakeEmbeddingService


def test_corpus_reingest_preserves_uploaded_document(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    upload_dir = tmp_path / "uploads"
    corpus_dir.mkdir()
    upload_dir.mkdir()

    (corpus_dir / "request_models.md").write_text(
        "# Request Models\nFastAPI validates request bodies.",
        encoding="utf-8",
    )
    upload_content = b"The intern assignment requires building a LangGraph RAG assistant."

    chroma_dir = tmp_path / "chroma"
    store = VectorStoreService(
        persist_directory=chroma_dir,
        collection_name="integration_docs",
        embedding_service=FakeEmbeddingService(),
    )
    pipeline = IngestionPipeline()

    monkeypatch.setattr(upload_api, "UPLOAD_DIRECTORY", upload_dir)

    app.dependency_overrides.clear()
    app.dependency_overrides[ingest_api.get_ingestion_pipeline] = lambda: pipeline
    app.dependency_overrides[ingest_api.get_vector_store] = lambda: store
    app.dependency_overrides[upload_api.get_ingestion_pipeline] = lambda: pipeline
    app.dependency_overrides[upload_api.get_vector_store] = lambda: store
    app.dependency_overrides[documents_api.get_vector_store] = lambda: store

    client = TestClient(app)

    ingest_response = client.post("/ingest", json={"path": str(corpus_dir)})
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "success"

    upload_response = client.post(
        "/upload",
        files={"file": ("assignment.txt", upload_content, "text/plain")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["status"] == "success"

    documents_before = client.get("/documents").json()
    assert len(documents_before) == 2

    (corpus_dir / "request_models.md").write_text(
        "# Request Models\nFastAPI validates request bodies using Pydantic models.",
        encoding="utf-8",
    )
    reingest_response = client.post("/ingest", json={"path": str(corpus_dir)})
    assert reingest_response.status_code == 200

    documents_after = client.get("/documents").json()
    assert len(documents_after) == 2
    assert {doc["source_title"] for doc in documents_after} == {
        "request_models",
        "assignment",
    }

    upload_hits = store.similarity_search("intern assignment LangGraph", k=3)
    assert any(hit["metadata"]["source_title"] == "assignment" for hit in upload_hits)

    corpus_hits = store.similarity_search("FastAPI validates request bodies", k=3)
    assert any(hit["metadata"]["source_title"] == "request_models" for hit in corpus_hits)

    app.dependency_overrides.clear()
