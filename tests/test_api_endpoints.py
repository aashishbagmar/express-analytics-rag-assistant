from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api import documents as documents_api
from app.api import feedback as feedback_api
from app.api import ingest as ingest_api
from app.api import query as query_api
from app.api import upload as upload_api
from app.main import app


class FakeGraph:
    def invoke(self, state, config=None):
        return {
            "query_id": state["query_id"],
            "answer": "FastAPI validates request bodies with Pydantic.",
            "citations": [
                {
                    "source_id": "request",
                    "source_title": "Request Models",
                    "chunk_index": 0,
                }
            ],
            "confidence": "high",
            "retry_count": 0,
            "fallback_used": False,
        }


class FakePipeline:
    def ingest(self, path: str):
        return [
            Document(
                page_content="FastAPI validates data.",
                metadata={"source_id": "request", "chunk_index": 0},
            )
        ]

    def ingest_file(self, path: str):
        return self.ingest(path)


class FakeVectorStore:
    def build_index(self, docs):
        return len(docs)

    def add_documents(self, docs):
        return len(docs)

    def replace_documents_for_sources(self, docs):
        return len(docs)

    def list_documents(self):
        return [
            {
                "source_id": "request",
                "source_title": "Request Models",
                "filename": "request_models.md",
                "file_type": "md",
                "pages": None,
                "chunk_count": 3,
            }
        ]


class FakeFeedbackStore:
    def __init__(self) -> None:
        self.records = []

    def save_feedback(self, *, query_id: str, rating: int, comment: str | None = None):
        self.records.append({"query_id": query_id, "rating": rating, "comment": comment})


def make_client():
    app.dependency_overrides.clear()
    return TestClient(app)


def test_required_routes_are_registered():
    client = make_client()

    paths = set(client.get("/openapi.json").json()["paths"])

    assert {"/query", "/ingest", "/upload", "/documents", "/feedback", "/health"} <= paths


def test_health_endpoint():
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_query_endpoint(monkeypatch):
    client = make_client()
    app.dependency_overrides[query_api.get_compiled_graph] = lambda: FakeGraph()

    response = client.post("/query", json={"question": "How does FastAPI validate request bodies?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["confidence"] == "high"
    assert body["retry_count"] == 0
    assert body["fallback_used"] is False
    assert body["citations"][0]["source_id"] == "request"


def test_query_validation_error():
    client = make_client()

    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_ingest_endpoint():
    client = make_client()
    app.dependency_overrides[ingest_api.get_ingestion_pipeline] = lambda: FakePipeline()
    app.dependency_overrides[ingest_api.get_vector_store] = lambda: FakeVectorStore()

    response = client.post("/ingest", json={"path": "data/corpus"})

    assert response.status_code == 200
    assert response.json() == {
        "documents_processed": 1,
        "chunks_created": 1,
        "status": "success",
    }


def test_documents_endpoint():
    client = make_client()
    app.dependency_overrides[documents_api.get_vector_store] = lambda: FakeVectorStore()

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "source_id": "request",
            "source_title": "Request Models",
            "filename": "request_models.md",
            "file_type": "md",
            "pages": None,
            "chunks": 3,
        }
    ]


def test_feedback_endpoint():
    client = make_client()
    store = FakeFeedbackStore()
    app.dependency_overrides[feedback_api.get_feedback_store] = lambda: store

    response = client.post(
        "/feedback",
        json={"query_id": "q1", "rating": 5, "comment": "Helpful answer"},
    )

    assert response.status_code == 201
    assert response.json() == {"query_id": "q1", "status": "success"}
    assert store.records == [{"query_id": "q1", "rating": 5, "comment": "Helpful answer"}]


def test_feedback_rating_validation():
    client = make_client()

    response = client.post("/feedback", json={"query_id": "q1", "rating": 6})

    assert response.status_code == 422


def test_upload_endpoint_ingests_supported_file(tmp_path, monkeypatch):
    client = make_client()
    app.dependency_overrides[upload_api.get_ingestion_pipeline] = lambda: FakePipeline()
    app.dependency_overrides[upload_api.get_vector_store] = lambda: FakeVectorStore()
    monkeypatch.setattr(upload_api, "UPLOAD_DIRECTORY", tmp_path / "uploads")

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"FastAPI validates data.", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["filename"] == "notes.txt"
    assert body["chunks_created"] == 1
    assert body["processing_time_seconds"] is not None
    assert (tmp_path / "uploads" / "notes.txt").exists()


def test_upload_endpoint_rejects_duplicate_file(tmp_path, monkeypatch):
    client = make_client()
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True)
    (upload_dir / "notes.txt").write_text("existing", encoding="utf-8")
    monkeypatch.setattr(upload_api, "UPLOAD_DIRECTORY", upload_dir)

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"new content", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "already_exists"


def test_upload_endpoint_rejects_oversized_file(tmp_path, monkeypatch):
    client = make_client()
    monkeypatch.setattr(upload_api, "UPLOAD_DIRECTORY", tmp_path / "uploads")
    monkeypatch.setattr(upload_api, "MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/upload",
        files={"file": ("large.txt", b"x" * 20, "text/plain")},
    )

    assert response.status_code == 413


def test_upload_endpoint_rejects_unsupported_extension(tmp_path, monkeypatch):
    client = make_client()
    monkeypatch.setattr(upload_api, "UPLOAD_DIRECTORY", tmp_path / "uploads")

    response = client.post(
        "/upload",
        files={"file": ("archive.zip", b"zip", "application/zip")},
    )

    assert response.status_code == 400
