from __future__ import annotations

import pytest

from app.graph.nodes.retrieval import RetrievalNodeError, retrieve_documents


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int = 5):
        self.calls.append((query, k))
        return [
            {
                "content": "FastAPI validates request bodies.",
                "metadata": {"source_id": "request", "chunk_index": 0},
                "score": 0.9,
            }
        ]


def test_retrieval_uses_rewritten_query_and_returns_docs():
    store = FakeVectorStore()

    update = retrieve_documents(
        {"rewritten_query": "How does FastAPI validate request bodies?"},
        vector_store=store,
        k=3,
    )

    assert store.calls == [("How does FastAPI validate request bodies?", 3)]
    assert update["retrieved_docs"][0]["content"] == "FastAPI validates request bodies."
    assert update["retrieved_docs"][0]["score"] == 0.9


def test_retrieval_rejects_invalid_result_shape():
    class BadStore:
        def similarity_search(self, query: str, k: int = 5):
            return [{"content": "x", "metadata": {}, "score": "bad"}]

    with pytest.raises(RetrievalNodeError):
        retrieve_documents({"rewritten_query": "query"}, vector_store=BadStore())


def test_retrieval_validates_k():
    with pytest.raises(ValueError):
        retrieve_documents({"rewritten_query": "query"}, vector_store=FakeVectorStore(), k=0)
