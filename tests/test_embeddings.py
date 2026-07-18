from __future__ import annotations

import sys
import types

from app.services.embeddings import EmbeddingService


class FakeArray:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class FakeSentenceTransformer:
    load_count = 0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        FakeSentenceTransformer.load_count += 1

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(texts, list):
            return FakeArray([[float(len(text)), 1.0] for text in texts])
        return FakeArray([float(len(texts)), 1.0])


def install_fake_sentence_transformers(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    EmbeddingService._models.clear()
    FakeSentenceTransformer.load_count = 0


def test_embedding_model_loads_lazily_and_is_shared(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)

    service_a = EmbeddingService("fake-model")
    service_b = EmbeddingService("fake-model")

    assert FakeSentenceTransformer.load_count == 0
    assert service_a.embed_query("abc") == [3.0, 1.0]
    assert service_b.embed_query("abcd") == [4.0, 1.0]
    assert FakeSentenceTransformer.load_count == 1


def test_embed_documents_returns_one_vector_per_text(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)

    vectors = EmbeddingService("fake-model").embed_documents(["one", "three"])

    assert vectors == [[3.0, 1.0], [5.0, 1.0]]


def test_embed_documents_empty_input_does_not_load_model(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)

    vectors = EmbeddingService("fake-model").embed_documents([])

    assert vectors == []
    assert FakeSentenceTransformer.load_count == 0


def test_model_id_returns_configured_model_name(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)

    assert EmbeddingService("fake-model").model_id() == "fake-model"
