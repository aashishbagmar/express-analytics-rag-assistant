from __future__ import annotations

from app.storage.feedback_store import FeedbackStore


def test_feedback_store_saves_and_reads_latest_feedback(tmp_path):
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path)

    first = store.save_feedback(query_id="q1", rating=3, comment="ok")
    second = store.save_feedback(query_id="q1", rating=5, comment="great")

    latest = store.get_feedback("q1")

    assert first["rating"] == 3
    assert second["rating"] == 5
    assert latest is not None
    assert latest["rating"] == 5
    assert latest["comment"] == "great"
    assert path.exists()


def test_feedback_store_returns_none_for_unknown_query(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    assert store.get_feedback("missing") is None
