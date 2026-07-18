"""
Feedback persistence boundary.

Purpose:
    Persist thumbs-up/down feedback and optional comments for generated
    answers, keeping file I/O out of the FastAPI route layer.

Responsibilities:
    - Store feedback keyed by query_id.
    - Support reading the most recent feedback for a query_id.
    - Keep persistence details out of /feedback route logic.

Public interfaces:
    - FeedbackStore
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackStore:
    """JSONL-backed persistence boundary for answer feedback."""

    DEFAULT_PATH = Path("app/storage/feedback.jsonl")

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        """Create a feedback store backed by a JSON Lines file."""
        self.path = Path(path)

    def save_feedback(
        self,
        *,
        query_id: str,
        rating: int,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """
        Persist one feedback submission.

        Duplicate submissions are appended with a new timestamp. Consumers
        that need the latest value can use ``get_feedback()``, which returns
        the newest record for the query_id.
        """
        if not query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if rating < 1 or rating > 5:
            raise ValueError("rating must be between 1 and 5")

        record: dict[str, Any] = {
            "query_id": query_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as feedback_file:
            feedback_file.write(json.dumps(record, ensure_ascii=True) + "\n")

        logger.info("Saved feedback for query_id=%s rating=%d", query_id, rating)
        return record

    def get_feedback(self, query_id: str) -> dict[str, Any] | None:
        """Read the most recent stored feedback record for a query_id."""
        if not self.path.exists():
            return None

        latest: dict[str, Any] | None = None
        with self.path.open("r", encoding="utf-8") as feedback_file:
            for line in feedback_file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed feedback record")
                    continue
                if isinstance(record, dict) and record.get("query_id") == query_id:
                    latest = record

        return latest
