"""
Configuration boundary for environment-driven settings.

Purpose:
    Define the central settings boundary for environment-driven configuration.

Responsibilities:
    - Represent LLM, embedding, vector-store, retry, and filesystem settings.
    - Keep runtime constants out of graph state (see app/graph/state.py header).
    - Provide a single, cached dependency point for app, graph, services, and
      scripts.

Public interfaces:
    - Settings
    - get_settings
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration model, populated from environment variables
    (and an optional .env file) with sane local-dev defaults.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------ #
    # LLM (Ollama) - used by graph/nodes/grading.py and, in later phases,
    # query rewriting, generation, and groundedness checking.
    # ------------------------------------------------------------------ #

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server. No API key required.",
    )
    ollama_model: str = Field(
        default="qwen2.5:14b",
        description="Ollama model name used for grading/generation calls.",
    )
    llm_request_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for a single LLM request before failing over.",
    )
    llm_grading_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for LLM-based grading. When False, or when the LLM "
            "call fails/times out, grading falls back to the deterministic "
            "heuristic grader so the graph never hard-fails on grading."
        ),
    )

    # ------------------------------------------------------------------ #
    # Embeddings / vector store
    # ------------------------------------------------------------------ #

    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    chroma_persist_dir: str = Field(default="./data/chroma")

    # ------------------------------------------------------------------ #
    # Retry / retrieval bounds (run-config, deliberately not in GraphState)
    # ------------------------------------------------------------------ #

    max_retries: int = Field(default=2)
    max_regen: int = Field(default=1)
    default_top_k: int = Field(default=5)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
