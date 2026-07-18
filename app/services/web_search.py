"""
Web search service skeleton.

Purpose:
    Provide the optional Tavily/Serper/Exa fallback boundary when local corpus
    retrieval is insufficient.

Responsibilities:
    - Hide web-search provider details.
    - Normalize external search results into DocChunk-compatible structures.
    - Keep bonus fallback optional and isolated from the core RAG pipeline.

Public interfaces:
    - WebSearchService
"""


class WebSearchService:
    """Optional web-search fallback boundary."""

    def search(self):
        """Search the web for supplemental context."""
