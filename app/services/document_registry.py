"""
Document registry service skeleton.

Purpose:
    Track corpus documents independently from the vector store so /documents can
    list indexed content and ingestion status reliably.

Responsibilities:
    - Store source metadata, content hashes, chunk counts, status, and embedding model id.
    - Support ingestion deduplication by content hash.
    - Provide document summaries to the API layer.

Public interfaces:
    - DocumentRegistry
"""


class DocumentRegistry:
    """Metadata registry boundary for ingested documents."""

    def register_document(self):
        """Register or update a document ingestion record."""

    def list_documents(self):
        """List indexed documents and ingestion statuses."""

    def find_by_content_hash(self):
        """Find an existing document by content hash."""
