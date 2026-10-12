"""embeddings/base.py -- re-exports the interface for convenient imports:
`from manual_rag.embeddings.base import EmbeddingProvider`."""

from manual_rag.core.interfaces import EmbeddingProvider

__all__ = ["EmbeddingProvider"]
