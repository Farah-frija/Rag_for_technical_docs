"""vectorstore/base.py -- re-exports the interface + value types for
convenient imports: `from manual_rag.vectorstore.base import VectorStore, VectorRecord, VectorMatch`."""

from manual_rag.core.interfaces import VectorMatch, VectorRecord, VectorStore

__all__ = ["VectorStore", "VectorRecord", "VectorMatch"]
