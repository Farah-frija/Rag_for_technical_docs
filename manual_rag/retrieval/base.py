"""retrieval/base.py -- re-exports the interface + value types for
convenient imports."""

from manual_rag.core.interfaces import (
    RetrievalResult,
    Retriever,
    RetrievedItem,
    SubQueryResult,
)

__all__ = ["Retriever", "RetrievalResult", "RetrievedItem", "SubQueryResult"]
