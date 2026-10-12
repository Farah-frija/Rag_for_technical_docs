"""vectorstore/factory.py -- selects a VectorStore by config. To add a new
backend (Qdrant, Pinecone, pgvector, Weaviate, ...): implement `VectorStore`
in this package and add one branch here plus settings fields."""

from __future__ import annotations

from manual_rag.config import VectorStoreSettings
from manual_rag.core.interfaces import VectorStore

_singleton: dict[str, VectorStore] = {}


def get_vector_store(settings: VectorStoreSettings) -> VectorStore:
    key = f"{settings.provider}:{settings.persist_dir}:{settings.collection_name}"
    if key in _singleton:
        return _singleton[key]

    if settings.provider == "chroma":
        from manual_rag.vectorstore.chroma_store import ChromaVectorStore
        store: VectorStore = ChromaVectorStore(settings)
    elif settings.provider == "in_memory":
        from manual_rag.vectorstore.in_memory_store import InMemoryVectorStore
        store = InMemoryVectorStore()
    else:
        raise ValueError(f"Unknown vector store provider: {settings.provider!r}")

    _singleton[key] = store
    return store
