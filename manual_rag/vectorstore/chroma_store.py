"""
vectorstore/chroma_store.py
------------------------------
Chroma-backed implementation of `VectorStore`. Chroma is embedded (no
separate server needed) and persists to `settings.persist_dir`, which keeps
the "just pluggable, chroma or whatever" requirement cheap to satisfy by
default while still being swappable.

We store embeddings we've already computed via `EmbeddingProvider`
(`upsert` always receives `record.embedding`), rather than letting Chroma
compute embeddings itself -- this keeps exactly ONE embedding code path in
the system (`manual_rag.embeddings`), used consistently whether the vector
store is Chroma, in-memory, or a future Qdrant/Pinecone adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from manual_rag.config import VectorStoreSettings
from manual_rag.core.exceptions import VectorStoreError
from manual_rag.core.interfaces import VectorMatch, VectorRecord, VectorStore
from manual_rag.logging_config import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(self, settings: VectorStoreSettings) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError(
                "chromadb is not installed. `pip install chromadb`, or set "
                "MANUAL_RAG_VECTORSTORE_PROVIDER=in_memory."
            ) from exc

        self._client = chromadb.PersistentClient(path=settings.persist_dir)
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        missing = [r.id for r in records if r.embedding is None]
        if missing:
            raise VectorStoreError(
                f"{len(missing)} record(s) missing an embedding; embed before upserting "
                f"(e.g. {missing[0]!r})."
            )
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.embedding for r in records],
            documents=[r.text for r in records],
            metadatas=[_sanitize_metadata(r.metadata) for r in records],
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[VectorMatch]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where,
        )
        matches: List[VectorMatch] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns a distance (lower = closer); convert to a
            # similarity-like score (higher = better) for a consistent
            # interface regardless of the store's native metric.
            score = 1.0 - dist if dist is not None else 0.0
            matches.append(VectorMatch(id=id_, score=score, text=doc, metadata=meta or {}))
        return matches

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)

    def count(self) -> int:
        return self._collection.count()


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool -- flatten anything
    else (e.g. lists of section titles) to a JSON string."""
    import json

    clean: Dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = json.dumps(v, ensure_ascii=False)
    return clean
