"""
vectorstore/in_memory_store.py
---------------------------------
Zero-dependency `VectorStore` implementation: brute-force cosine similarity
over records held in a Python dict. Useful for unit tests, small documents,
or environments where installing Chroma isn't wanted. Same interface as
`ChromaVectorStore`, so retrieval code never has to know which one is active.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from manual_rag.core.interfaces import VectorMatch, VectorRecord, VectorStore
from manual_rag.utils.text import cosine_similarity


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._records: Dict[str, VectorRecord] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for r in records:
            self._records[r.id] = r

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[VectorMatch]:
        candidates = self._records.values()
        if where:
            candidates = [
                r for r in candidates
                if all(r.metadata.get(k) == v for k, v in where.items())
            ]
        scored = [
            (r, cosine_similarity(query_embedding, r.embedding or []))
            for r in candidates
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            VectorMatch(id=r.id, score=score, text=r.text, metadata=r.metadata)
            for r, score in scored[:top_k]
        ]

    def delete_collection(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)
