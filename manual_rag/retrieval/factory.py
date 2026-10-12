from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from manual_rag.config import Settings
from manual_rag.core.interfaces import EmbeddingProvider, LLMClient, Reranker, Retriever, VectorStore
from manual_rag.core.models import ManualIndex
from manual_rag.indexing.bm25 import BM25Index

STRATEGIES = {"tree", "vanilla", "layout"}


@dataclass
class RetrieverDeps:
    # Optional because the layout retriever can skip LLM / embedder / reranker
    # depending on the step toggles in settings.retrieval.
    llm: Optional[LLMClient] = None
    embedder: Optional[EmbeddingProvider] = None
    reranker: Optional[Reranker] = None
    # chunk-based strategies only (vanilla, layout)
    vector_store: Optional[VectorStore] = None
    corpus: Optional[List[Dict[str, Any]]] = None
    # layout only; vanilla is dense-only and ignores it
    bm25_index: Optional[BM25Index] = None


def get_retriever(
    index: ManualIndex,
    settings: Settings,
    deps: RetrieverDeps,
    strategy: Optional[str] = None,
) -> Retriever:
    strategy = strategy or settings.retrieval.default_strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy {strategy!r}; choose from {STRATEGIES}")

    if strategy == "tree":
        from manual_rag.retrieval.tree_retriever import TreeRetriever
        return TreeRetriever(index, deps.llm, deps.embedder, deps.reranker, settings.retrieval)

    if deps.corpus is None:
        raise ValueError(f"Strategy {strategy!r} needs corpus in deps")
    id_to_chunk = {c["id"]: c for c in deps.corpus}

    if strategy == "vanilla":
        if deps.vector_store is None:
            raise ValueError("Strategy 'vanilla' needs vector_store in deps")
        from manual_rag.retrieval.vanilla_retriever import VanillaRetriever
        return VanillaRetriever(
            index, deps.llm, deps.embedder, deps.reranker, deps.vector_store,
            settings.retrieval, deps.corpus, id_to_chunk,
        )

    # layout: which deps are required depends on the step toggles
    r = settings.retrieval
    if r.use_dense and deps.vector_store is None:
        raise ValueError("use_dense=True needs vector_store in deps")
    from manual_rag.retrieval.layout_retriever import LayoutRetriever
    return LayoutRetriever(
        index, deps.llm, deps.embedder, deps.reranker, deps.vector_store,
        settings.retrieval, deps.corpus, deps.bm25_index, id_to_chunk,
    )
