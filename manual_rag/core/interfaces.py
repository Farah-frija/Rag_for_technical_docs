"""
core/interfaces.py
-------------------
The abstract contracts every pluggable component implements. This is the
single most important file for "pluggability": indexing and retrieval code
depend ONLY on these interfaces, never on a concrete provider (Gemini,
Ollama, Chroma, sentence-transformers, ...). Swapping a provider means
writing one new adapter class + one line in the matching `factory.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ── LLM ──────────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """A structured-output-capable chat LLM. Every concrete provider (Gemini,
    Ollama, ...) must be able to take a prompt + a JSON schema and return a
    parsed dict matching that schema -- this is the only shape the rest of
    the codebase (prompts + retrieval + indexing) relies on."""

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        label: str = "unlabeled",
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        trace: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Call the LLM and return a dict matching `schema`. Implementations
        are responsible for their own rate limiting, retries, and for
        logging the call (prompt/output/tokens/errors) via
        `manual_rag.logging_config.get_llm_call_logger()`. If `trace` (a
        `manual_rag.utils.tracing.TraceRecorder`) is provided, the call
        should also be appended there so callers get one place to inspect
        every LLM call made during a query/indexing run."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError


# ── Embeddings ───────────────────────────────────────────────────────────

class EmbeddingProvider(ABC):
    """Turns text into vectors. Implementations should cache/lazy-load their
    underlying model as a singleton so repeated calls are cheap."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns one vector per input text, in order."""
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Number of tokens the model's tokenizer produces for `text` (no special tokens)."""

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Max input length the model accepts (special tokens included)."""


# ── Vector store ─────────────────────────────────────────────────────────

@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorMatch:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Pluggable vector database. `retrieval/vanilla_retriever.py` and the
    indexing pipeline talk ONLY to this interface, so switching between
    Chroma, an in-memory index, or (later) Qdrant/Pinecone/pgvector is a
    config change (`MANUAL_RAG_VECTORSTORE_PROVIDER=...`), not a code change."""

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[VectorMatch]:
        raise NotImplementedError

    @abstractmethod
    def delete_collection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


# ── Reranker ─────────────────────────────────────────────────────────────

class Reranker(ABC):
    """Cross-encoder-style (query, document) relevance scorer, used as a
    precision stage after a recall stage (embedding / BM25 / LLM routing)."""

    @abstractmethod
    def score(self, query: str, document: str) -> float:
        """Return a relevance score, higher = more relevant. Implementations
        should normalise to roughly [0, 1] (e.g. via sigmoid) so thresholds
        configured for one reranker roughly transfer to another."""
        raise NotImplementedError

    def score_many(self, query: str, documents: Sequence[str]) -> List[float]:
        return [self.score(query, doc) for doc in documents]


# ── Retrieval ────────────────────────────────────────────────────────────

@dataclass
class RetrievedItem:
    """A single retrieved content unit (a leaf TEXT/TABLE/IMAGE node, or a
    flat chunk), normalised across retrieval strategies so downstream
    synthesis/eval code doesn't need to know which retriever produced it."""

    id: str
    content: str
    node_type: str
    page: Optional[int] = None
    section_id: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubQueryResult:
    subquery: str
    partial_answer: str
    confidence: str
    items: List[RetrievedItem] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Normalised output of any Retriever implementation."""

    query: str
    answer: str
    confidence: str
    subquery_results: List[SubQueryResult] = field(default_factory=list)
    stage_capture: Dict[str, Any] = field(default_factory=dict)
    trace_entries: List[Dict[str, Any]] = field(default_factory=list)


class Retriever(ABC):
    """Common contract for every retrieval strategy (tree-aware,
    vanilla dense+BM25, page-index hierarchical traversal, and any future
    strategy). `retrieval/factory.py` selects an implementation by name so
    the CLI/eval harness can do `--retriever tree|vanilla|page_index`
    without any other code caring which one is running."""

    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult:
        raise NotImplementedError
