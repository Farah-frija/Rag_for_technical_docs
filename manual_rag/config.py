"""
config.py
---------
Single source of truth for every tunable/pluggable knob in the system.

Everything is env-var driven (via pydantic-settings) so the same code runs
in dev / CI / prod by swapping an `.env` file or environment variables --
no code edits. Copy `.env.example` to `.env` and adjust as needed.

This is the ONLY place that should read `os.environ` directly for
application config. Every other module receives a `Settings` instance
(or one of its nested sub-configs) rather than reading env vars itself --
that's what makes the LLM / embedder / vector-store / reranker choices
truly pluggable: `manual_rag.llm.factory.get_llm_client(settings.llm)`
doesn't care whether `settings.llm.provider` came from an env var, a CLI
flag, or a unit test fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── LLM provider config ───────────────────────────────────────────────────

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANUAL_RAG_LLM_", extra="ignore")

    provider: Literal["gemini", "ollama"] = "gemini"

    # Gemini
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-3.1-flash-lite"

    # Ollama (local)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Shared behaviour
    min_call_interval_sec: float = 4.1     # simple client-side rate limit
    max_retries: int = 20
    request_timeout_sec: float = 120.0


# ── Embedding provider config ─────────────────────────────────────────────

class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANUAL_RAG_EMBED_", extra="ignore")

    provider: Literal["sentence_transformers"] = "sentence_transformers"
    model_name: str = "all-mpnet-base-v2"
    device: Optional[str] = None  # None -> auto-detect (cuda if available, else cpu)


# ── Reranker config ────────────────────────────────────────────────────────

class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANUAL_RAG_RERANKER_", extra="ignore")
    chunk_overlap_tokens: int = 64
    provider: Literal["cross_encoder", "none"] = "cross_encoder"
    model_name: str = "BAAI/bge-reranker-v2-m3"


# ── Vector store config ────────────────────────────────────────────────────

class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANUAL_RAG_VECTORSTORE_", extra="ignore")
    provider: Literal["chroma", "in_memory"] = "chroma"
    persist_dir: str = "./data/chroma"
    collection_name: str = "manual_rag_chunks"


# ── Retrieval tuning ───────────────────────────────────────────────────────

class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANUAL_RAG_RETRIEVAL_", extra="ignore")

    default_strategy: Literal["tree", "vanilla", "layout"] = "tree"

    # ── Step toggles (layout retriever) ───────────────────────────────────
    # Pipeline: [split] -> [dense] + [bm25] -> RRF -> [rerank] -> [LLM answer]
    # Each flag switches one step on/off. Disabling the two LLM steps makes the
    # pipeline stop at the step before them.
    use_subquery_split: bool = False   # LLM splits the query into sub-queries; off -> 1 sub-query
    use_dense: bool = True            # dense vector recall
    use_bm25: bool = True             # BM25 recall (also gates loading the BM25 index)
    use_reranker: bool = True     # cross-encoder rerank; off -> keep RRF order
    use_llm_responder: bool = True # LLM answer; off -> return retrieved chunks, no generation

    # tree retriever
    embed_threshold: float = 0.15
    max_candidates_per_subquery: int = 5
    top_nodes_for_context: int = 3
    max_context_items: int = 20
    embed_signal_weight_entity: float = 1.0
    embed_signal_weight_summary: float = 0.0
    embed_signal_weight_title: float = 0.0

    # vanilla retriever
    top_k_fused: int = 20
    # layout retriever: per-retriever depth -> union -> rerank -> cap
    top_n_dense: int = 50      # candidates taken from dense
    top_n_bm25: int = 50       # candidates taken from BM25
    top_n_fusion: int = 20     # kept after rerank (measured as stage 3)
    # shared: how many of the final chunks the LLM actually sees
    top_n_for_context: int = 5
    rrf_k: int = 60


# ── Indexing tuning ────────────────────────────────────────────────────────

class IndexingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANUAL_RAG_INDEXING_", extra="ignore")

    image_output_dir: str = "./data/extracted_images"
    min_image_size_px: int = 500
    chunk_token_size: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None
    chunk_overlap_ratio : float =0.1
    # chunk_token_size: Optional[int] = None        # None -> embedder dimension
    # chunk_overlap_tokens: Optional[int] = None    # None -> chunk_overlap_ratio * chunk size
    toc_candidate_page_cap: int = 10
    chunk_with_layout_context: bool = False
    chunk_with_layout: bool = False
    # Indexing-time toggles. IndexingPipeline must read these (see notes).
    build_bm25: bool = True           # write the BM25 corpus during indexing
    # What happens the moment an index finishes building/enriching.
    auto_retrieve_on_complete: bool = True
    smoke_test_queries: list[str] = Field(
        default_factory=lambda: ["What is this document about?"]
    )


# ── Observability ──────────────────────────────────────────────────────────

class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MANUAL_RAG_LOG_", extra="ignore")

    level: str = "INFO"
    log_dir: str = "./logs"
    json_logs: bool = True          # structured logs for machine consumption
    console_logs: bool = True       # human-readable stream handler
    log_llm_calls: bool = True      # full prompt/response dump per LLM call


# ── Top-level settings ─────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Aggregate settings object. Instantiate once (`get_settings()`) and pass
    the relevant sub-config into each factory -- never read env vars deeper
    in the stack."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: str = "./data"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.indexing.image_output_dir,
            self.vectorstore.persist_dir,
            self.logging.log_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Process-wide settings singleton. `reload=True` forces re-reading env/.env
    (useful in tests that monkeypatch environment variables)."""
    global _settings
    if _settings is None or reload:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
