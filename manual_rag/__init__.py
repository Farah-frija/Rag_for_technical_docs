"""
manual_rag
==========

A production-oriented Retrieval-Augmented Generation stack for long
technical manuals (PDF -> Markdown -> hierarchical index -> retrieval -> answer).

Design principles (see README.md for the full write-up):

  * Every external dependency (LLM provider, embedding model, vector store,
    reranker) sits behind a small abstract interface in `manual_rag.core.interfaces`
    and is instantiated through a `factory.py` in its own package, selected by
    `manual_rag.config.Settings`. Swapping Gemini <-> Ollama, or an in-memory
    vector index <-> Chroma, is a one-line config/env-var change, never a
    code change in `indexing/` or `retrieval/`.
  * Prompts never live inline in pipeline code -- they live in `manual_rag.prompts`.
  * Anything duplicated across two or more modules (rate limiting, text
    normalisation/tokenisation, timed-block logging, JSON tree IO, cosine
    similarity) lives once in `manual_rag.utils` and is imported everywhere.
  * Retrieval strategies (tree-aware, vanilla dense+BM25, page-index /
    hierarchical-traversal) all implement `manual_rag.retrieval.base.Retriever`,
    so they're interchangeable from the CLI/eval harness via `--retriever`.
  * Indexing ends with an explicit, observable event: `IndexingPipeline` calls
    an `on_complete` hook once the enriched tree is saved. The CLI wires this
    hook to "launch retrieval" automatically (interactive query loop, or a
    fixed batch of smoke-test queries) unless `--no-auto-retrieve` is passed,
    so indexing and retrieval are never accidentally left disconnected --
    but retrieval can always be launched manually too (`manual_rag query ...`).
"""

__version__ = "2.0.0"
