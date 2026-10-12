# manual-rag

A production-oriented Retrieval-Augmented-Generation stack for long technical
manuals: **PDF -> Markdown -> hierarchical index -> pluggable retrieval -> answer**.

This is a from-scratch reorganisation of an earlier prototype into a
maintainable, observable, pluggable architecture. Nothing about *what* the
system does changed (same PDF conversion, same TOC-detection state machine,
same three retrieval strategies) — what changed is *how the pieces are wired
together*.

## Why this layout

The prototype had three structural problems this rewrite fixes:

1. **Providers were hardcoded.** Gemini clients, `SentenceTransformer`
   instances, and rate limiters were instantiated as module-level globals
   inside `retriever.py`, `page_index_approach.py`, and `summarize_tree.py`
   independently. Swapping the LLM or the embedding model meant editing
   three files. Now every provider sits behind an interface in
   `manual_rag/core/interfaces.py`, built by a `factory.py` in its own
   package, selected by one config value.
2. **Prompts lived inline.** Every f-string prompt was defined next to the
   code that used it. Now every prompt lives in `manual_rag/prompts/`,
   grouped by pipeline stage, and nowhere else.
3. **No vector database.** The prototype cached embeddings inside the tree's
   JSON and did brute-force cosine similarity in a Python loop. There was no
   abstraction to swap in a real vector DB. `manual_rag/vectorstore/` adds
   that abstraction (Chroma by default, or a zero-dependency in-memory
   store), used by the flat/vanilla retriever and the indexing pipeline's
   sync step.

## Architecture

```
manual_rag/
├── config.py            # Settings (pydantic-settings, env-driven) -- the
│                         # ONLY place that reads os.environ for app config
├── logging_config.py     # One place that configures logging (JSON logs,
│                         # console, a dedicated llm_calls.log)
├── core/
│   ├── models.py          # TreeNode / ManualIndex / NodeType / KnowledgeGraph
│   ├── interfaces.py      # LLMClient, EmbeddingProvider, VectorStore,
│   │                       # Reranker, Retriever -- the pluggability contracts
│   └── exceptions.py
├── utils/                 # Shared, reusable, stateless logic
│   ├── rate_limiter.py     # One RateLimiter, used by every LLM client
│   ├── text.py             # tokenize / normalise_title / cosine_similarity
│   ├── tracing.py          # timed() context manager + TraceRecorder
│   └── json_io.py          # atomic save_tree/load_tree
├── llm/                   # Pluggable LLM provider
│   ├── base.py              # BaseLLMClient: rate limit + retry + logging
│   ├── gemini_client.py
│   ├── ollama_client.py
│   └── factory.py            # get_llm_client(settings) -> LLMClient
├── embeddings/            # Pluggable embedding provider
│   ├── sentence_transformers_provider.py
│   └── factory.py
├── rerankers/             # Pluggable cross-encoder reranker
│   ├── cross_encoder.py
│   └── factory.py
├── vectorstore/           # Pluggable vector database
│   ├── chroma_store.py      # default: embedded, persistent
│   ├── in_memory_store.py   # zero-dependency fallback / tests
│   └── factory.py
├── prompts/               # Every prompt string + JSON schema, nothing else
│   ├── toc_extraction.py
│   ├── section_analysis.py
│   ├── retrieval.py
│   └── vanilla_rag.py
├── indexing/              # PDF -> ManualIndex pipeline
│   ├── pdf_to_markdown.py   # tables / images / text-around-obstacles
│   ├── toc_graph.py          # LangGraph page-probing state machine
│   ├── toc_extractor.py      # page scoring/ranking + orchestration
│   ├── tree_builder.py       # TOC-sectioned tree OR flat page-chunked corpus
│   ├── summarizer.py         # bottom-up section summaries + entity embeddings
│   ├── bm25.py
│   └── pipeline.py           # IndexingPipeline -- orchestrates all of the
│                              # above and fires on_complete() when done
├── retrieval/              # Three interchangeable retrieval strategies
│   ├── tree_retriever.py     # embed-prune -> LLM router -> CE rerank -> synth
│   ├── vanilla_retriever.py  # dense (vector store) + BM25 -> RRF -> CE -> LLM
│   ├── page_index_retriever.py  # pure LLM hierarchical traversal
│   └── factory.py            # get_retriever(index, settings, strategy)
├── eval/evaluator.py       # Stage-by-stage precision/recall/F1/hit-rate,
│                            # works against ANY Retriever via stage_capture
├── app/viewer.py           # Streamlit tree viewer (self-contained)
└── cli.py                  # `manual-rag index|query|chat|evaluate|view`
scripts/
└── watch_and_index.py     # optional: fully-automatic folder-watch indexing
tests/
└── test_pluggability.py   # fakes-based tests, no network/model downloads
```

### The pluggability contracts (`core/interfaces.py`)

| Interface         | What it abstracts                     | Built-in implementations                  |
|--------------------|----------------------------------------|--------------------------------------------|
| `LLMClient`         | structured-output chat completion      | Gemini, Ollama (local)                      |
| `EmbeddingProvider`  | text -> vector                         | sentence-transformers                       |
| `VectorStore`        | vector upsert/query                    | Chroma, in-memory                           |
| `Reranker`           | (query, doc) -> relevance score        | HF cross-encoder, no-op                     |
| `Retriever`          | query -> `RetrievalResult`             | tree, vanilla (dense+BM25), page_index      |

Every pipeline module (`indexing/*`, `retrieval/*`) depends **only** on these
interfaces. Swapping a provider is: write one adapter class implementing the
interface, add one branch to that package's `factory.py`, flip one env var.

### Indexing ends with an explicit event, not a convention

`IndexingPipeline.run(..., on_complete=hook)` calls `hook(index_path, index)`
the moment the enriched index is built, embedded, saved, and synced to the
vector store. The CLI wires this to auto-launch retrieval by default:

```bash
# builds the index AND immediately runs the configured smoke-test queries
# against it (or drops into an interactive loop with --auto-retrieve-mode interactive)
manual-rag index manual.pdf manual_index.json

# skip auto-retrieve, run it yourself later
manual-rag index manual.pdf manual_index.json --no-auto-retrieve
manual-rag query manual_index.json "What is the torque spec for bolt X?"
manual-rag chat manual_index.json                 # interactive loop
```

For a fully automatic pipeline (drop a PDF in a folder, get an indexed +
smoke-tested manual with zero commands), see `scripts/watch_and_index.py`,
which uses the exact same `on_complete` hook, just triggered by a filesystem
poll instead of a CLI invocation.

## Quickstart

```bash
python -m venv .venv 
venv\Scripts\activate   
pip install -r requirements.txt
cp .env.example .env        # fill in GEMINI_API_KEY, or set MANUAL_RAG_LLM_PROVIDER=ollama

manual-rag index docs/my_manual.pdf data/my_manual.json --no-vector-sync
manual-rag query data/my_manual.json "How do I reset the device?"
manual-rag evaluate data/my_manual.json eval_questions.csv results/run1 --retriever tree
streamlit run manual_rag/app/viewer.py    # upload data/my_manual.json in the sidebar
```

Switch providers with env vars, no code changes:

```bash
export MANUAL_RAG_LLM_PROVIDER=ollama
export MANUAL_RAG_LLM_OLLAMA_MODEL=llama3.1
export MANUAL_RAG_VECTORSTORE_PROVIDER=in_memory   # no Chroma dependency needed
manual-rag index docs/my_manual.pdf data/my_manual.json
```

## Observability

Everything logs through `manual_rag/logging_config.py` into `./logs/`:

- `logs/app.log` — every module's structured (JSON by default) log output.
- `logs/llm_calls.log` — every LLM prompt, output, token usage, and error,
  regardless of provider (this is what `BaseLLMClient` guarantees).

Every retrieval call also returns a `RetrievalResult.trace_entries` list (one
entry per LLM call made while answering that query) and a `stage_capture`
dict (which node/chunk ids survived each pipeline stage) — this is what
`manual_rag/eval/evaluator.py` reads to compute precision/recall per stage,
and what a future UI could render as a "why did it answer this way" trace.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

`tests/test_pluggability.py` proves the interfaces are genuinely
implementable/substitutable using in-memory fakes — no API keys, no model
downloads, no network access required to run it.

## Extending

- **New LLM provider**: implement `LLMClient` (or subclass `llm/base.BaseLLMClient`)
  in `manual_rag/llm/`, add a branch in `llm/factory.py`, add settings fields
  to `config.LLMSettings`.
- **New vector database**: implement `VectorStore` in `manual_rag/vectorstore/`,
  add a branch in `vectorstore/factory.py`.
- **New retrieval strategy**: implement `Retriever` in `manual_rag/retrieval/`,
  add a branch in `retrieval/factory.py`, add one entry to
  `eval/evaluator.STAGE_KEY_MAP` if you want it evaluable.
- **Shared logic**: if you find yourself copy-pasting a helper across two
  pipeline modules, it belongs in `manual_rag/utils/`, not in either module.
