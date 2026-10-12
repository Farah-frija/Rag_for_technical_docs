from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

from manual_rag.config import Settings, get_settings
from manual_rag.core.interfaces import Retriever
from manual_rag.core.models import ManualIndex
from manual_rag.embeddings.factory import get_embedding_provider
from manual_rag.indexing.bm25 import BM25Index, bm25_corpus_path, load_bm25_corpus
from manual_rag.indexing.pipeline import IndexingPipeline
from manual_rag.llm.factory import get_llm_client
from manual_rag.logging_config import configure_logging, get_logger
from manual_rag.rerankers.factory import get_reranker
from manual_rag.retrieval.factory import STRATEGIES, RetrieverDeps, get_retriever
from manual_rag.utils.json_io import load_tree
from manual_rag.vectorstore.factory import get_vector_store

logger = get_logger("cli")

_SUPPORTS_SPLIT_SUBQUERIES = {"tree", "layout"}


def _print_result(result) -> None:
    print("\n" + "=" * 70)
    print(f"QUERY      : {result.query}")
    print(f"CONFIDENCE : {result.confidence}")
    print("=" * 70)
    print(result.answer)
    print("=" * 70)


def _configure_stores(
    settings: Settings,
    index_path: str,
    chunk_with_layout: bool = False,
    chunk_with_layout_context: bool = False,
) -> None:
    stem = Path(index_path).stem
    if chunk_with_layout_context:
        settings.vectorstore.collection_name = f"layout_contextual"
    elif chunk_with_layout:
        settings.vectorstore.collection_name = f"layout"
    else:
        settings.vectorstore.collection_name = f"vanilla"
        
def _configure_chunk_sizes(settings: Settings) -> None:
    cfg = settings.indexing
    dim = get_embedding_provider(settings.embedding).max_tokens

    if cfg.chunk_token_size is None:
        cfg.chunk_token_size = dim
    else:
        cfg.chunk_token_size = min(cfg.chunk_token_size, dim)

    if cfg.chunk_overlap_tokens is None:
        cfg.chunk_overlap_tokens = int(cfg.chunk_token_size * cfg.chunk_overlap_ratio)

    logger.info(f"Chunk size: {cfg.chunk_token_size} tokens, overlap: {cfg.chunk_overlap_tokens} (embedder dim={dim})")

def _make_retriever(index: ManualIndex, settings: Settings, strategy: str) -> Retriever:
    """Build only the components the enabled steps need, then hand them to the factory.
    Must run after _configure_stores: the collection name decides which
    vector store and BM25 corpus are loaded."""
    r = settings.retrieval
    embedder = get_embedding_provider(settings.embedding)

    if strategy == "tree":
        deps = RetrieverDeps(
            get_llm_client(settings.llm), embedder, get_reranker(settings.reranker),
        )
        return get_retriever(index, settings, deps, strategy=strategy)

    if strategy == "layout":
        need_llm = r.use_subquery_split or r.use_llm_responder
        llm = get_llm_client(settings.llm) if need_llm else None
        emb = embedder if r.use_dense else None
        reranker = get_reranker(settings.reranker) if r.use_reranker else None
        vector_store = get_vector_store(settings.vectorstore) if r.use_dense else None
        corpus = load_bm25_corpus(bm25_corpus_path(settings.vectorstore.collection_name))
        bm25_index = BM25Index.build(corpus) if (r.use_bm25 and corpus) else None
        deps = RetrieverDeps(llm, emb, reranker, vector_store, corpus, bm25_index)
        return get_retriever(index, settings, deps, strategy=strategy)

    # vanilla
    deps = RetrieverDeps(
        get_llm_client(settings.llm), embedder, get_reranker(settings.reranker),
        get_vector_store(settings.vectorstore),
        load_bm25_corpus(bm25_corpus_path(settings.vectorstore.collection_name)),
    )
    return get_retriever(index, settings, deps, strategy=strategy)


def _run_query(
    retriever: Retriever,
    query: str,
    strategy: str,
    split_subqueries: bool = True,
) -> None:
    """Run one query on an already-built retriever."""
    if strategy in _SUPPORTS_SPLIT_SUBQUERIES:
        # layout: False forces split off, True defers to settings.retrieval.use_subquery_split
        if strategy == "layout" and split_subqueries:
            split_subqueries = None
        result = retriever.retrieve(query, split_subqueries=split_subqueries)
    else:
        result = retriever.retrieve(query)
    _print_result(result)


def _auto_retrieve_hook(settings: Settings, strategy: str, mode: str, split_subqueries: bool = True):
    def hook(index_path: str, index: ManualIndex) -> None:
        # Built once, after the index/corpus/vector store exist, then reused.
        retriever = _make_retriever(index, settings, strategy)
        if mode == "smoke_test":
            queries = settings.indexing.smoke_test_queries
            print(f"\n[auto-retrieve] Index ready -> running {len(queries)} smoke-test quer(y/ies) "
                  f"with retriever='{strategy}' (split_subqueries={split_subqueries}) ...")
            for q in queries:
                _run_query(retriever, q, strategy, split_subqueries=split_subqueries)
        elif mode == "interactive":
            print(f"\n[auto-retrieve] Index ready -> entering interactive query mode (retriever='{strategy}').")
            _interactive_loop(retriever, strategy, split_subqueries=split_subqueries)
        else:
            raise ValueError(f"Unknown auto-retrieve mode: {mode!r}")
    return hook


def _interactive_loop(retriever: Retriever, strategy: str, split_subqueries: bool = True) -> None:
    print(f"Interactive query mode (retriever='{strategy}', split_subqueries={split_subqueries}). "
          f"Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        try:
            _run_query(retriever, query, strategy, split_subqueries=split_subqueries)
        except Exception as exc:
            logger.error(f"query failed: {exc}")
            print(f"[error] {exc}")


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_index(args: argparse.Namespace, settings: Settings) -> None:
    _configure_stores(settings, args.output, args.chunk_with_layout, args.chunk_with_layout_context)
    _configure_chunk_sizes(settings)
    pipeline = IndexingPipeline(settings)
    on_complete = None
    if not args.no_auto_retrieve:
        on_complete = _auto_retrieve_hook(
            settings, args.retriever, args.auto_retrieve_mode,
            split_subqueries=not args.no_split_subqueries,
        )
    pipeline.run(
        pdf_path=args.pdf,
        output_path=args.output,
        force_mode=args.mode if args.mode != "auto" else None,
        skip_llm_summarization=args.skip_llm_summarization,
        on_complete=on_complete,
        chunk_with_layout=args.chunk_with_layout,
        chunk_with_layout_context=args.chunk_with_layout_context,
    )
    print(f"\n[index] Saved -> {args.output}")


def cmd_query(args: argparse.Namespace, settings: Settings) -> None:
    _configure_stores(settings, args.index_path, args.chunk_with_layout, args.chunk_with_layout_context)
    index = load_tree(args.index_path)
    retriever = _make_retriever(index, settings, args.retriever)
    _run_query(retriever, args.query, args.retriever, split_subqueries=not args.no_split_subqueries)


def cmd_chat(args: argparse.Namespace, settings: Settings) -> None:
    _configure_stores(settings, args.index_path, args.chunk_with_layout, args.chunk_with_layout_context)
    index = load_tree(args.index_path)
    retriever = _make_retriever(index, settings, args.retriever)  # once per session
    _interactive_loop(retriever, args.retriever, split_subqueries=not args.no_split_subqueries)


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> None:
    from manual_rag.eval.evaluator import run_evaluation
    # _configure_stores must run before _make_retriever: the collection name it
    # sets decides which vector store and BM25 corpus get loaded.
    _configure_stores(settings, args.index_path, args.chunk_with_layout, args.chunk_with_layout_context)
    index = load_tree(args.index_path)
    retriever = _make_retriever(index, settings, args.retriever)  # once for the whole run
    run_evaluation(
        settings, index, args.csv_path, args.output_prefix,
        retriever_name=args.retriever, limit=args.limit, start_at=args.start_at,
        split_subqueries=not args.no_split_subqueries,
        retriever=retriever,
    )


def cmd_view(args: argparse.Namespace, settings: Settings) -> None:
    print("Streamlit apps can't be launched from argparse directly. Run:\n")
    print("    streamlit run -m manual_rag.app.viewer\n")
    print(f"then upload '{args.index_path}' in the sidebar.")


# ── Parser ────────────────────────────────────────────────────────────────────

def _add_layout_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chunk-with-layout", action="store_true", default=False,
                   help="Use layout-batched embeddings (no path prefix).")
    p.add_argument("--chunk-with-layout-context", action="store_true", default=False,
                   help="Use layout-batched embeddings with 2-level section path prefix.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manual-rag", description="Production RAG stack for technical manuals.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build, enrich, and save an index from a PDF.")
    p_index.add_argument("pdf")
    p_index.add_argument("output")
    p_index.add_argument("--mode", choices=["auto", "toc", "no_toc"], default="auto")
    p_index.add_argument("--skip-llm-summarization", action="store_true")
    p_index.add_argument("--retriever", choices=STRATEGIES, default="tree")
    p_index.add_argument("--auto-retrieve-mode", choices=["smoke_test", "interactive"], default="smoke_test")
    p_index.add_argument("--no-auto-retrieve", action="store_true")
    p_index.add_argument("--no-split-subqueries", action="store_true")
    _add_layout_args(p_index)
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="Run one query against a saved index.")
    p_query.add_argument("index_path")
    p_query.add_argument("query")
    p_query.add_argument("--retriever", choices=STRATEGIES, default="tree")
    p_query.add_argument("--no-split-subqueries", action="store_true")
    _add_layout_args(p_query)
    p_query.set_defaults(func=cmd_query)

    p_chat = sub.add_parser("chat", help="Interactive query loop against a saved index.")
    p_chat.add_argument("index_path")
    p_chat.add_argument("--retriever", choices=STRATEGIES, default="tree")
    p_chat.add_argument("--no-split-subqueries", action="store_true")
    _add_layout_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("evaluate", help="Run the retrieval evaluation harness.")
    p_eval.add_argument("index_path")
    p_eval.add_argument("csv_path")
    p_eval.add_argument("output_prefix")
    p_eval.add_argument("--retriever", choices=STRATEGIES, default="tree")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--start-at", type=int, default=0)
    p_eval.add_argument("--no-split-subqueries", action="store_true")
    p_eval.add_argument("--resume", action="store_true")
    _add_layout_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_view = sub.add_parser("view", help="Print the command to launch the Streamlit viewer.")
    p_view.add_argument("index_path")
    p_view.set_defaults(func=cmd_view)

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())