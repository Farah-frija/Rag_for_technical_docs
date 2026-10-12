from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from manual_rag.config import Settings
from manual_rag.core.models import ManualIndex
from manual_rag.embeddings.factory import get_embedding_provider
from manual_rag.indexing.bm25 import bm25_corpus_path, save_bm25_corpus
from manual_rag.indexing.summarizer import embed_entity_cache, summarize_tree
from manual_rag.indexing.tree_builder import build_tree
from manual_rag.llm.factory import get_llm_client
from manual_rag.logging_config import get_logger
from manual_rag.utils.json_io import save_tree
from manual_rag.utils.tracing import TraceRecorder, timed
from manual_rag.vectorstore.factory import get_vector_store

logger = get_logger(__name__)

OnCompleteHook = Callable[[str, ManualIndex], None]


@dataclass
class IndexingRunResult:
    index_path: str
    index: ManualIndex
    trace: TraceRecorder


class IndexingPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = get_llm_client(settings.llm)
        self.embedder = get_embedding_provider(settings.embedding)

    def run(
        self,
        pdf_path: str,
        output_path: str,
        force_mode: Optional[str] = None,
        skip_llm_summarization: bool = False,
        chunk_with_layout: bool = False,
        chunk_with_layout_context: bool = False,
        on_complete: Optional[OnCompleteHook] = None,
    ) -> IndexingRunResult:
        trace = TraceRecorder()
        cfg = self.settings.indexing
        store = get_vector_store(self.settings.vectorstore)

        with timed("build_tree", logger):
            index, corpus = build_tree(
                pdf_path,
                llm=self.llm,
                embedder=self.embedder,
                image_output_dir=cfg.image_output_dir,
                min_image_size=cfg.min_image_size_px,
                chunk_token_size=cfg.chunk_token_size,
                chunk_overlap_tokens=cfg.chunk_overlap_tokens,
                toc_candidate_page_cap=cfg.toc_candidate_page_cap,
                force_mode=force_mode,
                chunk_with_layout=chunk_with_layout,
                chunk_with_layout_context=chunk_with_layout_context,
                plain_store=store if not (chunk_with_layout or chunk_with_layout_context) else None,
                contextual_store=store if (chunk_with_layout or chunk_with_layout_context) else None,
                trace=trace,
            )

        if not skip_llm_summarization:
            with timed("summarize_tree", logger):
                summarize_tree(index, self.llm, trace=trace)

        with timed("embed_entity_cache", logger):
            embed_entity_cache(index, self.embedder)

        with timed("save_tree", logger):
            save_tree(index, output_path)

        with timed("save_bm25_corpus", logger):
         save_bm25_corpus(records=corpus,path=bm25_corpus_path(
            self.settings.vectorstore.collection_name,
            
        )) 

        logger.info(
            "Indexing complete",
            extra={
                "pdf_path": pdf_path,
                "output_path": output_path,
                "node_count": len(index.nodes),
                "corpus_size": len(corpus),
                "llm_tokens_used": trace.total_tokens(),
                "chunk_with_layout": chunk_with_layout,
                "chunk_with_layout_context": chunk_with_layout_context,
            },
        )

        result = IndexingRunResult(index_path=output_path, index=index, trace=trace)

        if on_complete is not None:
            logger.info("Firing on_complete hook ...")
            on_complete(output_path, index)
        elif cfg.auto_retrieve_on_complete:
            logger.info(
                "auto_retrieve_on_complete is set but no on_complete hook was provided -- "
                "nothing to launch. Pass one (see manual_rag.cli) to auto-launch retrieval."
            )

        return result