"""
retrieval/vanilla_retriever.py
-----------------------------------
Flat (non-hierarchical), dense-only retrieval baseline. Section structure is
ignored; every chunk in the corpus is treated as one flat pool of retrievable
chunks. No sparse/BM25 stage and no fusion: this is the naive approach.
Fair apples-to-apples baseline against `TreeRetriever` and `LayoutRetriever`.

Uses the same pre-built chunk corpus as `LayoutRetriever` (and as
`eval/evaluator.py`, which resolves chunk ids -> section names through that
same corpus via `source_sections`), so ids and section labels are guaranteed
to line up across retrieval and evaluation.

The corpus and the id -> chunk lookup are built once by
`retrieval/factory.py` and injected here; nothing corpus-related is rebuilt
per query.

Pipeline (per query):
  1. Dense   : query the vector store for the top-K nearest chunks.
  2. Rerank  : cross-encoder score the dense top-K against the raw query.
  3. Respond : concatenate the top-N post-rerank chunks with chunk id /
               page / section provenance and call the LLM for a single-pass
               answer. The LLM also returns the chunk ids it actually used.

`stage_capture` mirrors the per-subquery shape used by the other retrievers
even though vanilla never decomposes the query: `stage2_per_subquery`,
`stage3_per_subquery` and `stage4_per_subquery` are each a list containing
exactly ONE id list (the degenerate one-subquery case).
  - stage2: dense-recall chunk ids
  - stage3: post-cross-encoder chunk ids (what the LLM saw)
  - stage4: chunk ids as cited by the LLM, stored RAW (unvalidated); the
            evaluator validates them against stage3 and resolves them to
            sections exactly like the other stages.
"""

from __future__ import annotations

from typing import Any, Dict, List

from manual_rag.config import RetrievalSettings
from manual_rag.core.interfaces import (
    EmbeddingProvider,
    LLMClient,
    Reranker,
    RetrievalResult,
    Retriever,
    RetrievedItem,
    SubQueryResult,
    VectorStore,
)
from manual_rag.core.models import ManualIndex
from manual_rag.logging_config import get_logger
from manual_rag.prompts.vanilla_rag import RESPONDER_PROMPT, RESPONDER_SCHEMA
from manual_rag.utils.shared_helpers import fmt_sections
from manual_rag.utils.tracing import TraceRecorder, timed

logger = get_logger(__name__)


class VanillaRetriever(Retriever):
    name = "vanilla"

    def __init__(
        self,
        index: ManualIndex,
        llm: LLMClient,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        vector_store: VectorStore,
        settings: RetrievalSettings,
        corpus: List[Dict[str, Any]],
        id_to_chunk: Dict[str, Dict[str, Any]],
    ) -> None:
        self.index = index  # kept for factory/interface symmetry; not used for retrieval
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.vector_store = vector_store
        self.settings = settings
        self.corpus = corpus
        self.id_to_chunk = id_to_chunk

    def retrieve(self, query: str) -> RetrievalResult:
        trace = TraceRecorder()
        logger.info(f"{len(self.corpus)} chunks in corpus")

        # Vanilla never splits: always the degenerate one-subquery case.
        trace.set_stage("num_subqueries", 1)
        trace.set_stage("subqueries_text", [query])

        if not self.corpus:
            trace.set_stage("stage2_per_subquery", [[]])
            trace.set_stage("stage3_per_subquery", [[]])
            trace.set_stage("stage4_per_subquery", [[]])
            return RetrievalResult(
                query, "No content indexed.", "not_found",
                stage_capture=trace.stage_capture,
            )

        with timed("query_embedding", logger):
            query_vec = self.embedder.embed_one(query)

        # 1. dense recall, via the pluggable vector store
        with timed("dense_recall", logger):
            dense_matches = self.vector_store.query(query_vec, top_k=self.settings.top_k_fused)
            dense_ids = [m.id for m in dense_matches]
        trace.set_stage("stage2_per_subquery", [dense_ids])

        # 2. cross-encoder rerank
        with timed("cross_encoder_rerank", logger):
            ce_scored: List[Dict[str, Any]] = []
            for cid in dense_ids:
                chunk = self.id_to_chunk.get(cid)
                if not chunk:
                    continue
                score = self.reranker.score(query, chunk["content"])
                ce_scored.append({**chunk, "ce_score": score})
            ce_scored.sort(key=lambda x: x["ce_score"], reverse=True)

        top_chunks = ce_scored[: self.settings.top_n_for_context]
        ce_ids = [c["id"] for c in top_chunks]
        trace.set_stage("stage3_per_subquery", [ce_ids])

        if not top_chunks:
            trace.set_stage("stage4_per_subquery", [[]])
            return RetrievalResult(
                query, "No relevant content found.", "not_found",
                stage_capture=trace.stage_capture,
                trace_entries=trace.entries,
            )

        # 3. respond. ce_score is deliberately NOT shown to the LLM: it must
        #    cite by content, not by reranker score.
        context_blocks = [
            f"[chunk_id={c['id']} | Page {c.get('page', '?')} | "
            f"Section: {fmt_sections(c)}]\n{c['content'].strip()}"
            for c in top_chunks
        ]
        prompt = RESPONDER_PROMPT.format(query=query, context="\n\n---\n\n".join(context_blocks))
        with timed("responder_llm", logger):
            result = self.llm.generate_structured(
                prompt, RESPONDER_SCHEMA, label="VANILLA_RESPONDER", trace=trace,
            )

        # Raw LLM citations; validated + resolved to sections by the evaluator.
        trace.set_stage("stage4_per_subquery", [result.get("cited_chunks") or []])

        items = [
            RetrievedItem(
                id=c["id"],
                content=c["content"],
                node_type="chunk",
                page=c.get("page"),
                score=c["ce_score"],
                metadata={"source_sections": c.get("source_sections", "")},
            )
            for c in top_chunks
        ]
        subquery_results = [
            SubQueryResult(
                subquery=query,
                partial_answer=result.get("answer", ""),
                confidence=result.get("confidence", "low"),
                items=items,
            )
        ]

        return RetrievalResult(
            query=query,
            answer=result.get("answer", "") or "No relevant content found.",
            confidence=result.get("confidence", "low"),
            subquery_results=subquery_results,
            stage_capture=trace.stage_capture,
            trace_entries=trace.entries,
        )