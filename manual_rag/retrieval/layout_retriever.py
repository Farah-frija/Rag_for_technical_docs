from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
from manual_rag.core.models import ManualIndex, NodeType
from manual_rag.indexing.bm25 import BM25Index
from manual_rag.logging_config import get_logger
from manual_rag.prompts.retrieval import (
    RESPONDER_PROMPT,
    RESPONDER_SCHEMA,
    SUBQUERY_SPLIT_PROMPT,
    SUBQUERY_SPLIT_SCHEMA,
)
from manual_rag.utils.shared_helpers import fmt_sections
from manual_rag.utils.tracing import TraceRecorder, timed

logger = get_logger(__name__)

_CONFIDENCE_RANK = {"not_found": 0, "low": 1, "medium": 2, "high": 3}
_NO_LLM_CONFIDENCE = "n/a"  # used when the LLM responder step is disabled


class LayoutRetriever(Retriever):
    """Step toggles come from `settings` (RetrievalSettings):

        use_subquery_split -> LLM query splitting
        use_dense          -> dense vector recall
        use_bm25           -> BM25 recall
        use_reranker       -> cross-encoder rerank (off: keep RRF order)
        use_llm_responder  -> LLM answer (off: stop at retrieved chunks)
    """
    name = "layout"

    def __init__(
        self,
        index: ManualIndex,
        llm: Optional[LLMClient],
        embedder: Optional[EmbeddingProvider],
        reranker: Optional[Reranker],
        vector_store: Optional[VectorStore],
        settings: RetrievalSettings,
        corpus: List[Dict[str, Any]],
        bm25_index: Optional[BM25Index],
        id_to_chunk: Dict[str, Dict[str, Any]],
    ) -> None:
        if not (settings.use_dense or settings.use_bm25):
            raise ValueError("At least one of use_dense / use_bm25 must be enabled.")
        if settings.use_dense and (embedder is None or vector_store is None):
            raise ValueError("use_dense=True requires an embedder and a vector_store.")
        if (settings.use_subquery_split or settings.use_llm_responder) and llm is None:
            raise ValueError("use_subquery_split / use_llm_responder require an LLM client.")

        self.index = index
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.vector_store = vector_store
        self.settings = settings
        self.corpus = corpus
        self.bm25_index = bm25_index
        self.id_to_chunk = id_to_chunk

        self.use_rerank = settings.use_reranker and reranker is not None
        if settings.use_reranker and reranker is None:
            logger.warning("use_reranker=True but no reranker provided; rerank step skipped.")

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_toc_string(self) -> str:
        lines: List[str] = []

        def _walk(node_id: str, depth: int) -> None:
            node = self.index.nodes.get(node_id)
            if not node or node.node_id == self.index.root_id:
                return
            if node.node_type == NodeType.SECTION:
                lines.append("  " * depth + node.content)
                for cid in node.children:
                    _walk(cid, depth + 1)

        for cid in self.index.nodes[self.index.root_id].children:
            _walk(cid, 0)
        return "\n".join(lines) or "(No sections)"

    def _split_query(self, query: str, toc: str, trace: TraceRecorder) -> List[str]:
        prompt = SUBQUERY_SPLIT_PROMPT.format(toc=toc, query=query)
        with timed("subquery_split_llm", logger):
            result = self.llm.generate_structured(
                prompt, SUBQUERY_SPLIT_SCHEMA,
                label="LAYOUT_SPLIT", trace=trace,
            )
        subqueries = [sq for sq in result.get("subqueries", []) if sq and sq.strip()]
        return subqueries or [query]

    @staticmethod
    def _format_chunks(items: List[RetrievedItem]) -> str:
        return "\n\n---\n\n".join(
            f"[chunk_id={it.id} | Page {it.page if it.page is not None else '?'}]\n{it.content.strip()}"
            for it in items
        )

    # ── per-subquery pipeline ─────────────────────────────────────────────

    def _run_subquery(
        self,
        subquery: str,
        trace: TraceRecorder,
    ) -> Tuple[SubQueryResult, List[str], List[str], List[str], List[str], List[str]]:
        """Returns (result, fused_ids, ce_ids, cited_ids_raw, dense_ranked, bm25_ranked).
        `cited_ids_raw` is exactly what the LLM returned in `cited_chunks`
        (empty when the responder is disabled)."""
        s = self.settings

        # 1) recall: fixed depth per retriever
        dense_ranked: List[str] = []
        if s.use_dense:
            with timed(f"query_embedding[{subquery[:30]}]", logger):
                query_vec = self.embedder.embed_one(subquery)
            with timed(f"dense_recall[{subquery[:30]}]", logger):
                dense_matches = self.vector_store.query(query_vec, top_k=s.top_n_dense)
                dense_ranked = [m.id for m in dense_matches]

        bm25_ranked: List[str] = []
        if s.use_bm25 and self.bm25_index is not None:
            with timed(f"bm25_recall[{subquery[:30]}]", logger):
                bm25_scores = self.bm25_index.score(subquery)
                bm25_ranked = [
                    cid for cid, _ in sorted(bm25_scores.items(), key=lambda kv: kv[1], reverse=True)
                ][: s.top_n_bm25]

        # 2) union of both lists (RRF only orders it; NO cut here)
        fused_raw = _rrf_fuse(dense_ranked, bm25_ranked, k=s.rrf_k)
        fused_ids = [cid for cid, _ in fused_raw]
        fused_score_map = {cid: score for cid, score in fused_raw}

        # 3) rerank the whole union, then cap to top_n_fusion
        top_chunks: List[Dict[str, Any]] = []
        if self.use_rerank:
            with timed(f"ce_rerank[{subquery[:30]}]", logger):
                ce_scored: List[Dict[str, Any]] = []
                for cid in fused_ids:
                    chunk = self.id_to_chunk.get(cid)
                    if not chunk:
                        continue
                    score = self.reranker.score(subquery, chunk["content"])
                    ce_scored.append({**chunk, "ce_score": score})
                ce_scored.sort(key=lambda x: x["ce_score"], reverse=True)
            top_chunks = ce_scored[: s.top_n_fusion]
        else:
            for cid in fused_ids:
                chunk = self.id_to_chunk.get(cid)
                if not chunk:
                    continue
                top_chunks.append({**chunk, "ce_score": fused_score_map.get(cid, 0.0)})
                if len(top_chunks) >= s.top_n_fusion:
                    break
        ce_ids = [c["id"] for c in top_chunks]             # stage 3 (measured)
        llm_chunks = top_chunks[: s.top_n_for_context]     # what the LLM sees

        if not top_chunks:
            return (
                SubQueryResult(subquery, "No relevant content found.", "not_found", []),
                fused_ids,
                ce_ids,
                [],
                dense_ranked,
                bm25_ranked,
            )

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

        # 4) LLM answer (optional) -- off: stop here with the retrieved chunks
        if not s.use_llm_responder:
            return (
                SubQueryResult(
                    subquery=subquery,
                    partial_answer="",
                    confidence=_NO_LLM_CONFIDENCE,
                    items=items,
                ),
                fused_ids,
                ce_ids,
                [],
                dense_ranked,
                bm25_ranked,
            )

        context_blocks = [
            f"[chunk_id={c['id']} | Page {c.get('page', '?')} | "
            f"Section: {fmt_sections(c)}]\n{c['content'].strip()}"
            for c in llm_chunks
        ]
        prompt = RESPONDER_PROMPT.format(
            query=subquery,
            context="\n\n---\n\n".join(context_blocks),
        )
        with timed(f"responder_llm[{subquery[:30]}]", logger):
            result = self.llm.generate_structured(
                prompt, RESPONDER_SCHEMA,
                label="LAYOUT_RESPONDER", trace=trace,
            )
        cited_raw = result.get("cited_chunks") or []

        return (
            SubQueryResult(
                subquery=subquery,
                partial_answer=result.get("answer", ""),
                confidence=result.get("confidence", "low"),
                items=items,
            ),
            fused_ids,
            ce_ids,
            cited_raw,
            dense_ranked,
            bm25_ranked,
        )

    # ── public entry point ────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        split_subqueries: Optional[bool] = None,
    ) -> RetrievalResult:
        """`split_subqueries=None` -> use settings.use_subquery_split.
        A caller can still force it off with False, but cannot force it on
        when the setting disabled it."""
        trace = TraceRecorder()
        s = self.settings
        logger.info(f"{len(self.corpus)} chunks in corpus")

        no_index = not self.corpus or (s.use_bm25 and not s.use_dense and self.bm25_index is None)
        if no_index:
            trace.set_stage("stage2_per_subquery", [[]])
            trace.set_stage("stage3_per_subquery", [[]])
            trace.set_stage("stage4_per_subquery", [[]])
            trace.set_stage("num_subqueries", 1)
            trace.set_stage("subqueries_text", [query])
            return RetrievalResult(
                query, "No content indexed.", "not_found",
                stage_capture=trace.stage_capture,
            )

        do_split = s.use_subquery_split if split_subqueries is None else (
            s.use_subquery_split and split_subqueries
        )
        if do_split:
            subquery_texts = self._split_query(query, self._build_toc_string(), trace)
        else:
            subquery_texts = [query]

        logger.info(f"Sub-queries ({len(subquery_texts)}): {subquery_texts}")
        trace.set_stage("num_subqueries", len(subquery_texts))
        trace.set_stage("subqueries_text", subquery_texts)

        stage2: List[List[str]] = []
        stage3: List[List[str]] = []
        stage4: List[List[str]] = []
        stage1_dense: List[List[str]] = []
        stage1_bm25: List[List[str]] = []
        subquery_results: List[SubQueryResult] = []

        for sq in subquery_texts:
            sq_result, fused_ids, ce_ids, cited_raw, d_ids, b_ids = self._run_subquery(sq, trace)
            stage1_dense.append(d_ids)
            stage1_bm25.append(b_ids)
            subquery_results.append(sq_result)
            stage2.append(fused_ids)
            stage3.append(ce_ids)
            stage4.append(cited_raw)

        trace.set_stage("stage1_dense_per_subquery", stage1_dense)
        trace.set_stage("stage1_bm25_per_subquery", stage1_bm25)
        trace.set_stage("stage2_per_subquery", stage2)
        trace.set_stage("stage3_per_subquery", stage3)
        trace.set_stage("stage4_per_subquery", stage4)

        if not s.use_llm_responder:
            # Retrieval-only: show the chunks instead of a generated answer.
            all_items = [it for r in subquery_results for it in r.items]
            return RetrievalResult(
                query=query,
                answer=self._format_chunks(all_items) or "No relevant content found.",
                confidence=_NO_LLM_CONFIDENCE,
                subquery_results=subquery_results,
                stage_capture=trace.stage_capture,
                trace_entries=trace.entries,
            )

        final_answer = " ".join(
            r.partial_answer for r in subquery_results if r.partial_answer
        ).strip()

        ranks = [_CONFIDENCE_RANK.get(r.confidence, 0) for r in subquery_results]
        worst = min(ranks, default=0)
        confidence = next(l for l, r in _CONFIDENCE_RANK.items() if r == worst)

        return RetrievalResult(
            query=query,
            answer=final_answer or "No relevant content found.",
            confidence=confidence,
            subquery_results=subquery_results,
            stage_capture=trace.stage_capture,
            trace_entries=trace.entries,
        )


def _rrf_fuse(*ranked_lists: List[str], k: int = 60) -> List[Tuple[str, float]]:
    fused: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
