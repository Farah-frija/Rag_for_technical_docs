"""
retrieval/tree_retriever.py
-------------------------------
The tree-aware, multi-stage retrieval pipeline. Ported from the original
`Answering/retriever.py`, restructured as a `Retriever` implementation with
every external dependency injected (LLM, embedder, reranker) rather than
imported as module globals -- so this class is fully testable with fakes
and fully swappable via `manual_rag.config`.

Pipeline (per query):
  1.  Extract entities from the query via the LLM (one call).
  2.  Embed the query entities -> entity query vector, AND embed the raw
      query text -> text query vector.
  3.  Score every content-bearing SECTION with three cosine similarities
      (entity/summary/title), fused into a single embed_score. The entity
      signal reflects how well the query's entities match a section's
      content -- it is a relevance signal, not a pruning gate, so when
      query-entity extraction yields nothing, fusion renormalizes across
      the remaining (summary/title) signals rather than zeroing a third
      of every section's score.
  4.  Prune sections below an adaptive embed_threshold -- survivors only
      go to the LLM. When no entities were extracted, or when pruning
      eliminates every section, the full set of content-bearing sections
      is used instead (no pruning applied).
  5.  Build a pruned TOC, send it to the LLM in ONE call. When
      `split_subqueries` is True (default), the LLM decomposes the query
      into sub-queries and, per sub-query, proposes candidate section ids
      with an llm_score. When False, the LLM is prompted to treat the
      query as indivisible and returns a single flat candidate list, which
      is wrapped into one "sub-query" whose text is the original query.
  6.  For each sub-query independently: dedup candidates (also against
      section ids already claimed by earlier sub-queries), cross-encoder
      rerank (leaves are batched into the reranker's context window;
      a leaf exceeding the full budget is chunked independently; section
      score = max across all batches/chunks), then pack the LLM context
      by appending leaf text in ce_score order until the char budget is
      exhausted -- the leaf that tips it over is truncated at the boundary.
      Images contribute their description text throughout.
      The LLM then produces a grounded partial answer with per-fragment
      section attribution.
  7.  The final answer is the concatenation of every sub-query's partial
      answer; there is no separate synthesis LLM call.

`stage_capture` in the returned `RetrievalResult` records, per stage:
  - `stage1_ids`            : ordered list of node_ids surviving embed pruning
  - `stage2_per_subquery`   : List[List[str]] — router-ordered candidate ids
                              per sub-query (content-bearing sections only)
  - `stage3_per_subquery`   : List[List[str]] — CE-ranked section ids per
                              sub-query, recorded BEFORE context packing
  - `num_subqueries`        : int
  - `subqueries_text`       : List[str]

All three stage id sets are normalized to the same definition of "section"
(a content-bearing section, i.e. one with direct leaves): stage1 and
stage3 naturally satisfy this; stage2 candidates from the router may be
structural/container sections, so they're expanded down to their
content-bearing descendants before being recorded.

The evaluator consumes `stage2_per_subquery` and `stage3_per_subquery`
to compute per-subquery Context Precision@K, then means over sub-queries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from manual_rag.config import RetrievalSettings
from manual_rag.core.interfaces import (
    EmbeddingProvider,
    LLMClient,
    Reranker,
    RetrievalResult,
    Retriever,
    RetrievedItem,
    SubQueryResult,
)
from manual_rag.core.models import ManualIndex, NodeType, TreeNode
from manual_rag.logging_config import get_logger
from manual_rag.prompts.retrieval import (
    QUERY_ENTITY_PROMPT,
    QUERY_ENTITY_SCHEMA,
    SUBQUERY_ANSWER_PROMPT,
    SUBQUERY_ANSWER_SCHEMA,
    TOC_ROUTER_PROMPT,
    TOC_ROUTER_SCHEMA,
    TOC_ROUTER_NOSPLIT_PROMPT,
    TOC_ROUTER_NOSPLIT_SCHEMA,
)
from manual_rag.utils.shared_helpers import has_direct_leaves, section_children
from manual_rag.utils.text import cosine_similarity
from manual_rag.utils.tracing import TraceRecorder, timed

logger = get_logger(__name__)

_LEAF_TYPES = (NodeType.TEXT, NodeType.TABLE, NodeType.IMAGE)

_CONFIDENCE_RANK = {"not_found": 0, "low": 1, "medium": 2, "high": 3}


class TreeRetriever(Retriever):
    name = "tree"

    def __init__(
        self,
        index: ManualIndex,
        llm: LLMClient,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        settings: RetrievalSettings,
    ) -> None:
        self.index = index
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self._weights = {
            "entity": settings.embed_signal_weight_entity,
            "summary": settings.embed_signal_weight_summary,
            "title": settings.embed_signal_weight_title,
        }

    # ── Public entrypoint ──────────────────────────────────────────────

    def retrieve(self, query: str, split_subqueries: bool = True) -> RetrievalResult:
        trace = TraceRecorder()

        subquery_results = self._retrieve_relevant_items(query, trace, split_subqueries=split_subqueries)

        final_answer = " ".join(r.partial_answer for r in subquery_results if r.partial_answer).strip()
        confidence = self._aggregate_confidence(subquery_results)

        return RetrievalResult(
            query=query,
            answer=final_answer or "No relevant content found.",
            confidence=confidence,
            subquery_results=subquery_results,
            stage_capture=trace.stage_capture,
            trace_entries=trace.entries,
        )

    @staticmethod
    def _aggregate_confidence(subquery_results: List[SubQueryResult]) -> str:
        if not subquery_results:
            return "not_found"
        ranks = [_CONFIDENCE_RANK.get(r.confidence, 0) for r in subquery_results]
        worst = min(ranks)
        for label, rank in _CONFIDENCE_RANK.items():
            if rank == worst:
                return label
        return "low"

    # ── Step 1-2: entities + embeddings ───────────────────────────────

    def _extract_query_entities(self, query: str, trace: TraceRecorder) -> List[str]:
        with timed("entity_extraction", logger):
            result = self.llm.generate_structured(
                QUERY_ENTITY_PROMPT.format(query=query), QUERY_ENTITY_SCHEMA,
                label="QUERY_ENTITIES", trace=trace,
            )
        return result.get("entities", [])

    def _ensure_node_text_embeddings(self, node: TreeNode) -> None:
        if "summary_embedding" not in node.metadata:
            summary = node.metadata.get("summary", "")
            text = summary.strip() if summary and summary.strip() else node.content
            node.metadata["summary_embedding"] = self.embedder.embed_one(text)
        if "title_embedding" not in node.metadata:
            node.metadata["title_embedding"] = self.embedder.embed_one(node.content)

    def _fused_embed_score(
        self, entity_score: Optional[float], summary_score: float, title_score: float,
    ) -> float:
        if entity_score is None:
            total = self._weights["summary"] + self._weights["title"]
            if total <= 0:
                return 0.0
            return (
                self._weights["summary"] / total * summary_score
                + self._weights["title"] / total * title_score
            )
        return (
            self._weights["entity"] * entity_score
            + self._weights["summary"] * summary_score
            + self._weights["title"] * title_score
        )

    def _score_all_sections(
        self, query_entity_vec: Optional[List[float]], query_text_vec: List[float]
    ) -> None:
        for nid, node in self.index.nodes.items():
            if node.node_type != NodeType.SECTION or nid == self.index.root_id:
                continue
            if not has_direct_leaves(self.index, node):
                continue
            self._ensure_node_text_embeddings(node)
            e_score = (
                cosine_similarity(query_entity_vec, node.metadata.get("entity_embedding") or [])
                if query_entity_vec else None
            )
            s_score = cosine_similarity(query_text_vec, node.metadata.get("summary_embedding") or [])
            t_score = cosine_similarity(query_text_vec, node.metadata.get("title_embedding") or [])
            node.metadata["entity_score"] = e_score if e_score is not None else 0.0
            node.metadata["summary_score"] = s_score
            node.metadata["title_score"] = t_score
            node.metadata["embed_score"] = self._fused_embed_score(e_score, s_score, t_score)

    # ── Step 3-4: prune ───────────────────────────────────────────────

    def _all_content_sections(self) -> List[TreeNode]:
        return [
            node for node in self.index.nodes.values()
            if node.node_type == NodeType.SECTION
            and node.node_id != self.index.root_id
            and has_direct_leaves(self.index, node)
        ]

    def _prune_tree(self, threshold: float) -> List[TreeNode]:
        survivors: List[TreeNode] = []

        def _walk(node_id: str) -> bool:
            node = self.index.nodes.get(node_id)
            if not node or node.node_type != NodeType.SECTION or node.node_id == self.index.root_id:
                return False
            sec_children = [n.node_id for n in section_children(self.index, node)]
            has_content = has_direct_leaves(self.index, node)
            own_score = node.metadata.get("embed_score", 0.0)

            child_results = [_walk(cid) for cid in sec_children]
            any_child_survives = any(child_results)

            if has_content and own_score > threshold:
                survivors.append(node)
                return True
            if any_child_survives:
                survivors.append(node)
                return True
            return False

        for cid in self.index.nodes[self.index.root_id].children:
            _walk(cid)

        survivors.sort(key=lambda n: n.metadata.get("embed_score", 0.0), reverse=True)
        return survivors

    def _build_toc(self, survivors: List[TreeNode]) -> str:
        if not survivors:
            return "(No sections passed the relevance threshold)"
        survivor_ids = {n.node_id for n in survivors}
        lines: List[str] = []

        def _walk(node_id: str, depth: int) -> None:
            if node_id not in survivor_ids:
                return
            node = self.index.nodes.get(node_id)
            if not node or node.node_type != NodeType.SECTION or node.node_id == self.index.root_id:
                return
            indent = "  " * depth
            if has_direct_leaves(self.index, node):
                summary_text = node.metadata.get("summary", "No summary.").strip() or "No summary."
                e_score = node.metadata.get("embed_score", 0.0)
                lines.append(
                    f"{indent}[ID: {node.node_id}] {node.content}\n"
                    f"{indent}  Summary     : {summary_text}\n"
                    f"{indent}  embed_score : {e_score:.3f}"
                )
            else:
                lines.append(f"{indent}[ID: {node.node_id}] {node.content}")
            for cid in node.children:
                _walk(cid, depth + 1)

        for cid in self.index.nodes[self.index.root_id].children:
            _walk(cid, 0)
        return "\n".join(lines)

    # ── Step 5: TOC routing ───────────────────────────────────────────

    def _route_toc(
        self, query: str, toc: str, trace: TraceRecorder, split_subqueries: bool,
    ) -> List[Dict[str, Any]]:
        if split_subqueries:
            prompt = TOC_ROUTER_PROMPT.format(
                query=query, toc=toc, max_k=self.settings.max_candidates_per_subquery,
            )
            with timed("toc_router_llm", logger):
                router_result = self.llm.generate_structured(
                    prompt, TOC_ROUTER_SCHEMA, label="TOC_ROUTER", trace=trace,
                )
            return router_result.get("subqueries", [])

        prompt = TOC_ROUTER_NOSPLIT_PROMPT.format(
            query=query, toc=toc, max_k=self.settings.max_candidates_per_subquery,
        )
        with timed("toc_router_llm_nosplit", logger):
            router_result = self.llm.generate_structured(
                prompt, TOC_ROUTER_NOSPLIT_SCHEMA, label="TOC_ROUTER_NOSPLIT", trace=trace,
            )
        candidates = router_result.get("candidates", [])
        if not candidates:
            return []
        return [{"subquery": query, "candidates": candidates}]

    # ── Step 6: cross-encoder scoring ─────────────────────────────────

    def _item_text(self, child: TreeNode, parent: TreeNode) -> str:
        if child.node_type in (NodeType.TEXT, NodeType.TABLE):
            return child.content
        if child.node_type == NodeType.IMAGE:
            for img in parent.metadata.get("image_descriptions", []):
                if img.get("source_id") == child.node_id:
                    return img.get("description", "")
            return ""
        return child.content

    def _collect_all_leaves_under(self, node_id: str) -> List[Dict[str, Any]]:
        node = self.index.nodes.get(node_id)
        if not node:
            return []
        items: List[Dict[str, Any]] = []
        for cid in node.children:
            child = self.index.nodes.get(cid)
            if not child:
                continue
            if child.node_type == NodeType.SECTION:
                items.extend(self._collect_all_leaves_under(cid))
            elif child.node_type in _LEAF_TYPES:
                items.append(self._leaf_item(child, node))
        return items

    def _collect_direct_leaves(self, node: TreeNode) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for cid in node.children:
            child = self.index.nodes.get(cid)
            if child and child.node_type in _LEAF_TYPES:
                items.append(self._leaf_item(child, node))
        return items

    def _leaf_item(self, child: TreeNode, parent: TreeNode) -> Dict[str, Any]:
        return {
            "parent_id": child.parent_id, "id": child.node_id, "type": child.node_type.value,
            "page": child.page, "content": child.content, "ce_text": self._item_text(child, parent),
        }

    def _score_and_batch_leaves(
        self, leaves: List[Dict[str, Any]], subquery: str, node: TreeNode,
    ) -> float:
        """Batch leaves into the reranker's context window and score each batch.
        A single leaf exceeding the full budget is chunked independently.
        Section score = max across all batches/chunks.
        Per-leaf scores are cached on the node for later context packing."""
        budget = self.reranker.get_context_budget(subquery)
        leaf_scores: Dict[str, float] = node.metadata.setdefault("leaf_ce_scores", {})

        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_tokens = 0

        for item in leaves:
            text = item.get("ce_text", "")
            if not text.strip():
                leaf_scores[item["id"]] = 0.0
                continue
            item_tokens = self.reranker.count_tokens(text)
            if item_tokens > budget:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([item])
            elif current_tokens + item_tokens > budget:
                batches.append(current_batch)
                current_batch = [item]
                current_tokens = item_tokens
            else:
                current_batch.append(item)
                current_tokens += item_tokens

        if current_batch:
            batches.append(current_batch)

        batch_scores: List[float] = []
        for batch in batches:
            combined = "\n\n".join(item.get("ce_text", "") for item in batch)
            if self.reranker.count_tokens(combined) > budget:
                chunks = self.reranker.chunk_for_scoring(subquery, combined)
                s = max(self.reranker.score(subquery, chunk) for chunk in chunks) if chunks else 0.0
            else:
                s = self.reranker.score(subquery, combined)
            batch_scores.append(s)
            for item in batch:
                leaf_scores[item["id"]] = s

        return max(batch_scores) if batch_scores else 0.0

    def _ce_score_candidate(self, node_id: str, subquery: str) -> float:
        node = self.index.nodes.get(node_id)
        if not node:
            return 0.0
        if "ce_score" in node.metadata:
            return node.metadata["ce_score"]

        sec_children = [n.node_id for n in section_children(self.index, node)]
        has_leaves = has_direct_leaves(self.index, node)

        if sec_children:
            for cid in sec_children:
                self._ce_score_candidate(cid, subquery)
            leaves = self._collect_direct_leaves(node) if has_leaves else []
        else:
            leaves = self._collect_all_leaves_under(node_id)

        score = self._score_and_batch_leaves(leaves, subquery, node) if leaves else 0.0
        node.metadata["ce_score"] = score
        return score

    def _flatten_scored_nodes(self, node_id: str, seen: Set[str]) -> List[Dict[str, Any]]:
        if node_id in seen or node_id not in self.index.nodes:
            return []
        seen.add(node_id)
        node = self.index.nodes[node_id]
        sec_children = [n.node_id for n in section_children(self.index, node)]
        result: List[Dict[str, Any]] = []
        if sec_children:
            for cid in sec_children:
                result.extend(self._flatten_scored_nodes(cid, seen))
            if has_direct_leaves(self.index, node):
                result.append({"node_id": node_id, "ce_score": node.metadata.get("ce_score", 0.0)})
        else:
            result.append({"node_id": node_id, "ce_score": node.metadata.get("ce_score", 0.0)})
        return result

    def _dedup_candidates(
        self, candidates: List[Dict[str, Any]], claimed: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        claimed = claimed or set()
        ids = {c["node_id"] for c in candidates}
        kept: List[Dict[str, Any]] = []
        for c in candidates:
            nid = c["node_id"]
            descendants = self.index.all_descendant_ids(nid)
            if descendants & ids:
                continue
            if nid in claimed or descendants & claimed:
                continue
            kept.append(c)
        return kept

    def _clear_ce_scores(self) -> None:
        for node in self.index.nodes.values():
            node.metadata.pop("ce_score", None)
            node.metadata.pop("leaf_ce_scores", None)

    def _expand_to_content_sections(self, node_id: str) -> List[str]:
        node = self.index.nodes.get(node_id)
        if not node:
            return []
        if has_direct_leaves(self.index, node):
            return [node_id]
        result: List[str] = []
        for child in section_children(self.index, node):
            result.extend(self._expand_to_content_sections(child.node_id))
        return result

    # ── Context window packing (LLM) ──────────────────────────────────

    def _build_context_blocks(
        self,
        all_scored: List[Dict[str, Any]],
        char_budget: int,
    ) -> tuple[List[str], List[RetrievedItem], Set[str]]:
        """Pack leaf text into the LLM context window in ce_score order.
        Appends leaves one by one; the leaf that tips over the char budget
        is truncated at the boundary (simple slice). Section lineage header
        is written once per section, only when it contributes at least one leaf.

        NOTE: does NOT update stage3_ids -- stage 3 is recorded from the
        full CE-ranked list before this method is called.
        """
        context_blocks: List[str] = []
        all_items: List[RetrievedItem] = []
        claimed: Set[str] = set()
        remaining = char_budget

        for entry in all_scored[: self.settings.top_nodes_for_context]:
            nid = entry["node_id"]
            node = self.index.nodes.get(nid)
            if not node:
                continue

            leaf_items = (
                self._collect_direct_leaves(node)
                if section_children(self.index, node)
                else self._collect_all_leaves_under(nid)
            )
            if not leaf_items:
                continue

            leaf_scores = node.metadata.get("leaf_ce_scores", {})
            section_lines: List[str] = []
            section_header = f"[Section ID: {nid} | ce_score {entry['ce_score']:.3f}]"
            header_cost = len(section_header)

            for item in leaf_items:
                text = item.get("ce_text", "")
                if not text.strip():
                    continue

                header_charge = header_cost if not section_lines else 0
                available = remaining - header_charge

                if available <= 0:
                    break

                truncated = text if len(text) <= available else text[:available]
                remaining -= len(truncated) + header_charge
                section_lines.append(truncated)
                all_items.append(RetrievedItem(
                    id=item["id"], content=item["content"], node_type=item["type"],
                    page=item.get("page"), section_id=nid,
                    score=leaf_scores.get(item["id"], entry["ce_score"]),
                ))

                if len(truncated) < len(text) or remaining <= 0:
                    break

            if section_lines:
                context_blocks.append(section_header + "\n" + "\n\n".join(section_lines))
                claimed.add(nid)

            if remaining <= 0:
                break

        return context_blocks, all_items, claimed

    def _process_subquery(
        self,
        subquery: str,
        candidates: List[Dict[str, Any]],
        trace: TraceRecorder,
        claimed: Set[str],
        subquery_idx: int,
    ) -> SubQueryResult:
        candidates = self._dedup_candidates(candidates, claimed)
        if not candidates:
            # Still append empty lists so per-subquery indices stay aligned
            trace.stage_capture.setdefault("stage2_per_subquery", []).append([])
            trace.stage_capture.setdefault("stage3_per_subquery", []).append([])
            return SubQueryResult(subquery, "No relevant content found for this sub-query.", "not_found", [])

        # ── Stage 2: router-ordered content-bearing sections ──────────
        # Preserve the router's ranking; dedup while keeping first occurrence.
        s2_ids: List[str] = []
        s2_seen: Set[str] = set()
        for c in candidates:
            for sid in self._expand_to_content_sections(c["node_id"]):
                if sid not in s2_seen:
                    s2_ids.append(sid)
                    s2_seen.add(sid)
        trace.stage_capture.setdefault("stage2_per_subquery", []).append(s2_ids)

        # ── Cross-encoder rerank ───────────────────────────────────────
        self._clear_ce_scores()
        with timed(f"cross_encoder[{subquery[:30]}]", logger):
            for c in candidates:
                if c["node_id"] in self.index.nodes:
                    self._ce_score_candidate(c["node_id"], subquery)

        seen: Set[str] = set()
        all_scored: List[Dict[str, Any]] = []
        for c in candidates:
            all_scored.extend(self._flatten_scored_nodes(c["node_id"], seen))
        all_scored.sort(key=lambda x: x["ce_score"], reverse=True)

        # ── Stage 3: CE-ranked order, recorded BEFORE context packing ──
        s3_ids: List[str] = [entry["node_id"] for entry in all_scored]
        trace.stage_capture.setdefault("stage3_per_subquery", []).append(s3_ids)

        # ── Context packing + LLM answer ──────────────────────────────
        context_blocks, all_items, section_claimed = self._build_context_blocks(
            all_scored, self.llm.get_context_budget(),
        )
        claimed.update(section_claimed)

        if not context_blocks:
            return SubQueryResult(subquery, "No relevant content found for this sub-query.", "not_found", [])

        prompt = SUBQUERY_ANSWER_PROMPT.format(
            subquery=subquery,
            context="\n\n---\n\n".join(context_blocks),
        )
        with timed(f"llm_answer[{subquery[:30]}]", logger):
            result = self.llm.generate_structured(
                prompt, SUBQUERY_ANSWER_SCHEMA, label=f"SUBQUERY_ANSWER[{subquery[:30]}]", trace=trace,
            )

        fragments = result.get("fragments", [])
        partial_answer = " ".join(f["answer_fragment"] for f in fragments)
        return SubQueryResult(subquery, partial_answer, result.get("confidence", "low"), all_items)

    # ── Full pipeline ──────────────────────────────────────────────────

    def _retrieve_relevant_items(
        self, query: str, trace: TraceRecorder, split_subqueries: bool = True,
    ) -> List[SubQueryResult]:

        query_entities = self._extract_query_entities(query, trace)
        print(query_entities)
        query_entity_vec = self.embedder.embed(query_entities) if query_entities else None
        if query_entity_vec:
            dim = len(query_entity_vec[0])
            query_entity_vec = [sum(v[i] for v in query_entity_vec) / len(query_entity_vec) for i in range(dim)]
        query_text_vec = self.embedder.embed_one(query)

        with timed("embed_scoring", logger):
            self._score_all_sections(query_entity_vec, query_text_vec)

        with timed("pruning", logger):
            if not query_entities:
                survivors = self._all_content_sections()
            else:
                threshold = self.settings.embed_threshold
                survivors = self._prune_tree(threshold)
                if not survivors:
                    survivors = self._all_content_sections()

        # Stage 1: embed-pruned sections, sorted by embed_score desc
        trace.set_stage("stage1_ids", [n.node_id for n in survivors if has_direct_leaves(self.index, n)])

        toc = self._build_toc(survivors)
        subqueries = self._route_toc(query, toc, trace, split_subqueries)

        trace.set_stage("num_subqueries", len(subqueries))
        trace.set_stage("subqueries_text", [sq.get("subquery", "") for sq in subqueries])

        # Initialise per-subquery stage lists (populated inside _process_subquery)
        trace.stage_capture["stage2_per_subquery"] = []
        trace.stage_capture["stage3_per_subquery"] = []

        claimed_ids: Set[str] = set()
        results = [
            self._process_subquery(
                sq["subquery"], sq["candidates"], trace, claimed_ids, idx
            )
            for idx, sq in enumerate(subqueries)
        ]
        return results