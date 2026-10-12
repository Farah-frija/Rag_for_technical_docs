"""
indexing/summarizer.py
--------------------------
Bottom-up summarisation pass over a `ManualIndex` tree.

For each SECTION node (leaves first, then parents):
  - Sections with direct content leaves (TEXT / TABLE / IMAGE) ->
    LEAF_SECTION_ANALYSIS_PROMPT (summary + entities + image/table descriptions)
  - Sections with no direct content leaves (pure parents) -> skipped entirely;
    their sub-sections carry their own summaries.

After all LLM calls, a single pass encodes every section's entity list via
the injected `EmbeddingProvider` and caches it in
`node.metadata["entity_embedding"]` so retrieval can skip re-encoding at
query time.
"""

from __future__ import annotations

from typing import List, Optional

from manual_rag.core.interfaces import EmbeddingProvider, LLMClient
from manual_rag.core.models import ManualIndex, NodeType, TreeNode
from manual_rag.logging_config import get_logger
from manual_rag.prompts.section_analysis import (
    LEAF_SCHEMA,
    LEAF_SECTION_ANALYSIS_PROMPT,
    format_items,
)
from manual_rag.utils.shared_helpers import direct_leaves
from manual_rag.utils.tracing import TraceRecorder

logger = get_logger(__name__)


def _section_path(index: ManualIndex, node: TreeNode) -> str:
    parts: List[str] = []
    current: Optional[TreeNode] = node
    while current and current.node_id != index.root_id:
        parts.append(current.content[:60])
        current = index.nodes.get(current.parent_id) if current.parent_id else None
    return " > ".join(reversed(parts))


def _postorder(index: ManualIndex, node_id: str) -> List[str]:
    node = index.nodes.get(node_id)
    if not node:
        return []
    result: List[str] = []
    for cid in node.children:
        result.extend(_postorder(index, cid))
    result.append(node_id)
    return result


def summarize_tree(index: ManualIndex, llm: LLMClient, trace: Optional[TraceRecorder] = None) -> None:
    """Populate node.metadata["summary"] / ["entities"] / ["image_descriptions"]
    / ["table_descriptions"] for every SECTION node that has direct content
    leaves. Pure parent sections (no direct leaves) are skipped."""
    order = _postorder(index, index.root_id)

    for node_id in order:
        node = index.nodes.get(node_id)
        if not node or node.node_type != NodeType.SECTION or node.node_id == index.root_id:
            continue

        leaves = direct_leaves(index, node)
        if not leaves:
            continue

        items = [
            {"id": c.node_id, "type": c.node_type.value,
             "content": c.content, "page": c.page}
            for c in leaves
        ]
        prompt = LEAF_SECTION_ANALYSIS_PROMPT.format(
            title=node.content,
            path=_section_path(index, node),
            items=format_items(items),
        )
        try:
            result = llm.generate_structured(
                prompt, LEAF_SCHEMA,
                label=f"LEAF|{node.content}", trace=trace,
            )
            node.metadata["summary"]            = result.get("summary", "")
            node.metadata["entities"]           = result.get("entities", [])
            node.metadata["image_descriptions"] = result.get("image_descriptions", [])
            node.metadata["table_descriptions"] = result.get("table_descriptions", [])
        except Exception as exc:  # noqa: BLE001
            logger.error(f"summarisation failed for '{node.content}': {exc}")

        logger.info(
            f"summarised '{node.content}': "
            f"summary={bool(node.metadata.get('summary'))} "
            f"entities={len(node.metadata.get('entities', []))}"
        )


def embed_entity_cache(index: ManualIndex, embedder: EmbeddingProvider) -> None:
    """Encode each section's entity list as a mean-pooled embedding, cached
    in node.metadata["entity_embedding"]. Call AFTER summarize_tree() and
    BEFORE saving the index."""
    nodes_with_entities = [
        n for n in index.nodes.values()
        if n.node_type == NodeType.SECTION and n.metadata.get("entities")
    ]
    if not nodes_with_entities:
        logger.warning("No section entities found -- skipping entity embedding cache.")
        return

    logger.info(f"Encoding entities for {len(nodes_with_entities)} sections ...")
    for node in nodes_with_entities:
        embeddings = embedder.embed(node.metadata["entities"])
        dim = len(embeddings[0])
        mean_emb = [sum(v[i] for v in embeddings) / len(embeddings) for i in range(dim)]
        node.metadata["entity_embedding"] = mean_emb
    logger.info("Entity embeddings cached.")