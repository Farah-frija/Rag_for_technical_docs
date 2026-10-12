"""Shared structural predicates over ManualIndex / TreeNode."""

from __future__ import annotations
from manual_rag.core.models import ManualIndex, NodeType, TreeNode


import re

from typing import Any, Dict, List, Tuple

_LEAF_TYPES = frozenset({NodeType.TEXT, NodeType.TABLE, NodeType.IMAGE})

_CITE_WRAP_RE = re.compile(r"^\s*\[?\s*(?:chunk_id\s*=\s*)?|\s*\]?\s*$", re.IGNORECASE)

def direct_leaves(index: ManualIndex, node: TreeNode) -> List[TreeNode]:
    """Return direct children that are content leaves (TEXT / TABLE / IMAGE)."""
    return [
        index.nodes[cid]
        for cid in node.children
        if cid in index.nodes and index.nodes[cid].node_type in _LEAF_TYPES
    ]


def section_children(index: ManualIndex, node: TreeNode) -> List[TreeNode]:
    """Return direct children that are SECTION nodes."""
    return [
        index.nodes[cid]
        for cid in node.children
        if cid in index.nodes and index.nodes[cid].node_type == NodeType.SECTION
    ]


def has_direct_leaves(index: ManualIndex, node: TreeNode) -> bool:
    return any(
        index.nodes[cid].node_type in _LEAF_TYPES
        for cid in node.children
        if cid in index.nodes
    )
def get_ancestry_path(index: "ManualIndex", node_id: str) -> str:
    """Returns ' > '-joined section titles from root down to node_id's parent."""
    path: list[str] = []
    nid = index.nodes[node_id].parent_id if node_id in index.nodes else None
    while nid and nid in index.nodes:
        node = index.nodes[nid]
        if node.node_id == index.root_id:
            break
        path.append(node.content)
        nid = node.parent_id
    return " > ".join(reversed(path))


def build_contextual_text(index: "ManualIndex", node_id: str, chunk_text: str) -> str:
    """Prepends ancestry path to chunk text for contextual embedding."""
    path = get_ancestry_path(index, node_id)
    return f"{path}\n\n{chunk_text}" if path else chunk_text


def fmt_sections(chunk: Dict[str, Any]) -> str:
    """`source_sections` may be a list or a string depending on how the corpus
    was built; normalise for display in the prompt."""
    val = chunk.get("source_sections")
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v) or "unknown"
    return str(val) if val else "unknown"


def clean_cited_id(raw: Any) -> str:
    """Strip '[chunk_id=...]' wrappers and quotes the LLM may have copied."""
    return _CITE_WRAP_RE.sub("", str(raw)).strip().strip("'\"")


def resolve_cited_ids(
    raw_cited: List[str],
    shown_ids: List[str],
) -> Tuple[List[str], List[str]]:
    """Keep only cited ids that were actually shown to the LLM (exact match
    after stripping wrappers). Valid ids are returned in the order of
    `shown_ids` (cross-encoder rank), so order-sensitive context precision
    stays comparable with the previous stage.

    Returns (valid_ids, unresolved_raw_strings)."""
    shown = {str(c) for c in shown_ids}
    cited = {clean_cited_id(r) for r in (raw_cited or [])}
    valid = [cid for cid in shown_ids if str(cid) in cited]
    unresolved = [r for r in (raw_cited or []) if clean_cited_id(r) not in shown]
    return valid, unresolved