"""
utils/json_io.py
------------------
Save/load a `ManualIndex` to/from JSON. Pulled out of `indexing/tree_builder.py`
so retrieval, eval, and the Streamlit viewer can all load an index without
importing the (heavier) indexing package.

Writes are atomic (write to a temp file, then os.replace) so a crash
mid-write never leaves a corrupt/truncated index JSON on disk -- important
since indexing runs can take a long time (many LLM calls) and you don't
want to lose the whole enriched tree to a crash on the final write.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from manual_rag.core.models import ManualIndex, NodeType, TreeNode
from manual_rag.logging_config import get_logger

logger = get_logger(__name__)


def save_tree(index: ManualIndex, output_path: str) -> None:
    output = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(),
        "root_id": index.root_id,
        "nodes": {
            nid: {
                "node_id": n.node_id,
                "node_type": n.node_type.value,
                "content": n.content,
                "level": n.level,
                "parent_id": n.parent_id,
                "page": n.page,
                "metadata": n.metadata,
                "children": n.children,
            }
            for nid, n in index.nodes.items()
        },
        "statistics": {
            "total_nodes": len(index.nodes),
            "sections": sum(1 for n in index.nodes.values() if n.node_type == NodeType.SECTION),
            "texts": sum(1 for n in index.nodes.values() if n.node_type == NodeType.TEXT),
            "tables": sum(1 for n in index.nodes.values() if n.node_type == NodeType.TABLE),
            "images": sum(1 for n in index.nodes.values() if n.node_type == NodeType.IMAGE),
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=out_path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info(f"Tree saved -> {output_path} ({len(index.nodes)} nodes)")


def load_tree(input_path: str) -> ManualIndex:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    index = ManualIndex()
    index.root_id = data["root_id"]
    for nid, nd in data["nodes"].items():
        node = TreeNode(
            node_id=nd["node_id"],
            node_type=NodeType(nd["node_type"]),
            content=nd["content"],
            level=nd["level"],
            parent_id=nd.get("parent_id"),
            page=nd.get("page"),
            metadata=nd.get("metadata", {}),
        )
        node.children = nd.get("children", [])
        index.nodes[nid] = node
    logger.info(f"Tree loaded <- {input_path} ({len(index.nodes)} nodes)")
    return index
