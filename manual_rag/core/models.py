"""
core/models.py
--------------
Shared dataclasses for the whole project (indexing, retrieval, eval).

Moved verbatim in spirit from the original `utils/models.py`, kept here
under `core` because these types are the contract every other layer
(indexing, retrieval, vector store adapters) depends on -- "utils" is
reserved for small stateless helper functions, not domain models.

Mirrors the triplet B = (T, G, M):
  T  -> tree of TreeNode objects (ManualIndex.nodes)
  G  -> KnowledgeGraph (Entity + Relation) -- optional, not populated by the
        current indexing pipeline but kept as an extension point.
  M  -> GTLink mapping entity_id -> set of node_ids
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class NodeType(str, Enum):
    MATERIAL = "material"   # top-level entry from a materials table
    SECTION = "section"     # a section/subsection heading
    TEXT = "text"           # plain paragraph block
    TABLE = "table"         # table block
    IMAGE = "image"         # figure / image block
    WARNING = "warning"     # warning / caution / note callout
    FORMULA = "formula"     # equation block


@dataclass
class TreeNode:
    node_id: str
    node_type: NodeType
    content: str
    page: Optional[int] = None
    level: int = 0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class Entity:
    entity_id: str
    name: str
    entity_type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None



@dataclass
class ManualIndex:
    """The full index B = (T, G, M). T is a flat dict of TreeNode, with tree
    structure encoded via parent_id/children on each node."""

    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    root_id: Optional[str] = None

    def add_node(self, node: TreeNode) -> None:
        self.nodes[node.node_id] = node

    def subtree(self, node_id: str) -> List[TreeNode]:
        result: List[TreeNode] = []
        queue = [node_id]
        while queue:
            nid = queue.pop(0)
            node = self.nodes.get(nid)
            if node:
                result.append(node)
                queue.extend(node.children)
        return result

    def material_nodes(self) -> List[TreeNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.MATERIAL]

    def section_children(self, node: TreeNode) -> List[str]:
        """Direct SECTION children of a node."""
        return [
            cid for cid in node.children
            if cid in self.nodes and self.nodes[cid].node_type == NodeType.SECTION
        ]

    def has_direct_leaves(self, node: TreeNode) -> bool:
        """True if node has at least one direct TEXT/TABLE/IMAGE child."""
        leaf_types = (NodeType.TEXT, NodeType.TABLE, NodeType.IMAGE)
        return any(
            self.nodes[cid].node_type in leaf_types
            for cid in node.children
            if cid in self.nodes
        )

    def all_descendant_ids(self, node_id: str) -> Set[str]:
        result: Set[str] = set()
        node = self.nodes.get(node_id)
        if not node:
            return result
        for cid in node.children:
            result.add(cid)
            result |= self.all_descendant_ids(cid)
        return result
