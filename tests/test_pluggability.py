"""
Smoke tests proving the pluggability contracts hold, using fakes -- no
network calls, no model downloads, no API keys required. Run with:

    pytest tests/ -v
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from manual_rag.core.interfaces import (
    EmbeddingProvider,
    LLMClient,
    VectorRecord,
)
from manual_rag.core.models import ManualIndex, NodeType, TreeNode
from manual_rag.utils.text import cosine_similarity, normalise_title, tokenize
from manual_rag.vectorstore.in_memory_store import InMemoryVectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words embedding -- good enough to test wiring."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [[float(len(t)), float(t.count(" ") + 1)] for t in texts]

    @property
    def dimension(self) -> int:
        return 2


class FakeLLMClient(LLMClient):
    def __init__(self, canned_response: Dict[str, Any]) -> None:
        self._canned = canned_response

    def generate_structured(
        self, prompt: str, schema: Dict[str, Any], *, label: str = "unlabeled",
        system_instruction: Optional[str] = None, temperature: float = 0.0, trace=None,
    ) -> Dict[str, Any]:
        return self._canned

    @property
    def model_name(self) -> str:
        return "fake-model"


def test_in_memory_vector_store_round_trip():
    store = InMemoryVectorStore()
    store.upsert([
        VectorRecord(id="a", text="hello world", embedding=[1.0, 0.0]),
        VectorRecord(id="b", text="goodbye", embedding=[0.0, 1.0]),
    ])
    assert store.count() == 2
    matches = store.query([1.0, 0.0], top_k=1)
    assert matches[0].id == "a"


def test_fake_embedding_provider_satisfies_interface():
    provider = FakeEmbeddingProvider()
    vectors = provider.embed(["hi", "a longer sentence here"])
    assert len(vectors) == 2
    assert provider.dimension == 2
    assert provider.embed_one("x") == vectors_for_single("x")


def vectors_for_single(text: str) -> List[float]:
    return FakeEmbeddingProvider().embed([text])[0]


def test_fake_llm_client_satisfies_interface():
    llm = FakeLLMClient({"entities": ["foo", "bar"]})
    result = llm.generate_structured("prompt", {"type": "object"}, label="TEST")
    assert result == {"entities": ["foo", "bar"]}


def test_text_utils():
    assert tokenize("The Quick Brown Fox!") == ["quick", "brown", "fox"]
    assert normalise_title("  Section 4.1.  ") == "section 4.1"
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_manual_index_tree_helpers():
    index = ManualIndex()
    root = TreeNode(node_id="root", node_type=NodeType.SECTION, content="Doc", level=0)
    section = TreeNode(node_id="s1", node_type=NodeType.SECTION, content="Intro", level=1, parent_id="root")
    leaf = TreeNode(node_id="t1", node_type=NodeType.TEXT, content="hello", level=2, parent_id="s1")
    root.children = ["s1"]
    section.children = ["t1"]
    for n in (root, section, leaf):
        index.add_node(n)
    index.root_id = "root"

    assert index.has_direct_leaves(section) is True
    assert index.section_children(root) == ["s1"]
    assert index.all_descendant_ids("root") == {"s1", "t1"}
