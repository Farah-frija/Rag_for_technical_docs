from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from manual_rag.core.interfaces import EmbeddingProvider, LLMClient, VectorRecord, VectorStore
from manual_rag.core.models import ManualIndex, NodeType, TreeNode
from manual_rag.indexing.pdf_to_markdown import extract_pages_text_as_md
from manual_rag.indexing.toc_extractor import TOCResult, extract_toc
from manual_rag.logging_config import get_logger
from manual_rag.utils.text import make_title_pattern, normalise_title, tokenize
from manual_rag.utils.tracing import TraceRecorder

logger = get_logger(__name__)

_PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)
_BLOCK_RE = re.compile(r"<!--(TABLE|IMAGE|TEXT)_START-->(.*?)<!--\1_END-->", re.DOTALL | re.IGNORECASE)


def build_tree(
    pdf_path: str,
    llm: LLMClient,
    embedder: EmbeddingProvider,
    image_output_dir: str = "extracted_images",
    min_image_size: int = 500,
    chunk_token_size: int = 220,
    chunk_overlap_tokens: int = 40,
    toc_candidate_page_cap: int = 10,
    force_mode: Optional[str] = None,
    chunk_with_layout: bool = False,
    chunk_with_layout_context: bool = False,
    plain_store: Optional[VectorStore] = None,
    contextual_store: Optional[VectorStore] = None,
    trace: Optional[TraceRecorder] = None,
) -> Tuple[ManualIndex, List[Dict[str, Any]]]:
    if force_mode is not None and force_mode not in ("toc",):
        raise ValueError(f"force_mode must be 'toc' or None -- got {force_mode!r}")

    path = Path(pdf_path)
    logger.info(f"Extracting pages from '{path.name}' ...")
    pages = extract_pages_text_as_md(pdf_path, image_output_dir=image_output_dir, min_image_size=min_image_size)
    logger.info(f"{len(pages)} pages extracted")

    full_md = "\n\n".join(pages)
    pages_for_llm = [_PAGE_MARKER_RE.sub("", p).strip() for p in pages]

    logger.info("Running TOC extractor ...")
    toc_result: TOCResult = extract_toc(llm, pages_for_llm, candidate_page_cap=toc_candidate_page_cap, trace=trace)

    index = ManualIndex()
    root = TreeNode(node_id="root", node_type=NodeType.SECTION, content=path.stem, level=0)
    index.add_node(root)
    index.root_id = "root"

    toc_detected = bool(toc_result.found and toc_result.toc_tree)

    if force_mode == "toc" and not toc_detected:
        raise ValueError("force_mode='toc' requested but no usable TOC was detected in this PDF.")

    has_toc = toc_detected
    root.metadata["has_toc"] = has_toc
    root.metadata["toc_detected"] = toc_detected
    root.metadata["chunk_with_layout"] = chunk_with_layout
    root.metadata["chunk_with_layout_context"] = chunk_with_layout_context
    root.metadata["chunk_token_size"] = chunk_token_size
    root.metadata["chunk_overlap_tokens"] = chunk_overlap_tokens
    if force_mode is not None:
        root.metadata["forced_mode"] = force_mode

    flat_entries = _flatten_toc_tree(toc_result.toc_tree.get("entries", [])) if toc_detected else []
    corpus: List[Dict[str, Any]] = []

    if has_toc:
        root.metadata["structure"] = "toc_sections"
        logger.info(f"{len(flat_entries)} TOC entries -- building section tree")
        section_nodes = _build_section_nodes(index, root, flat_entries)
        _split_by_sections(
            index=index,
            section_nodes=section_nodes,
            full_md=full_md,
            flat_entries=flat_entries,
            toc_last_page=toc_result.pages[-1],
            embedder=embedder,
            chunk_with_layout=chunk_with_layout,
            chunk_with_layout_context=chunk_with_layout_context,
            chunk_token_size=chunk_token_size,
            chunk_overlap_tokens=chunk_overlap_tokens,
            plain_store=plain_store,
            contextual_store=contextual_store,
            corpus=corpus,
        )
    else:
        logger.warning("No TOC detected -- index will have no material nodes.")
        root.metadata["structure"] = "no_toc"

    logger.info(
        f"Build done: {len(index.nodes)} nodes total, {len(index.material_nodes())} material nodes "
        f"(has_toc={has_toc}, chunk_with_layout={chunk_with_layout}, "
        f"chunk_with_layout_context={chunk_with_layout_context}, "
        f"chunk_token_size={chunk_token_size}, chunk_overlap_tokens={chunk_overlap_tokens})"
    )
    return index, corpus


# ── TOC helpers ───────────────────────────────────────────────────────────────

def _flatten_toc_tree(entries: List[dict]) -> List[dict]:
    flat: List[dict] = []
    for e in entries:
        flat.append({"title": e.get("title", ""), "level": e.get("level", 1), "page": e.get("page")})
        flat.extend(_flatten_toc_tree(e.get("children", [])))
    return flat


def _build_section_nodes(index: ManualIndex, root: TreeNode, flat_entries: List[dict]) -> Dict[str, TreeNode]:
    stack: Dict[int, TreeNode] = {0: root}
    title_to_node: Dict[str, TreeNode] = {}

    for entry in flat_entries:
        title, level, page = entry["title"], entry["level"], entry.get("page")
        parent_level = level - 1
        while parent_level > 0 and parent_level not in stack:
            parent_level -= 1
        parent = stack.get(parent_level, root)

        node = _make_node(NodeType.SECTION, title, level=level, parent_id=parent.node_id, page=page)
        index.add_node(node)
        parent.children.append(node.node_id)

        for lvl in list(stack.keys()):
            if lvl >= level:
                del stack[lvl]
        stack[level] = node
        title_to_node[normalise_title(title)] = node

    return title_to_node


def _page_at_offset(full_md: str, offset: int) -> Optional[int]:
    last_page: Optional[int] = None
    for m in _PAGE_MARKER_RE.finditer(full_md):
        if m.start() > offset:
            break
        last_page = int(m.group(1))
    return last_page


def _find_body_start(full_md_norm: str, flat_entries: List[dict], toc_last_page: int) -> int:
    marker = normalise_title(f"<!-- page {toc_last_page + 1} -->")
    page_start = full_md_norm.find(marker)
    if page_start == -1:
        return 0

    next_marker = normalise_title(f"<!-- page {toc_last_page + 2} -->")
    page_end = full_md_norm.find(next_marker, page_start)
    if page_end == -1:
        page_end = len(full_md_norm)

    last_title = flat_entries[-1]["title"]
    pattern = make_title_pattern(last_title)
    m = pattern.search(full_md_norm, page_start, page_end)
    return m.end() if m else page_end


def _get_two_level_path(index: ManualIndex, node_id: str) -> str:
    node = index.nodes.get(node_id)
    if not node:
        return ""
    parts: List[str] = []
    current = node
    for _ in range(2):
        pid = current.parent_id
        if not pid or pid == "root":
            break
        parent = index.nodes.get(pid)
        if not parent:
            break
        parts.append(parent.content)
        current = parent
    return " > ".join(reversed(parts))


# ── Batch embed + upsert ──────────────────────────────────────────────────────

def _upsert_section_leaves(
    leaves: List[TreeNode],
    section_title: str,
    embedder: EmbeddingProvider,
    index: ManualIndex,
    chunk_token_size: int,
    overlap_tokens: int,
    context: bool,
    store: VectorStore,
    corpus: List[Dict[str, Any]],
) -> None:
    text_leaves = [l for l in leaves if l.node_type != NodeType.IMAGE]

    for leaf in leaves:
        leaf.metadata["bm25_tokens"] = tokenize(leaf.content)
        leaf.metadata["source_sections"] = section_title

    if not text_leaves:
        return

    # Token budget is measured with the embedder's own tokenizer. The reserve for
    # special tokens ([CLS]/[SEP]) is already handled at the embedder level.
    path = _get_two_level_path(index, text_leaves[0].node_id) if context else ""
    path_tokens = embedder.count_tokens(path) if path else 0
    effective_window = max(chunk_token_size - path_tokens, 1)

    def _token_count(text: str) -> int:
        return embedder.count_tokens(text)

    def _upsert_batch(batch: List[TreeNode]) -> None:
        combined = "\n\n".join(n.content for n in batch)
        embed_text = f"{path}\n\n{combined}" if path else combined
        embedding = embedder.embed_one(embed_text)
        record_id = str(uuid.uuid4())
        store.upsert([VectorRecord(
            id=record_id,
            text=combined,
            embedding=embedding,
            metadata={
                "page": batch[0].page or 0,
                "node_type": NodeType.TEXT.value,
                "parent_id": batch[0].parent_id or "",
                "source_sections": section_title,
            },
        )])
        corpus.append({
            "id": record_id,
            "content": combined,
            "page": batch[0].page or 0,
            "parent_id": batch[0].parent_id or "",
            "source_sections": section_title,
        })

    def _upsert_large_leaf(leaf: TreeNode) -> None:
        words = leaf.content.split()
        if not words:
            return
        # Convert the token budget into a word window using this leaf's measured tokens/word.
        tokens_per_word = max(_token_count(leaf.content) / len(words), 1e-6)
        window = max(int(effective_window / tokens_per_word), 1)
        overlap = int(overlap_tokens / tokens_per_word)
        step = max(window - overlap, 1)
        chunks = [" ".join(words[i: i + window]) for i in range(0, len(words), step)]
        for chunk_text in chunks:
            embed_text = f"{path}\n\n{chunk_text}" if path else chunk_text
            embedding = embedder.embed_one(embed_text)
            record_id = str(uuid.uuid4())
            store.upsert([VectorRecord(
                id=record_id,
                text=chunk_text,
                embedding=embedding,
                metadata={
                    "page": leaf.page or 0,
                    "node_type": leaf.node_type.value,
                    "parent_id": leaf.parent_id or "",
                    "source_sections": section_title,
                },
            )])
            corpus.append({
                "id": record_id,
                "content": chunk_text,
                "page": leaf.page or 0,
                "parent_id": leaf.parent_id or "",
                "source_sections": section_title,
            })

    batch: List[TreeNode] = []
    batch_tokens = 0

    for leaf in text_leaves:
        leaf_tokens = _token_count(leaf.content)

        if leaf_tokens > effective_window:
            if batch:
                _upsert_batch(batch)
                batch, batch_tokens = [], 0
            _upsert_large_leaf(leaf)
            continue

        if batch and batch_tokens + leaf_tokens > effective_window:
            _upsert_batch(batch)
            batch, batch_tokens = [], 0

        batch.append(leaf)
        batch_tokens += leaf_tokens

    if batch:
        _upsert_batch(batch)


# ── Core TOC splitter ─────────────────────────────────────────────────────────

def _split_by_sections(
    index: ManualIndex,
    section_nodes: Dict[str, TreeNode],
    full_md: str,
    flat_entries: List[dict],
    toc_last_page: int,
    embedder: Optional[EmbeddingProvider] = None,
    chunk_with_layout: bool = False,
    chunk_with_layout_context: bool = False,
    chunk_token_size: int = 220,
    chunk_overlap_tokens: int = 40,
    plain_store: Optional[VectorStore] = None,
    contextual_store: Optional[VectorStore] = None,
    corpus: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if corpus is None:
        corpus = []

    full_md_norm = "\n".join(normalise_title(line) for line in full_md.splitlines())
    search_from = _find_body_start(full_md_norm, flat_entries, toc_last_page)
    prev_end = search_from
    prev_norm: Optional[str] = None
    prev_title: Optional[str] = None

    def _normalize_chunk(chunk: str) -> str:
        if not re.match(r"\s*<!--(TABLE|IMAGE|TEXT)_START-->", chunk, re.IGNORECASE):
            chunk = "<!--TEXT_START-->\n" + chunk
        if not re.search(r"<!--(TABLE|IMAGE|TEXT)_END-->\s*$", chunk, re.IGNORECASE):
            starts = re.findall(r"<!--(TABLE|IMAGE|TEXT)_START-->", chunk, re.IGNORECASE)
            last_start = starts[-1].upper() if starts else "TEXT"
            chunk = chunk + f"\n<!--{last_start}_END-->"
        return chunk

    def _attach(norm: str, section_title: str, chunk_end: int) -> None:
        chunk = full_md_norm[prev_end:chunk_end].strip()
        node = section_nodes.get(norm)
        if not node or not chunk:
            return

        chunk = _normalize_chunk(chunk)
        page = _page_at_offset(full_md_norm, prev_end)
        type_map = {"TABLE": NodeType.TABLE, "IMAGE": NodeType.IMAGE, "TEXT": NodeType.TEXT}

        new_leaves: List[TreeNode] = []
        insert_at = len(node.children)
        for m in _BLOCK_RE.finditer(chunk):
            block_type, content = m.group(1).upper(), m.group(2).strip()
            if len(content) <= 2:
                continue
            leaf = _make_node(
                type_map[block_type], content, level=node.level + 1,
                parent_id=node.node_id, page=page,
                metadata={"order": insert_at},
            )
            index.add_node(leaf)
            node.children.insert(insert_at, leaf.node_id)
            new_leaves.append(leaf)
            insert_at += 1

        if not new_leaves or embedder is None:
            return

        if chunk_with_layout_context or (chunk_with_layout and contextual_store is not None):
            _upsert_section_leaves(
                leaves=new_leaves,
                section_title=section_title,
                embedder=embedder,
                index=index,
                chunk_token_size=chunk_token_size,
                overlap_tokens=chunk_overlap_tokens,
                context=chunk_with_layout_context,
                store=contextual_store,
                corpus=corpus,
            )
        elif plain_store is not None:
            for leaf in new_leaves:
                if leaf.node_type == NodeType.IMAGE:
                    leaf.metadata["bm25_tokens"] = tokenize(leaf.content)
                    leaf.metadata["source_sections"] = section_title
                    continue
                embedding = embedder.embed_one(leaf.content)
                leaf.metadata["bm25_tokens"] = tokenize(leaf.content)
                leaf.metadata["source_sections"] = section_title
                record_id = leaf.node_id
                plain_store.upsert([VectorRecord(
                    id=record_id,
                    text=leaf.content,
                    embedding=embedding,
                    metadata={
                        "page": leaf.page or 0,
                        "node_type": leaf.node_type.value,
                        "parent_id": leaf.parent_id or "",
                        "source_sections": section_title,
                    },
                )])
                corpus.append({
                    "id": record_id,
                    "content": leaf.content,
                    "page": leaf.page or 0,
                    "parent_id": leaf.parent_id or "",
                    "source_sections": section_title,
                })

    for entry in flat_entries:
        title = entry["title"]
        norm = normalise_title(title)
        pattern = make_title_pattern(title)
        match = pattern.search(full_md_norm, search_from)
        if not match:
            words = norm.split()[:8]
            short = r"\s+".join(re.escape(w) for w in words)
            match = re.compile(r"(?im)" + short).search(full_md_norm, search_from)

        if match:
            if prev_norm is not None:
                _attach(prev_norm, prev_title, match.start())
            prev_norm = norm
            prev_title = title
            prev_end = match.end()
            search_from = match.end()
            logger.debug(f"section located: '{title}' at offset {match.start()}")
        else:
            logger.warning(f"section NOT located: '{title}'")

    if prev_norm is not None:
        _attach(prev_norm, prev_title, len(full_md_norm))


def _make_node(
    node_type: NodeType, content: str, level: int,
    parent_id: Optional[str] = None, page: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> TreeNode:
    return TreeNode(
        node_id=str(uuid.uuid4()), node_type=node_type, content=content,
        level=level, parent_id=parent_id, page=page, metadata=metadata or {},
    )