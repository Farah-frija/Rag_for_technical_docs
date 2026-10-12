"""
app/viewer.py
-------------
Streamlit viewer for ManualRAG tree JSON files (ManualIndex.save_tree output).
Reads the saved tree JSON and reproduces the manual content in a structured,
readable way with node type indicators, images, summaries, and an entity
explorer. Self-contained (json/streamlit/pathlib only) so it never needs the
rest of the manual_rag package -- it only ever reads a JSON file the indexing
pipeline already produced.

Usage:
    streamlit run -m manual_rag.app.viewer
    # or:
    streamlit run manual_rag/app/viewer.py
"""

import json
import streamlit as st
from pathlib import Path


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "ManualRAG Viewer",
    page_icon   = "📖",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: bold;
        margin-right: 6px;
        vertical-align: middle;
    }
    .badge-section { background:#1f77b4; color:white; }
    .badge-text    { background:#2ca02c; color:white; }
    .badge-table   { background:#ff7f0e; color:white; }
    .badge-image   { background:#9467bd; color:white; }
    .badge-warning { background:#d62728; color:white; }

    .page-pill {
        display: inline-block;
        padding: 1px 7px;
        border-radius: 8px;
        font-size: 11px;
        background: #e0e0e0;
        color: #555;
        margin-left: 6px;
        vertical-align: middle;
    }

    .section-box {
        border-left: 3px solid #1f77b4;
        padding-left: 12px;
        margin-bottom: 8px;
    }

    .text-node {
        background: #f9f9f9;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 14px;
        line-height: 1.6;
    }

    .warning-node {
        background: #fff3cd;
        border-left: 4px solid #d62728;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 14px;
    }

    .summary-box {
        background: #eef4fb;
        border-left: 4px solid #1f77b4;
        border-radius: 4px;
        padding: 8px 14px;
        margin: 6px 0 10px 0;
        font-size: 13px;
        color: #333;
        font-style: italic;
    }

    .ai-desc-box {
        background: #f3f0fa;
        border-left: 4px solid #9467bd;
        border-radius: 4px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 12px;
        color: #444;
    }

    .stat-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        margin: 4px;
    }
    .stat-number { font-size: 28px; font-weight: bold; color: #1f77b4; }
    .stat-label  { font-size: 12px; color: #666; }

    .entity-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        background: #e8f4e8;
        color: #2a6e2a;
        border: 1px solid #b6d9b6;
        margin: 2px 3px 2px 0;
    }

    .entities-box {
        background: #f4faf4;
        border-left: 4px solid #2ca02c;
        border-radius: 4px;
        padding: 8px 14px;
        margin: 6px 0 10px 0;
    }

    .entities-box-title {
        font-size: 12px;
        font-weight: bold;
        color: #2a6e2a;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def badge(node_type: str) -> str:
    labels = {
        "section": ("SECTION", "section"),
        "text":    ("TEXT",    "text"),
        "table":   ("TABLE",   "table"),
        "image":   ("IMAGE",   "image"),
        "warning": ("WARN",    "warning"),
    }
    label, css = labels.get(node_type, (node_type.upper(), "text"))
    return f'<span class="badge badge-{css}">{label}</span>'


def page_pill(page: int | None) -> str:
    if page is None:
        return ""
    return f'<span class="page-pill">p.{page}</span>'


def render_entities(entities: list[str]) -> None:
    """Render a list of entity strings as green tags inside a box."""
    if not entities:
        return
    tags = "".join(f'<span class="entity-tag">{e}</span>' for e in entities)
    st.markdown(
        f'<div class="entities-box">'
        f'<div class="entities-box-title">🏷️ Entities</div>'
        f'{tags}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _lookup_desc(descriptions: list[dict], node_id: str) -> str | None:
    for d in descriptions:
        if d.get("source_id") == node_id:
            return d.get("description")
    return None


# ── Node renderer ─────────────────────────────────────────────────────────────

def render_leaf_node(
    node_id: str,
    nodes: dict,
    img_descs: list[dict],
    table_descs: list[dict],
    depth: int = 0,
) -> None:
    node = nodes.get(node_id)
    if node is None:
        return

    ntype    = node.get("node_type", "text")
    content  = node.get("content", "")
    page     = node.get("page")
    metadata = node.get("metadata", {})

    if ntype == "text":
        st.markdown(
            f'<div class="text-node">'
            f'{badge("text")}{page_pill(page)}'
            f'<br>{content}'
            f'</div>',
            unsafe_allow_html=True,
        )

    elif ntype == "warning":
        st.markdown(
            f'<div class="warning-node">'
            f'{badge("warning")}{page_pill(page)}'
            f'<br>⚠️ {content}'
            f'</div>',
            unsafe_allow_html=True,
        )

    elif ntype == "table":
        st.markdown(f'{badge("table")}{page_pill(page)}', unsafe_allow_html=True)
        _render_table(content)
        desc = _lookup_desc(table_descs, node_id)
        if desc:
            st.markdown(
                f'<div class="ai-desc-box">🤖 <b>AI description:</b> {desc}</div>',
                unsafe_allow_html=True,
            )

    elif ntype == "image":
        img_path = metadata.get("image_path", "")
        caption  = content or "Image"

        st.markdown(
            f'{badge("image")}{page_pill(page)} **{caption}**',
            unsafe_allow_html=True,
        )
        if img_path and Path(img_path).exists():
            st.image(img_path, caption=caption, use_container_width=False)
        else:
            st.warning(f"🖼️ Image not found: `{img_path}`")

        desc = _lookup_desc(img_descs, node_id)
        if desc:
            st.markdown(
                f'<div class="ai-desc-box">🤖 <b>AI description:</b> {desc}</div>',
                unsafe_allow_html=True,
            )


def _render_table(content: str) -> None:
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    if not lines:
        return
    if all(l.startswith("|") for l in lines[:2]):
        st.markdown(content, unsafe_allow_html=False)
        return
    if "|" in lines[0]:
        try:
            import pandas as pd
            rows = [[cell.strip() for cell in l.split("|")] for l in lines]
            df = pd.DataFrame(rows[1:], columns=rows[0])
            st.dataframe(df, use_container_width=True)
            return
        except Exception:
            pass
    st.code(content)


# ── Filtered subtree renderer ─────────────────────────────────────────────────

def _render_subtree_filtered(
    node_id: str,
    nodes: dict,
    visible_types: set[str],
    depth: int,
    show_entities: bool = True,
) -> None:
    node = nodes.get(node_id)
    if node is None:
        return

    ntype    = node.get("node_type", "text")
    children = node.get("children", [])
    metadata = node.get("metadata", {})

    if ntype == "section":
        content = node.get("content", "")
        page    = node.get("page")
        heading = min(depth + 1, 6)
        hashes  = "#" * heading

        st.markdown(
            f"{hashes} {badge('section')} {content} {page_pill(page)}",
            unsafe_allow_html=True,
        )

        summary = metadata.get("summary", "")
        if summary:
            st.markdown(
                f'<div class="summary-box">📝 <b>Summary:</b> {summary}</div>',
                unsafe_allow_html=True,
            )

        if show_entities:
            render_entities(metadata.get("entities", []))

        img_descs   = metadata.get("image_descriptions", [])
        table_descs = metadata.get("table_descriptions", [])

        for child_id in children:
            child = nodes.get(child_id, {})
            child_type = child.get("node_type", "")
            if child_type == "section":
                _render_subtree_filtered(child_id, nodes, visible_types, depth + 1, show_entities)
            elif child_type in visible_types:
                render_leaf_node(child_id, nodes, img_descs, table_descs, depth)

    elif ntype in visible_types:
        render_leaf_node(node_id, nodes, [], [], depth)


# ── Entity explorer tab ───────────────────────────────────────────────────────

def render_entity_explorer(nodes: dict) -> None:
    """Flat searchable view of all sections and their entities."""

    # Collect all sections that have entities
    sections_with_entities = [
        (nid, n)
        for nid, n in nodes.items()
        if n.get("node_type") == "section" and n.get("metadata", {}).get("entities")
    ]

    if not sections_with_entities:
        st.info("No entities found. Run `section_summarizer.py` to extract them.")
        return

    # Build flat entity → sections index
    entity_index: dict[str, list[str]] = {}
    for nid, n in sections_with_entities:
        for e in n["metadata"]["entities"]:
            entity_index.setdefault(e.lower(), []).append(n.get("content", nid))

    all_entities = sorted(entity_index.keys())

    col_search, col_stats = st.columns([3, 1])

    with col_search:
        search = st.text_input(
            "🔎 Search entities",
            placeholder="e.g. HTTP, threshold, TokenBucket …",
        )

    with col_stats:
        st.metric("Unique entities", len(all_entities))
        st.metric("Sections with entities", len(sections_with_entities))

    st.divider()

    # ── Per-section view ──────────────────────────────────────────────────────
    st.markdown("#### 📑 Entities per section")

    filtered_sections = [
        (nid, n) for nid, n in sections_with_entities
        if not search or any(
            search.lower() in e.lower()
            for e in n["metadata"]["entities"]
        )
    ]

    if not filtered_sections:
        st.warning("No sections match your search.")
    else:
        for nid, n in filtered_sections:
            entities = n["metadata"]["entities"]
            # highlight matching entities
            if search:
                entities_display = sorted(
                    entities,
                    key=lambda e: search.lower() not in e.lower(),  # matches first
                )
            else:
                entities_display = entities

            with st.expander(
                f"📑 {n.get('content', nid)}  ({len(entities)} entities)",
                expanded=bool(search),
            ):
                summary = n.get("metadata", {}).get("summary", "")
                if summary:
                    st.markdown(
                        f'<div class="summary-box">📝 {summary}</div>',
                        unsafe_allow_html=True,
                    )
                render_entities(entities_display)

    st.divider()

    # ── Entity → sections index ───────────────────────────────────────────────
    st.markdown("#### 🏷️ Entity → sections index")

    filtered_entities = (
        {e: v for e, v in entity_index.items() if search.lower() in e}
        if search else entity_index
    )

    if not filtered_entities:
        st.warning("No entities match your search.")
    else:
        for entity, section_titles in sorted(filtered_entities.items()):
            with st.expander(f"`{entity}`  — {len(section_titles)} section(s)"):
                for title in section_titles:
                    st.markdown(f"- 📑 {title}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str | None, dict]:
    with st.sidebar:
        st.title("📖 ManualRAG Viewer")
        st.divider()

        uploaded = st.file_uploader(
            "Load a tree JSON file",
            type=["json"],
            help="JSON file generated by tree_builder_pdf.py --save-tree",
        )
        if uploaded is not None:
            st.session_state["tree_data"] = json.loads(
                uploaded.read().decode("utf-8")
            )
            st.success(f"Loaded: `{uploaded.name}`")

        data  = st.session_state.get("tree_data", {})
        if not data:
            st.info("Upload a tree JSON file to begin.")
            return None, {}

        nodes = data.get("nodes", {})
        stats = data.get("statistics", {})

        st.divider()
        st.subheader("📊 Statistics")

        stat_items = [
            ("Total",    stats.get("total_nodes", len(nodes)), "🗂️"),
            ("Sections", stats.get("sections", 0),             "📑"),
            ("Texts",    stats.get("texts", 0),                "📝"),
            ("Tables",   stats.get("tables", 0),               "📊"),
            ("Images",   stats.get("images", 0),               "🖼️"),
        ]
        cols = st.columns(2)
        for i, (label, value, icon) in enumerate(stat_items):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-number">{value}</div>'
                    f'<div class="stat-label">{icon} {label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("🗺️ Sections")

        section_nodes = {
            nid: n for nid, n in nodes.items()
            if n.get("node_type") == "section" and nid != "root"
        }

        if section_nodes:
            options = {
                nid: (
                    f"{'  ' * (n.get('level', 1) - 1)}"
                    f"{'└─ ' if n.get('level', 1) > 1 else ''}"
                    f"{n.get('content', '')[:50]}"
                )
                for nid, n in section_nodes.items()
            }
            selected = st.selectbox(
                "Jump to section",
                options      = list(options.keys()),
                format_func  = lambda x: options[x],
            )
            return selected, nodes

        return data.get("root_id"), nodes


# ── Filters ───────────────────────────────────────────────────────────────────

def render_filters() -> tuple[set[str], bool]:
    col_types, col_entities = st.columns([4, 1])

    with col_types:
        st.markdown("#### 🔍 Show node types")
        cols  = st.columns(5)
        types = ["section", "text", "table", "image", "warning"]
        icons = ["📑", "📝", "📊", "🖼️", "⚠️"]
        visible = set()
        for col, ntype, icon in zip(cols, types, icons):
            with col:
                if st.checkbox(f"{icon} {ntype.capitalize()}", value=True, key=f"filter_{ntype}"):
                    visible.add(ntype)

    with col_entities:
        st.markdown("#### &nbsp;")
        show_entities = st.checkbox("🏷️ Show entities", value=True, key="filter_entities")

    return visible, show_entities


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    selected_section, nodes = render_sidebar()

    if not nodes:
        st.title("📖 ManualRAG Viewer")
        st.info(
            "👈 Upload a **tree JSON file** in the sidebar to visualise your manual.\n\n"
            "Generate one with:\n"
            "```bash\n"
            "python tree_builder_pdf.py my_manual.pdf --save-tree output.json\n"
            "```"
        )
        return

    root_id = next(
        (nid for nid, n in nodes.items()
         if n.get("node_type") == "section" and n.get("parent_id") is None),
        None,
    )

    root_node = nodes.get(root_id or "", {})
    st.title(f"📖 {root_node.get('content', 'Manual')}")

    with st.expander("🔍 Filter node types", expanded=False):
        visible_types, show_entities = render_filters()

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📄 Full Manual", "🔖 Jump to Section", "🏷️ Entity Explorer"])

    with tab1:
        if root_id:
            root_children = nodes.get(root_id, {}).get("children", [])
            for child_id in root_children:
                _render_subtree_filtered(child_id, nodes, visible_types, depth=1, show_entities=show_entities)

    with tab2:
        if selected_section and selected_section in nodes:
            section  = nodes[selected_section]
            metadata = section.get("metadata", {})
            st.markdown(
                f"### {badge('section')} {section.get('content', '')} "
                f"{page_pill(section.get('page'))}",
                unsafe_allow_html=True,
            )
            summary = metadata.get("summary", "")
            if summary:
                st.markdown(
                    f'<div class="summary-box">📝 <b>Summary:</b> {summary}</div>',
                    unsafe_allow_html=True,
                )
            if show_entities:
                render_entities(metadata.get("entities", []))
            st.divider()
            img_descs   = metadata.get("image_descriptions", [])
            table_descs = metadata.get("table_descriptions", [])
            for child_id in section.get("children", []):
                child = nodes.get(child_id, {})
                child_type = child.get("node_type", "")
                if child_type == "section":
                    _render_subtree_filtered(child_id, nodes, visible_types, depth=2, show_entities=show_entities)
                elif child_type in visible_types:
                    render_leaf_node(child_id, nodes, img_descs, table_descs, depth=2)
        else:
            st.info("Select a section in the sidebar to jump to it.")

    with tab3:
        render_entity_explorer(nodes)


if __name__ == "__main__":
    main()