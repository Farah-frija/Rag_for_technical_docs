"""
docling_viewer.py
------------------
Streamlit viewer for Docling `DocumentConverter` results.

Inspired by a ManualRAG tree-JSON viewer, adapted to Docling's own
`DoclingDocument` structure: it either runs `DocumentConverter().convert(...)`
on an uploaded/pasted PDF (or other Docling-supported format) itself, or
loads a `DoclingDocument` previously saved with `document.save_as_json(...)`.

Usage:
    pip install streamlit docling pandas pillow
    streamlit run docling_viewer.py
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DoclingDocument, PictureItem, TableItem
    from docling_core.types.doc.document import SectionHeaderItem
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Docling Viewer",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Styling ────────────────────────────────────────────────────────────────

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
    .badge-list    { background:#17becf; color:white; }
    .badge-table   { background:#ff7f0e; color:white; }
    .badge-image   { background:#9467bd; color:white; }
    .badge-code    { background:#393b79; color:white; }
    .badge-formula { background:#e377c2; color:white; }
    .badge-minor   { background:#999999; color:white; }

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

    .text-node {
        background: #f9f9f9;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 14px;
        line-height: 1.6;
    }

    .minor-node {
        background: #f4f4f4;
        border-radius: 6px;
        padding: 6px 10px;
        margin: 3px 0;
        font-size: 12px;
        color: #777;
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
    .stat-number { font-size: 26px; font-weight: bold; color: #1f77b4; }
    .stat-label  { font-size: 12px; color: #666; }
</style>
""", unsafe_allow_html=True)


# ── Label taxonomy ────────────────────────────────────────────────────────

HEADER_LABELS = {"title", "section_header"}
TABLE_LABELS = {"table"}
PICTURE_LABELS = {"picture"}
LIST_LABELS = {"list_item"}
CODE_LABELS = {"code"}
FORMULA_LABELS = {"formula"}
MINOR_LABELS = {"caption", "footnote", "page_header", "page_footer"}
# everything else (text, paragraph, key_value_region, document_index, ...)
# falls back to the plain "text" rendering

BADGE_INFO = {
    "title":           ("TITLE",    "section"),
    "section_header":  ("SECTION",  "section"),
    "table":           ("TABLE",    "table"),
    "picture":         ("IMAGE",    "image"),
    "list_item":       ("LIST",     "list"),
    "code":            ("CODE",     "code"),
    "formula":         ("FORMULA",  "formula"),
    "caption":         ("CAPTION",  "minor"),
    "footnote":        ("FOOTNOTE", "minor"),
    "page_header":     ("HEADER",   "minor"),
    "page_footer":     ("FOOTER",   "minor"),
}


def badge(label: str) -> str:
    text, css = BADGE_INFO.get(label, (label.upper().replace("_", " "), "text"))
    return f'<span class="badge badge-{css}">{text}</span>'


def page_pill(item) -> str:
    prov = getattr(item, "prov", None)
    if not prov:
        return ""
    page_no = getattr(prov[0], "page_no", None)
    if page_no is None:
        return ""
    return f'<span class="page-pill">p.{page_no}</span>'


def item_label(item) -> str:
    label = getattr(item, "label", "")
    return label.value if hasattr(label, "value") else str(label)


def item_text(item) -> str:
    return getattr(item, "text", "") or ""


# ── Converting / loading ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_converter(extract_images: bool) -> "DocumentConverter":
    """Cached so the underlying models are only loaded once per setting."""
    if not extract_images:
        return DocumentConverter()
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Skip OCR: 10x faster for digital PDFs
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def convert_uploaded_pdf(uploaded_file, extract_images: bool) -> "DoclingDocument":
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    converter = get_converter(extract_images)
    result = converter.convert(tmp_path)
    return result.document


def load_json_document(uploaded_file) -> "DoclingDocument":
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    return DoclingDocument.load_from_json(tmp_path)


# ── Item renderer ─────────────────────────────────────────────────────────

def render_picture_description(item, document) -> None:
    annotations = getattr(item, "annotations", None) or []
    for ann in annotations:
        desc = getattr(ann, "text", None)
        if desc:
            st.markdown(
                f'<div class="ai-desc-box">🤖 <b>AI description:</b> {desc}</div>',
                unsafe_allow_html=True,
            )


def render_item(item, level: int, document) -> None:
    label = item_label(item)
    indent_px = min(level, 6) * 18

    if label in HEADER_LABELS:
        heading_level = getattr(item, "level", None) if label == "section_header" else 1
        heading_level = min((heading_level or 1) + 1, 6)
        hashes = "#" * heading_level
        st.markdown(
            f'<div style="margin-left:{indent_px}px">{hashes} '
            f'{badge(label)} {item_text(item)} {page_pill(item)}</div>',
            unsafe_allow_html=True,
        )

    elif label in TABLE_LABELS:
        st.markdown(
            f'<div style="margin-left:{indent_px}px">{badge(label)}{page_pill(item)}</div>',
            unsafe_allow_html=True,
        )
        try:
            df = item.export_to_dataframe(doc=document)
            st.dataframe(df, use_container_width=True)
        except Exception:
            try:
                st.markdown(item.export_to_html(doc=document), unsafe_allow_html=True)
            except Exception:
                st.caption("(table could not be rendered)")
        try:
            img = item.get_image(document)
            if img is not None:
                with st.expander("🖼️ Table image"):
                    st.image(img, use_container_width=False)
        except Exception:
            pass

    elif label in PICTURE_LABELS:
        caption = ""
        try:
            caption = item.caption_text(document) or ""
        except Exception:
            pass
        st.markdown(
            f'<div style="margin-left:{indent_px}px">{badge(label)}{page_pill(item)} '
            f'{("**" + caption + "**") if caption else ""}</div>',
            unsafe_allow_html=True,
        )
        try:
            img = item.get_image(document)
        except Exception:
            img = None
        if img is not None:
            st.image(img, caption=caption or None, use_container_width=False)
        else:
            st.caption("🖼️ No embedded image (re-convert with image extraction on)")
        render_picture_description(item, document)

    elif label in MINOR_LABELS:
        text = item_text(item)
        if not text:
            return
        st.markdown(
            f'<div class="minor-node" style="margin-left:{indent_px}px">'
            f'{badge(label)}{page_pill(item)}<br>{text}</div>',
            unsafe_allow_html=True,
        )

    else:
        # text, paragraph, list_item, code, formula, key_value_region, ...
        text = item_text(item)
        if not text:
            return
        prefix = "• " if label in LIST_LABELS else ""
        if label in CODE_LABELS:
            st.markdown(
                f'<div style="margin-left:{indent_px}px">{badge(label)}{page_pill(item)}</div>',
                unsafe_allow_html=True,
            )
            st.code(text)
        else:
            st.markdown(
                f'<div class="text-node" style="margin-left:{indent_px}px">'
                f'{badge(label)}{page_pill(item)}<br>{prefix}{text}</div>',
                unsafe_allow_html=True,
            )


# ── Document-wide helpers ────────────────────────────────────────────────

@st.cache_data(show_spinner=False, hash_funcs={"docling_core.types.doc.document.DoclingDocument": id})
def build_item_list(_document):
    """Flatten the document once into (item, level, label) tuples."""
    items = []
    for item, level in _document.iterate_items():
        items.append((item, level, item_label(item)))
    return items


def compute_stats(items) -> dict:
    stats = {"total": len(items)}
    for _, _, label in items:
        stats[label] = stats.get(label, 0) + 1
    return stats


# ── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("📖 Docling Viewer")
        st.divider()

        if not DOCLING_AVAILABLE:
            st.error(
                "`docling` is not installed in this environment.\n\n"
                "Run `pip install docling` and restart the app."
            )
            return None

        mode = st.radio(
            "Source",
            ["Convert a document", "Load a saved Docling JSON"],
            help="Converting runs Docling's models now. Loading a JSON re-opens "
                 "a document you already converted with `document.save_as_json(...)`.",
        )

        if mode == "Convert a document":
            extract_images = st.checkbox(
                "Extract picture / table images", value=True,
                help="Slower, but lets the viewer show pictures and table snapshots.",
            )
            uploaded = st.file_uploader(
                "Upload a PDF (or other Docling-supported file)",
                type=["pdf", "docx", "pptx", "xlsx", "html", "md"],
            )
            if uploaded is not None and st.button("🚀 Convert", type="primary"):
                with st.spinner("Running Docling conversion… this can take a while on first run."):
                    document = convert_uploaded_pdf(uploaded, extract_images)
                st.session_state["document"] = document
                st.session_state["doc_name"] = uploaded.name
                st.success(f"Converted: `{uploaded.name}`")

        else:
            uploaded = st.file_uploader("Upload a Docling JSON export", type=["json"])
            if uploaded is not None:
                try:
                    document = load_json_document(uploaded)
                    st.session_state["document"] = document
                    st.session_state["doc_name"] = uploaded.name
                    st.success(f"Loaded: `{uploaded.name}`")
                except Exception as exc:
                    st.error(f"Could not load this file as a DoclingDocument: {exc}")

        document = st.session_state.get("document")
        if document is None:
            st.info("👈 Convert a document or load a saved Docling JSON to begin.")
            return None

        items = build_item_list(document)
        stats = compute_stats(items)

        st.divider()
        with st.expander("💾 Save this document"):
            st.download_button(
                "Download as Docling JSON",
                data=json.dumps(document.export_to_dict(), indent=2),
                file_name=f"{Path(st.session_state.get('doc_name', 'document')).stem}.docling.json",
                mime="application/json",
                help="Reload this later via 'Load a saved Docling JSON' to skip re-converting.",
            )

        st.divider()
        st.subheader("📊 Statistics")
        stat_items = [
            ("Total",    stats.get("total", 0),                                   "🗂️"),
            ("Sections", stats.get("section_header", 0) + stats.get("title", 0),  "📑"),
            ("Text",     stats.get("text", 0) + stats.get("paragraph", 0),        "📝"),
            ("Tables",   stats.get("table", 0),                                   "📊"),
            ("Pictures", stats.get("picture", 0),                                 "🖼️"),
            ("Pages",    getattr(document, "num_pages", lambda: None)() or len(getattr(document, "pages", {}) or {}), "📄"),
        ]
        cols = st.columns(2)
        for i, (lbl, value, icon) in enumerate(stat_items):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-number">{value}</div>'
                    f'<div class="stat-label">{icon} {lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("🗺️ Sections")
        headers = [(i, it, lvl) for i, (it, lvl, lbl) in enumerate(items) if lbl in HEADER_LABELS]
        selected_idx = None
        if headers:
            options = {i: f"{'  ' * lvl}{item_text(it)[:60] or '(untitled)'}" for i, it, lvl in headers}
            selected_idx = st.selectbox(
                "Jump to section",
                options=list(options.keys()),
                format_func=lambda x: options[x],
            )

        return {"document": document, "items": items, "stats": stats, "selected_idx": selected_idx}


# ── Filters ───────────────────────────────────────────────────────────────

def render_filters(present_labels: set) -> set:
    st.markdown("#### 🔍 Show element types")
    labels = sorted(present_labels)
    visible = set()
    cols = st.columns(min(len(labels), 6) or 1)
    for i, label in enumerate(labels):
        with cols[i % len(cols)]:
            nice = BADGE_INFO.get(label, (label.replace("_", " ").title(), ""))[0].title()
            if st.checkbox(nice, value=True, key=f"filter_{label}"):
                visible.add(label)
    return visible


# ── Tabs ──────────────────────────────────────────────────────────────────

def render_full_document(items, visible_labels):
    for item, level, label in items:
        if label in visible_labels:
            render_item(item, level, st.session_state["document"])


def render_jump_to_section(items, selected_idx, visible_labels):
    if selected_idx is None:
        st.info("No sections detected in this document — use 'Full Document' instead.")
        return
    start_item, start_level, _ = items[selected_idx]
    end = len(items)
    for j in range(selected_idx + 1, len(items)):
        _, lvl, lbl = items[j]
        if lbl in HEADER_LABELS and lvl <= start_level:
            end = j
            break
    for item, level, label in items[selected_idx:end]:
        if label in visible_labels:
            render_item(item, level, st.session_state["document"])


def render_tables_gallery(document):
    tables = getattr(document, "tables", [])
    if not tables:
        st.info("No tables detected.")
        return
    for i, table in enumerate(tables):
        st.markdown(f"{badge('table')}{page_pill(table)} **Table {i + 1}**", unsafe_allow_html=True)
        try:
            st.dataframe(table.export_to_dataframe(doc=document), use_container_width=True)
        except Exception:
            st.caption("(could not render as a dataframe)")
        st.divider()


def render_pictures_gallery(document):
    pictures = getattr(document, "pictures", [])
    if not pictures:
        st.info("No pictures detected.")
        return
    cols = st.columns(3)
    for i, pic in enumerate(pictures):
        with cols[i % 3]:
            try:
                img = pic.get_image(document)
            except Exception:
                img = None
            caption = ""
            try:
                caption = pic.caption_text(document) or f"Picture {i + 1}"
            except Exception:
                caption = f"Picture {i + 1}"
            if img is not None:
                st.image(img, caption=caption, use_container_width=True)
            else:
                st.caption(f"{caption} (no embedded image)")
            render_picture_description(pic, document)


def render_markdown_export(document):
    md = document.export_to_markdown()
    st.download_button("⬇️ Download Markdown", data=md, file_name="document.md", mime="text/markdown")
    st.markdown(md)


# ── Main app ──────────────────────────────────────────────────────────────

def main():
    ctx = render_sidebar()

    if ctx is None:
        st.title("📖 Docling Viewer")
        if DOCLING_AVAILABLE:
            st.info(
                "👈 Convert a document or load a saved Docling JSON in the sidebar.\n\n"
                "```python\n"
                "from docling.document_converter import DocumentConverter\n\n"
                "converter = DocumentConverter()\n"
                "result = converter.convert('dataset/Insurance_Handbook_20103.pdf')\n"
                "```"
            )
        return

    document = ctx["document"]
    items = ctx["items"]
    present_labels = {lbl for _, _, lbl in items}

    st.title(f"📖 {st.session_state.get('doc_name', 'Document')}")

    with st.expander("🔍 Filter element types", expanded=False):
        visible_labels = render_filters(present_labels)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📄 Full Document", "🔖 Jump to Section", "📊 Tables", "🖼️ Pictures", "📝 Markdown"]
    )

    with tab1:
        render_full_document(items, visible_labels)

    with tab2:
        render_jump_to_section(items, ctx["selected_idx"], visible_labels)

    with tab3:
        render_tables_gallery(document)

    with tab4:
        render_pictures_gallery(document)

    with tab5:
        render_markdown_export(document)


if __name__ == "__main__":
    main()