"""
indexing/pdf_to_markdown.py
------------------------------
Converts a PDF to a structured Markdown string + extracts raster images to
disk. Ported from the original `indexing/pdf_to_md.py`, relocated into the
package and switched to the shared logger, with one correctness fix: the
left/right text-flank extraction now uses its own `lt`/`rt` variable instead
of stale `t` (the original always emitted the "text above" fragment's
content for the left/right flanks too).

Strategy per page:
  1. pdfplumber -> tables first (Markdown table syntax)
  2. PyMuPDF    -> raster images embedded in the page
  3. pdfplumber -> extract text around tables and image coordinates
                   (left/right flanks + vertical gaps)

The returned Markdown has a `<!-- page N -->` comment at the start of each
page so downstream code can map content back to page numbers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber

from manual_rag.logging_config import get_logger

logger = get_logger(__name__)


def pdf_to_markdown(
    pdf_path: str,
    image_output_dir: str = "extracted_images",
    min_image_size: int = 500,
    target_page: Optional[int] = None,
) -> str:
    logger.info(f"Converting '{pdf_path}' to markdown", extra={"target_page": target_page})
    os.makedirs(image_output_dir, exist_ok=True)
    pages_md: List[str] = []

    fitz_doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber_pdf:
        total_pages = len(plumber_pdf.pages)
        logger.info(f"Loaded '{pdf_path}': {total_pages} pages")

        for page_idx, plumber_page in enumerate(plumber_pdf.pages):
            page_num = page_idx + 1
            if target_page is not None and page_num != target_page:
                continue
            page_md = _convert_page(
                plumber_page=plumber_page,
                fitz_page=fitz_doc[page_idx],
                fitz_doc=fitz_doc,
                page_number=page_num,
                image_output_dir=image_output_dir,
                min_image_size=min_image_size,
            )
            pages_md.append(page_md)

    fitz_doc.close()
    logger.info(f"Conversion finished: {len(pages_md)} page(s) processed")
    return "\n\n".join(pages_md)


# ── Step 1: image extraction ────────────────────────────────────────────────

def _extract_images_with_y(
    fitz_page: fitz.Page,
    fitz_doc: fitz.Document,
    page_number: int,
    image_output_dir: str,
    min_image_size: int,
) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, float, float]]]:
    parts: List[Tuple[float, float, str]] = []
    image_bboxes: List[Tuple[float, float, float, float]] = []
    seen_xrefs: set = set()

    for info in fitz_page.get_image_info(xrefs=True):
        xref = info.get("xref")
        if not xref or xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        bbox = info["bbox"]
        ox0, oy0, ox1, oy1 = bbox
        if info["width"] * info["height"] < min_image_size:
            continue

        try:
            extracted = fitz_doc.extract_image(xref)
        except Exception:
            continue

        filename = f"img_p{page_number:04d}_{xref:06d}.{extracted['ext']}"
        img_path = os.path.join(image_output_dir, filename)
        try:
            with open(img_path, "wb") as f:
                f.write(extracted["image"])
        except OSError:
            continue

        parts.append((oy0, ox0, f"<!--IMAGE_START-->\n![Image page {page_number}]({img_path})\n<!--IMAGE_END-->"))
        image_bboxes.append((ox0, oy0, ox1, oy1))

    return parts, image_bboxes


# ── Step 2: text around obstacles ───────────────────────────────────────────

def _extract_text_between_obstacles(
    plumber_page,
    table_bboxes: List[tuple],
    image_bboxes: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, str]]:
    page_height = plumber_page.height
    page_width = plumber_page.width

    obstacles = [(b[0], b[1], b[2], b[3]) for b in table_bboxes] + list(image_bboxes)
    obstacles.sort(key=lambda b: b[1])

    result: List[Tuple[float, float, str]] = []
    current_y = 0.0

    for ox0, oy0, ox1, oy1 in obstacles:
        # text strictly above this obstacle (full width)
        if oy0 > current_y + 0.5:
            strip = plumber_page.crop((0, current_y, page_width, oy0))
            t = strip.extract_text()
            if t and t.strip():
                result.append((current_y, 0.0, f"<!--TEXT_START-->\n{t.strip()}\n<!--TEXT_END-->"))

        # left flank of the obstacle
        if ox0 > 0.5:
            left_strip = plumber_page.crop((0, oy0, ox0, oy1))
            lt = left_strip.extract_text()
            if lt and lt.strip():
                result.append((oy0, 0.0, f"<!--TEXT_START-->\n{lt.strip()}\n<!--TEXT_END-->"))

        # right flank of the obstacle
        if ox1 < page_width - 0.5:
            right_strip = plumber_page.crop((ox1, oy0, page_width, oy1))
            rt = right_strip.extract_text()
            if rt and rt.strip():
                result.append((oy0, ox1, f"<!--TEXT_START-->\n{rt.strip()}\n<!--TEXT_END-->"))

        current_y = max(current_y, oy1)

    # remaining text below the last obstacle
    if current_y < page_height - 0.5:
        strip = plumber_page.crop((0, current_y, page_width, page_height))
        t = strip.extract_text()
        if t and t.strip():
            result.append((current_y, 0.0, f"<!--TEXT_START-->\n{t.strip()}\n<!--TEXT_END-->"))

    return result


# ── Per-page orchestrator ────────────────────────────────────────────────────

def _convert_page(
    plumber_page,
    fitz_page,
    fitz_doc: fitz.Document,
    page_number: int,
    image_output_dir: str,
    min_image_size: int,
) -> str:
    parts: List[Tuple[float, float, str]] = []

    # A. tables
    table_bboxes: List[tuple] = []
    tables = plumber_page.extract_tables()
    if tables:
        for tbl_obj in plumber_page.find_tables():
            bbox = tbl_obj.bbox
            table = tbl_obj.extract()
            if table:
                md_table = _table_to_markdown(table)
                parts.append((bbox[1], bbox[0], f"<!--TABLE_START-->\n{md_table}\n<!--TABLE_END-->"))
                table_bboxes.append(bbox)

    # B. images
    image_parts, image_bboxes = _extract_images_with_y(
        fitz_page, fitz_doc, page_number, image_output_dir, min_image_size
    )
    parts.extend(image_parts)

    # C. text around obstacles
    parts.extend(_extract_text_between_obstacles(plumber_page, table_bboxes, image_bboxes))

    # D. sort (top-to-bottom, then left-to-right) and join
    parts.sort(key=lambda t: (t[0], t[1]))
    body = "\n\n".join(fragment for _, _, fragment in parts)

    return f"<!-- page {page_number} -->\n\n{body}" if body.strip() else f"<!-- page {page_number} -->"


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    if not table:
        return ""
    rows = [[_clean_cell(c) for c in row] for row in table]
    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    widths = [max(max(len(rows[i][c]) for i in range(len(rows))), 3) for c in range(col_count)]

    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    header = fmt_row(rows[0])
    separator = "| " + " | ".join("-" * widths[i] for i in range(col_count)) + " |"
    data_rows = [fmt_row(r) for r in rows[1:]]
    return "\n".join([header, separator] + data_rows)


def _clean_cell(value: Optional[str]) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("|", "\\|")).strip()


def extract_pages_text_as_md(
    pdf_path: str,
    image_output_dir: str = "extracted_images",
    min_image_size: int = 500,
) -> List[str]:
    """Extract one Markdown string per page (used by the TOC extractor, which
    needs per-page text, and by the tree builder, which joins them back
    into one document)."""
    os.makedirs(image_output_dir, exist_ok=True)
    pages: List[str] = []
    pdf_path = str(Path(pdf_path))
    fitz_doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for idx, plumber_page in enumerate(pdf.pages):
            pages.append(_convert_page(
                plumber_page=plumber_page,
                fitz_page=fitz_doc[idx],
                fitz_doc=fitz_doc,
                page_number=idx + 1,
                image_output_dir=image_output_dir,
                min_image_size=min_image_size,
            ))
    fitz_doc.close()
    return pages
