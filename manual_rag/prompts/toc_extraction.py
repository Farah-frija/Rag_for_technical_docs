"""
prompts/toc_extraction.py
---------------------------
Prompts for detecting and extracting a document's Table of Contents,
page by page (fresh probe vs. continuation probe with prior context).
Consumed by `manual_rag.indexing.toc_graph`.
"""

from __future__ import annotations

_ROLE = "You are a document analysis expert and structure extractor."

_TOC_DEFINITION = """A TOC (Table of Contents) lists section/chapter titles paired with page numbers.
It may use:
  - dot leaders        (Introduction ......... 3)
  - tab spacing        (Chapter 1          12)
  - plain number at end of line
  - numbered entries   (1.1, 2.3.1 ...)
  - bullets / dashes / # prefixes followed by a title and a page number

It is NOT a TOC if it is:
  - a regular content section
  - an index
  - a list of figures or tables
  - a bibliography or references page"""

_ENTRY_RULES_FRESH = """Rules:
  - Return entries as a FLAT ordered list -- no nesting, no "children" field.
  - Use "level" to encode hierarchy: 1 = chapter, 2 = section, 3 = subsection.
  - Preserve the exact document order.
  - COMPLETENESS: every line of the TOC visible on this page must produce
    exactly one entry -- do not skip, merge, or summarise any line.
  - Do NOT invent entries not present in the text.
  - Numbered prefixes (1., 2.1, 3.2.4) are the primary level signal.
  - Indentation / whitespace alignment is a secondary signal.
  - If status is no_toc, return "entries": []."""

_ENTRY_RULES_CONTINUE = """Rules:
  - Do NOT repeat any entry already present in the existing tree.
  - Extract ONLY the entries that appear on this new page.
  - Return entries as a FLAT ordered list -- no nesting, no "children" field.
  - Use "level" to encode hierarchy: 1 = chapter, 2 = section, 3 = subsection.
  - COMPLETENESS: every line of the TOC visible on this page must produce
    exactly one entry -- do not skip, merge, or summarise any line.
  - Preserve the exact document order.
  - Do NOT invent entries not present in the text.
  - Numbered prefixes (1., 2.1, 3.2.4) are the primary level signal.
  - Indentation / whitespace alignment is a secondary signal.
  - If status is no_toc, return "entries": []."""

_OUTPUT_INSTRUCTION = "Respond ONLY with valid JSON -- no prose, no markdown fences."

TOC_PAGE_SYSTEM = f"""{_ROLE}

You receive one page of a PDF document (plain text), its page number,
and the total number of pages in the document.

=== STEP 1 -- Detect TOC status ===

{_TOC_DEFINITION}

Status values:
  toc_start     -- This is the FIRST page of the TOC.
  toc_continues -- The TOC started on a previous page and continues here.
  no_toc        -- This page has no TOC content.

CRITICAL -- How to distinguish toc_start from toc_continues:

  A REAL toc_start page satisfies AT LEAST ONE of:
    - Has an explicit title: "Table of Contents", "Sommaire", "Table des matieres"
    - First numbered entry starts with 1  (e.g. "1.", "1.1", "Introduction")
    - First entry is a preamble item (Foreword, Preface, Abstract)

  If the first numbered entry starts with ANY number > 1
  (e.g. "5.", "7.2", "10.1") -> chapters before it are MISSING from this page
  -> status MUST be "toc_continues", never "toc_start".

IMPORTANT -- If status is "toc_continues":
  - Return "entries": [] immediately.
  - Do NOT extract any entries.
  - We will re-process this page later with full context.

=== STEP 2 -- Extract entries (only if status != no_toc) ===

Each entry must have:
  - "title": section title (string)
  - "page":  referenced page number as integer (null if absent)
  - "level": depth starting at 1  (1 = chapter, 2 = section, 3 = subsection ...)

{_ENTRY_RULES_FRESH}

=== STEP 3 -- Report last referenced page ===

  - Set "last_page_referenced" to the highest page number found in the entries.
  - Set it to null if no page numbers are visible in the entries.

=== OUTPUT ===

{_OUTPUT_INSTRUCTION}

{{
  "status":               "toc_start" | "toc_continues" | "no_toc",
  "confidence":           "high" | "medium" | "low",
  "last_page_referenced": <int or null>,
  "reason":               "<one sentence>",
  "entries": [
    {{ "title": "Introduction",            "page": 1,  "level": 1 }},
    {{ "title": "Chapter 1 -- Background",  "page": 5,  "level": 1 }},
    {{ "title": "1.1 Context",             "page": 5,  "level": 2 }}
  ]
}}
"""

TOC_PAGE_USER = """Page {page_number} of {total_pages}:

{page_text}

Detect the TOC status and extract all TOC entries found on this page.
Remember: every visible TOC line must appear as a separate entry -- do not skip any."""

TOC_PAGE_CONTINUE_SYSTEM = f"""{_ROLE}

You receive:
  1. The last few entries already extracted from previous page(s).
  2. A new page of the PDF document (plain text), its page number,
     and the total number of pages in the document.

=== STEP 1 -- Detect TOC status ===

Status values:
  toc_continues -- The TOC continues on this page (more entries found).
  no_toc        -- This page has no TOC content (TOC ended on a previous page).

=== STEP 2 -- Extract NEW entries only (only if status = toc_continues) ===

{_ENTRY_RULES_CONTINUE}

Each entry must have:
  - "title": section title (string)
  - "page":  referenced page number as integer (null if absent)
  - "level": depth starting at 1  (1 = chapter, 2 = section, 3 = subsection ...)

=== STEP 3 -- Report last referenced page ===

  - Set "last_page_referenced" to the highest page number found in the NEW entries.
  - Set it to null if no page numbers are visible in the new entries.
  - If status is no_toc, set "last_page_referenced" to null.

=== OUTPUT ===

{_OUTPUT_INSTRUCTION}
"""

TOC_PAGE_USER_CONTINUE = """Here are the last {context_count} entries extracted so far \
(in document order -- the TOC continues after these):

{last_entries}

Now process this new page.

Page {page_number} of {total_pages}:

{page_text}

Detect whether the TOC continues on this page and extract ONLY the NEW entries \
that come after the last entry above.
Remember: every visible TOC line on this page must appear as a separate entry -- do not skip any."""


ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "level": {"type": "integer"},
        "page": {"type": "integer"},
    },
    "required": ["title", "level", "page"],
}

FRESH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["toc_start", "toc_continues", "no_toc"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "last_page_referenced": {"type": "integer"},
        "reason": {"type": "string"},
        "entries": {"type": "array", "items": ENTRY_SCHEMA},
    },
    "required": ["status", "confidence", "last_page_referenced", "reason", "entries"],
}

CONTINUE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["toc_continues", "no_toc"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "last_page_referenced": {"type": "integer"},
        "reason": {"type": "string"},
        "entries": {"type": "array", "items": ENTRY_SCHEMA},
    },
    "required": ["status", "confidence", "last_page_referenced", "reason", "entries"],
}
