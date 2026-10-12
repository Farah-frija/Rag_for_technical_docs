"""
prompts/section_analysis.py
------------------------------
Prompts for the bottom-up section-summarisation pass (leaf sections get a
summary + entities + image/table descriptions; parent sections merge their
children's summaries/entities). Consumed by `manual_rag.indexing.summarizer`.
"""

from __future__ import annotations

from typing import List

LEAF_SECTION_ANALYSIS_PROMPT = """\
You are an expert AI assistant analyzing a section of a technical document.
You will receive the section title, its path in the document, and all its \
content items (text blocks, tables, images) with their IDs.

Your task is to analyze the content and return the structured response.

SECTION INFORMATION
Title : {title}
Path  : {path}

CONTENT ITEMS:
{items}

YOUR 4 TASKS

1. "summary" (string)
   A concise 2-4 sentence summary of the ENTIRE section.
   - Capture the main concept, methodology, or findings.
   - Incorporate what images and tables contribute using their captions
     or surrounding text context.
   - Must be self-contained and distinguishable from other sections.
   - Do NOT start with "This section is about..." or "The section covers...".
   - Start directly with the core insight.

2. "entities" (list of strings)
   Key technical entities found across ALL content items.
   - Include: component names, parameter names, values with units, acronyms,
     product/model names, error codes, protocol names, thresholds, identifiers.
   - Exclude: generic words ("section", "value", "data", "system", "figure").
   - No duplicates. Preserve exact casing as found in the text.

3. "image_descriptions" (list of objects)
   For every IMAGE item: write a textual description inferred from the
   surrounding text context (text items before and after the image).
   If surrounding text gives no clues, write: "No description available."
   Each object: {{"source_id": "...", "description": "..."}}

4. "table_descriptions" (list of objects)
   For every TABLE item: write a one-sentence description of what the table
   shows, what it measures or compares, and any notable values.
   Each object: {{"source_id": "...", "description": "..."}}

Return ONLY valid JSON matching the schema you were given -- no prose, no markdown fences.
"""

PARENT_SECTION_SUMMARY_PROMPT = """\
You are an expert in generating summaries for sections of a technical document.
You are given:
  - The section's own summary (generated from its direct content, may be empty).
  - The section's own entities (may be empty).
  - The summaries and entities of its direct subsections, each identified by its full path.

Your tasks:
1. Generate a single cohesive 2-4 sentence summary that synthesises both the
   section's own content and its subsections. If the own summary is empty,
   synthesise only from the subsections.
2. Merge all entities (own + all subsections) into a single deduplicated list.
   Keep exact casing. Exclude generic words.

Section path : {path}
Own summary  : {own_summary}
Own entities : {own_entities}

Subsections:
{structure}

Return ONLY valid JSON matching the schema you were given.
"""

LEAF_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "image_descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}, "description": {"type": "string"}},
                "required": ["source_id", "description"],
            },
        },
        "table_descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}, "description": {"type": "string"}},
                "required": ["source_id", "description"],
            },
        },
    },
    "required": ["summary", "entities", "image_descriptions", "table_descriptions"],
}

PARENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "entities"],
}


def format_items(nodes: List[dict]) -> str:
    """Formats leaf nodes into the {items} block for the prompt."""
    lines = []
    for i, node in enumerate(nodes):
        ntype = node["type"].upper()
        content = node["content"].strip()
        content_display = (
            f"[IMAGE FILE: {content}]" if ntype == "IMAGE"
            else (content[:1500] + "\n... [truncated]" if len(content) > 1500 else content)
        )
        lines.append(
            f"[Item {i + 1}]\n"
            f"  ID     : {node['id']}\n"
            f"  Type   : {ntype}\n"
            f"  Page   : {node.get('page', '?')}\n"
            f"  Content:\n{content_display}\n"
        )
    return "\n".join(lines)


def build_structure_block(children: List[dict]) -> str:
    lines = []
    for c in children:
        entities_str = ", ".join(c["entities"]) if c["entities"] else "none"
        lines.append(f"- [{c['path']}]\n  Summary  : {c['summary']}\n  Entities : {entities_str}")
    return "\n".join(lines)
