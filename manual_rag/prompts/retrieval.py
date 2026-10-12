"""
prompts/retrieval.py
-----------------------
Prompts for the tree-aware retriever (`manual_rag.retrieval.tree_retriever`):
query entity extraction, TOC-guided routing/decomposition, per-sub-query
answer synthesis, and final multi-sub-query synthesis.
"""

from __future__ import annotations

QUERY_ENTITY_PROMPT = """\
You are a technical document query analyser.
Extract all key technical entities from the user query below.

Include: component names, parameter names, values with units, acronyms,
product/model names, error codes, protocol names, identifiers.
Exclude: generic words ("section", "value", "data", "system", "what", "how").

USER QUERY:
{query}

Return ONLY a JSON object:
{{
  "entities": ["entity_1", "entity_2", ...]
}}
"""

QUERY_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
}

TOC_ROUTER_PROMPT = """\
You are an expert technical document routing assistant.
You receive a filtered table of contents -- only sections with actual content
that passed a minimum relevance threshold against the user query.

Each content-bearing entry has a one-sentence summary and a pre-computed
embedding similarity score. Structural parent entries (no summary, no score)
are shown only as path context.

YOUR TASKS:
  1. Decompose the user query into n focused sub-queries (as many as needed to
     cover every distinct information need; minimum 1):
       - Write each sub-query as a self-contained question, not a keyword phrase.
         Repeat the entity names; never rely on the original query for context.

  2. The "embed_score" [0.0-1.0] measures semantic similarity between the query
     entities and the section's entities. Use it as a strong signal but also
     reason over the summaries to catch cases the embedding may miss.
     For each sub-query, select up to {max_k} candidate sections most likely to
     contain the answer, ordered by descending relevance, each with an llm_score.

USER QUERY:
{query}

DOCUMENT TABLE OF CONTENTS (filtered):
{toc}

Return ONLY a valid JSON object:
{{
  "subqueries": [
    {{
      "subquery": "focused sub-question text",
      "candidates": [
        {{"node_id": "...", "llm_score": 0.95}}
      ]
    }}
  ]
}}
Each candidates list has at most {max_k} entries, ordered by descending llm_score.
"""

TOC_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "subqueries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subquery": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "llm_score": {"type": "number"},
                            },
                            "required": ["node_id", "llm_score"],
                        },
                    },
                },
                "required": ["subquery", "candidates"],
            },
        },
    },
    "required": ["subqueries"],
}
TOC_ROUTER_NOSPLIT_PROMPT = """\
You are an expert technical document routing assistant.
You receive a filtered table of contents -- only sections with actual content
that passed a minimum relevance threshold against the user query.

Each content-bearing entry has a one-sentence summary and a pre-computed
embedding similarity score. Structural parent entries (no summary, no score)
are shown only as path context.

YOUR TASK:
  Treat the user query as a single, indivisible question -- do NOT decompose
  it into sub-questions. The "embed_score" [0.0-1.0] measures semantic
  similarity between the query entities and the section's entities. Use it as
  a strong signal but also reason over the summaries to catch cases the
  embedding may miss. Select up to {max_k} candidate sections most likely to
  contain the answer, ordered by descending relevance, each with an llm_score.

USER QUERY:
{query}

DOCUMENT TABLE OF CONTENTS (filtered):
{toc}

Return ONLY a valid JSON object:
{{
  "candidates": [
    {{"node_id": "...", "llm_score": 0.95}}
  ]
}}
The candidates list has at most {max_k} entries, ordered by descending llm_score.
"""

TOC_ROUTER_NOSPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "llm_score": {"type": "number"},
                },
                "required": ["node_id", "llm_score"],
            },
        },
    },
    "required": ["candidates"],
}
SUBQUERY_ANSWER_PROMPT = """\
You are an expert assistant answering questions about a technical document.
Use ONLY the provided context to answer. Each context block is preceded by its Section ID.
Decompose your answer into distinct fragments, and map each fragment directly to the Section ID it is derived from.
If the context does not contain enough information, state so explicitly.

SUB-QUERY:
{subquery}

RETRIEVED CONTEXT (with Section IDs):
{context}

Return ONLY a valid JSON object:
{{
  "fragments": [
    {{"answer_fragment": "clear and precise answer text segment", "section_id": "corresponding section ID from the context"}}
  ],
  "confidence": "high | medium | low | not_found"
}}
"""

SUBQUERY_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "answer_fragment": {"type": "string"},
                    "section_id": {"type": "string"},
                },
                "required": ["answer_fragment", "section_id"],
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "not_found"]},
    },
    "required": ["fragments", "confidence"],
}

SYNTHESIS_PROMPT = """\
You are an expert assistant synthesising answers to a multi-part technical query.
You receive the original query and a set of partial answers, one per sub-query.
Combine them into a single coherent, non-redundant answer.
If sub-answers conflict, flag the conflict explicitly.
If NONE of the partial answers contain information relevant to the query
(i.e. every partial answer is empty, off-topic, or explicitly states the
information was not found), you MUST set "answer" to the literal string
"Unknown" and "confidence" to "not_found". Do not guess or fabricate content
that is not grounded in the partial answers.

ORIGINAL QUERY:
{query}

PARTIAL ANSWERS:
{partial_answers}

Return ONLY a valid JSON object:
{{
  "answer": "final synthesised answer",
  "confidence": "high | medium | low | not_found"
}}
"""

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "not_found"]},
    },
    "required": ["answer", "confidence"],
}

SUBQUERY_SPLIT_PROMPT = """\
You are analyzing a user query against a document's table of contents.

TABLE OF CONTENTS:
{toc}

USER QUERY:
{query}

Your job is to decide whether answering this query requires retrieving content \
from multiple distinct sections of the TOC -- meaning a single retrieval path \
cannot cover everything the query asks.

Split the query into sub-queries if the answer requires visiting more than one \
section, whether those sections are siblings, cousins, or from completely \
different branches.

Examples of when to SPLIT:
  Query : "What are the input voltage range and the current consumption?"
  TOC   : Chapter 3 Electrical Specifications
            3.1 Voltage Input
            3.2 Current Consumption
  → Needs both 3.1 and 3.2, two different sections → split into:
      - "What is the input voltage range of the device?"
      - "What is the current consumption of the device?"

  Query : "What are the safety precautions and how do I configure the network interface?"
  TOC   : Chapter 2 Safety / Chapter 5 Network Configuration
  → Needs Chapter 2 and Chapter 5 → split into:
      - "What are the safety precautions?"
      - "How do I configure the network interface?"

Examples of when NOT to split:
  Query : "What is the operating temperature range?"
  TOC   : Chapter 4 Environmental Specifications / 4.1 Temperature
  → Everything needed is in one section → no split needed.

Rules:
- Each sub-query must be fully self-contained and understandable on its own, \
preserving enough context from the original query so it can be answered \
independently (do not write "the first part" or refer to other sub-queries).
- Do not invent topics not present in the original query.
- Prefer NOT splitting: only split when it is clear the query spans multiple \
distinct sections that each need to be retrieved separately.
- If no split is needed, return the original query as a single-element list.
- Do not add the path to the query.
"""

SUBQUERY_SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "subqueries": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of self-contained sub-queries. Single-element list when "
                "no split is needed."
            ),
        }
    },
    "required": ["subqueries"],
}

RESPONDER_PROMPT = """\
You are a precise technical assistant. Answer the question using ONLY the \
context below. If the context does not contain enough information, say so.

QUESTION:
{query}

CONTEXT:
Each passage starts with a header: [chunk_id=<id> | Page <n> | Section: <name>].

{context}

INSTRUCTIONS:
- Return a concise, grounded answer. Do not add information not present in the context.
- After every claim, append the chunk_id of the passage that supports it, \
e.g. "The torque is 25 Nm [chunk_id=abc123]".
- In `cited_chunks`, list ONLY the chunk_ids you actually used, copied exactly \
from the headers. Do not list passages you read but did not use.
- If the context is insufficient, say so and return an empty `cited_chunks` list.
"""

RESPONDER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["not_found", "low", "medium", "high"]},
        "cited_chunks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence", "cited_chunks"],
}