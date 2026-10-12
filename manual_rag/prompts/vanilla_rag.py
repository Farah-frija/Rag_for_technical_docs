"""
prompts/vanilla_rag.py
-------------------------
Prompts for the two simpler retrieval strategies:
  * `manual_rag.retrieval.vanilla_retriever` -- single-pass flat RAG responder.
  * `manual_rag.retrieval.page_index_retriever` -- hierarchical traversal router.
"""

from __future__ import annotations

RESPONDER_PROMPT = """\
You are an expert assistant answering questions about a technical document.
Use ONLY the provided context to answer. Each context block is preceded by
its page/chunk reference.
If the context does not contain enough information, state so explicitly.

QUERY:
{query}

RETRIEVED CONTEXT:
{context}

Return ONLY a valid JSON object:
{{
  "answer": "clear and precise answer, grounded only in the context above",
  "confidence": "high | medium | low | not_found"
}}
"""

RESPONDER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "not_found"]},
    },
    "required": ["answer", "confidence"],
}

HIERARCHICAL_ROUTER_PROMPT = """\
You are an expert technical document routing assistant.
You are currently inspecting a parent section of the document and evaluating its subsections to answer the user query.

USER QUERY:
{query}

CURRENT STATUS:
- Reasoning / Progress: {current_reasoning}

PARENT SECTION:
- Title   : {parent_title}
- Summary : {parent_summary}

CHILD SUBSECTIONS:
{children_toc}

YOUR TASKS:
1. "sufficient" (boolean): true if you have gathered enough information to fully answer the query
   (or no further subsections in this branch are needed); false if you need to explore deeper.
2. "reasoning" (string): update your reasoning -- what's been achieved, what's missing, why these children.
3. "selected_node_ids" (list of strings): child section IDs to explore/collect (empty list if none relevant).

Return ONLY a valid JSON object:
{{
  "sufficient": false,
  "reasoning": "We found X in this section, but still need Y from subsection Z.",
  "selected_node_ids": ["id_1", "id_2"]
}}
"""

HIERARCHICAL_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "selected_node_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient", "reasoning", "selected_node_ids"],
}
