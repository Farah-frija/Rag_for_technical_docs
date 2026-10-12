"""
indexing/toc_graph.py
------------------------
LangGraph state machine driving page-by-page TOC detection/extraction.
Ported from the original `indexing/client_llm.py`, with two changes:

  1. All Gemini-specific plumbing (schema objects, client construction,
     the bespoke `_io_logged` file-dumping decorator) is gone -- probing now
     calls `LLMClient.generate_structured(...)` with prompts from
     `manual_rag.prompts.toc_extraction`, so this module works with any
     configured provider (Gemini, Ollama, ...) and every call is logged
     uniformly via `manual_rag.logging_config`.
  2. Regex-based post-processing (numeric-first-entry guard) stays --
     that's genuine domain logic, not provider plumbing.

Probe order:
  1. first_page  — forced by toc_extractor.select_toc_start_candidate()
  2. ranked_pages — remaining pages in score order, skipping first_page
                    and any page already seen during backtrack/forward scan

Backtrack + buffer workflow:
  When a page is judged toc_continues during fresh probing, we backtrack.
  Every page seen as toc_continues is pushed into `pending_pages` (ordered,
  oldest first). Once toc_start is found, forward scan drains pending_pages
  in order using probe_and_extract_continue, then continues normally.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from manual_rag.core.interfaces import LLMClient
from manual_rag.logging_config import get_logger
from manual_rag.prompts.toc_extraction import (
    CONTINUE_RESPONSE_SCHEMA,
    FRESH_RESPONSE_SCHEMA,
    TOC_PAGE_CONTINUE_SYSTEM,
    TOC_PAGE_SYSTEM,
    TOC_PAGE_USER,
    TOC_PAGE_USER_CONTINUE,
)
from manual_rag.utils.tracing import TraceRecorder

logger = get_logger(__name__)


# ── has_more decision ────────────────────────────────────────────────────────

def decide_has_more(last_page_referenced: Optional[int], total_pages: int) -> str:
    if last_page_referenced is None:
        return "unknown"
    if total_pages > last_page_referenced:
        return "yes"
    if total_pages == last_page_referenced:
        return "no"
    return "unknown"


def _no_toc_default() -> dict:
    return {
        "status": "no_toc", "confidence": "low",
        "last_page_referenced": None, "has_more": "no",
        "reason": "call failed -- treated as no_toc", "entries": [],
    }


def _flatten_tree(entries: List[dict]) -> List[dict]:
    flat: List[dict] = []
    for entry in entries:
        flat.append({"title": entry.get("title", ""), "level": entry.get("level", 1), "page": entry.get("page")})
        flat.extend(_flatten_tree(entry.get("children", [])))
    return flat


def _build_tree_from_flat(entries: List[dict]) -> List[dict]:
    root: List[dict] = []
    stack: List[tuple] = []
    for e in entries:
        node = {**e, "children": []}
        level = e.get("level", 1)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            root.append(node)
        stack.append((level, node))
    return root


def _merge_fresh(base: Optional[dict], new_entries: List[dict]) -> dict:
    existing = list((base or {}).get("entries", []))
    existing.extend(new_entries)
    return {"entries": existing}


# ── LLM probe functions ───────────────────────────────────────────────────────

def probe_and_extract_fresh(
    llm: LLMClient, page_text: str, page_number: int, total_pages: int, trace: Optional[TraceRecorder] = None,
) -> dict:
    """Fresh probe -- used during backtracking and initial candidate scanning.
    If status == toc_continues, entries are forced to [] (no context available)."""
    user = TOC_PAGE_USER.format(page_number=page_number, total_pages=total_pages, page_text=page_text)
    try:
        result = llm.generate_structured(
            user, FRESH_RESPONSE_SCHEMA, label=f"TOC_FRESH_p{page_number}",
            system_instruction=TOC_PAGE_SYSTEM, trace=trace,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"toc fresh probe failed on page {page_number}: {exc}")
        result = _no_toc_default()

    if result.get("status") == "toc_continues":
        result["entries"] = []

    # Guard: toc_start but first entry number > 1 -> override to toc_continues
    if result.get("status") == "toc_start" and result.get("entries"):
        first_title = result["entries"][0].get("title", "")
        m = re.match(r"^(\d+)", first_title)
        if m and int(m.group(1)) > 1:
            logger.info(f"page {page_number}: overriding toc_start -> toc_continues (first entry '{first_title}')")
            result["status"] = "toc_continues"
            result["entries"] = []

    if "has_more" not in result:
        last_ref = result.get("last_page_referenced") or None
        result["has_more"] = decide_has_more(last_ref if last_ref != 0 else None, total_pages)

    if result.get("entries"):
        result["entries"] = _build_tree_from_flat(result["entries"])

    return result


def probe_and_extract_continue(
    llm: LLMClient, page_text: str, page_number: int, total_pages: int,
    toc_tree: dict, context_size: int = 5, trace: Optional[TraceRecorder] = None,
) -> dict:
    """Continuation probe -- used during forward scan. Sends the last N flat
    entries as context so the LLM knows the current hierarchy."""
    flat_all = _flatten_tree(toc_tree.get("entries", []))
    last_entries = flat_all[-context_size:] if flat_all else []

    user = TOC_PAGE_USER_CONTINUE.format(
        context_count=len(last_entries),
        last_entries=json.dumps(last_entries, ensure_ascii=False, indent=2),
        page_number=page_number, total_pages=total_pages, page_text=page_text,
    )
    try:
        result = llm.generate_structured(
            user, CONTINUE_RESPONSE_SCHEMA, label=f"TOC_CONTINUE_p{page_number}",
            system_instruction=TOC_PAGE_CONTINUE_SYSTEM, trace=trace,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"toc continue probe failed on page {page_number}: {exc}")
        result = _no_toc_default()

    if "has_more" not in result:
        last_ref = result.get("last_page_referenced") or None
        result["has_more"] = decide_has_more(last_ref if last_ref != 0 else None, total_pages)

    if result.get("entries"):
        result["entries"] = _build_tree_from_flat(result["entries"])

    return result


# ── LangGraph state ──────────────────────────────────────────────────────────

class TOCState(TypedDict):
    all_pages: List[str]
    ranked_pages: List[dict]
    total_pages: int
    first_page_idx: int
    state: str
    ranked_cursor: int
    doc_cursor: int
    seen_pages: List[int]
    toc_page_indices: List[int]
    toc_tree: Optional[dict]
    found: bool
    pending_pages: List[int]
    messages: Annotated[list, add_messages]


def route(state: TOCState) -> str:
    s = state["state"]
    return (
        "probe_first" if s == "probing_first" else
        "probe_candidate" if s == "not_started" else
        "backtrack" if s == "backtracking" else
        "forward_scan" if s == "in_progress" else
        "exhausted" if s == "exhausted" else
        END
    )


def build_toc_graph(llm: LLMClient, trace: Optional[TraceRecorder] = None):
    """Build the compiled LangGraph. `llm`/`trace` are closed over by the
    node functions so nodes stay plain `state -> state` callables (what
    LangGraph expects) while still using the injected, pluggable LLM client."""

    def node_probe_first(state: TOCState) -> TOCState:
        doc_idx = state["first_page_idx"]
        seen = set(state["seen_pages"]) | {doc_idx}
        result = probe_and_extract_fresh(
            llm, state["all_pages"][doc_idx], doc_idx + 1, state["total_pages"], trace
        )
        status = result["status"]
        if status == "toc_start":
            return {**state, "state": "in_progress", "doc_cursor": doc_idx + 1,
                    "toc_page_indices": [doc_idx], "toc_tree": _merge_fresh(None, result["entries"]),
                    "pending_pages": [], "seen_pages": sorted(seen)}
        if status == "toc_continues":
            return {**state, "state": "backtracking", "doc_cursor": doc_idx - 1,
                    "pending_pages": [doc_idx], "seen_pages": sorted(seen)}
        return {**state, "state": "not_started", "seen_pages": sorted(seen)}

    def node_probe_candidate(state: TOCState) -> TOCState:
        cursor, ranked, seen = state["ranked_cursor"], state["ranked_pages"], set(state["seen_pages"])
        while cursor < len(ranked) and ranked[cursor]["original_idx"] in seen:
            cursor += 1
        if cursor >= len(ranked):
            return {**state, "state": "exhausted", "ranked_cursor": cursor}

        doc_idx = ranked[cursor]["original_idx"]
        seen = seen | {doc_idx}
        result = probe_and_extract_fresh(llm, state["all_pages"][doc_idx], doc_idx + 1, state["total_pages"], trace)
        status = result["status"]

        if status == "toc_start":
            return {**state, "state": "in_progress", "doc_cursor": doc_idx + 1,
                    "toc_page_indices": [doc_idx], "toc_tree": _merge_fresh(None, result["entries"]),
                    "ranked_cursor": cursor + 1, "pending_pages": [], "seen_pages": sorted(seen)}
        if status == "toc_continues":
            return {**state, "state": "backtracking", "doc_cursor": doc_idx - 1,
                    "ranked_cursor": cursor + 1,
                    "pending_pages": sorted(set(state.get("pending_pages", [])) | {doc_idx}),
                    "seen_pages": sorted(seen)}
        return {**state, "ranked_cursor": cursor + 1, "seen_pages": sorted(seen)}

    def node_backtrack(state: TOCState) -> TOCState:
        doc_idx = state["doc_cursor"]
        seen = set(state["seen_pages"])
        pending = list(state.get("pending_pages", []))

        if doc_idx < 0:
            if 0 not in seen:
                r = probe_and_extract_fresh(llm, state["all_pages"][0], 1, state["total_pages"], trace)
                entries, seen = r["entries"], seen | {0}
            else:
                entries = []
            first_pending = pending[0] if pending else 1
            return {**state, "state": "in_progress", "doc_cursor": first_pending,
                    "toc_page_indices": [0], "toc_tree": _merge_fresh(None, entries),
                    "pending_pages": pending, "seen_pages": sorted(seen)}

        if doc_idx in seen:
            toc_indices_set = set(state["toc_page_indices"])
            if doc_idx in toc_indices_set:
                first_pending = pending[0] if pending else doc_idx + 1
                return {**state, "state": "in_progress", "doc_cursor": first_pending}
            real_start = doc_idx + 1
            first_pending = pending[0] if pending else real_start
            if real_start not in seen:
                seen = seen | {real_start}
                r2 = probe_and_extract_fresh(llm, state["all_pages"][real_start], real_start + 1, state["total_pages"], trace)
                return {**state, "state": "in_progress", "doc_cursor": first_pending,
                        "toc_page_indices": sorted({real_start, *toc_indices_set}),
                        "toc_tree": _merge_fresh(None, r2["entries"]),
                        "pending_pages": pending, "seen_pages": sorted(seen)}
            return {**state, "state": "in_progress", "doc_cursor": first_pending, "seen_pages": sorted(seen)}

        seen = seen | {doc_idx}
        result = probe_and_extract_fresh(llm, state["all_pages"][doc_idx], doc_idx + 1, state["total_pages"], trace)
        status = result["status"]
        toc_indices_set = set(state["toc_page_indices"])

        if status == "toc_start":
            first_pending = pending[0] if pending else doc_idx + 1
            return {**state, "state": "in_progress", "doc_cursor": first_pending,
                    "toc_page_indices": sorted({doc_idx, *toc_indices_set}),
                    "toc_tree": _merge_fresh(None, result["entries"]),
                    "pending_pages": pending, "seen_pages": sorted(seen)}
        if status == "toc_continues":
            return {**state, "doc_cursor": doc_idx - 1,
                    "pending_pages": sorted(set(pending) | {doc_idx}), "seen_pages": sorted(seen)}

        real_start = doc_idx + 1
        first_pending = pending[0] if pending else real_start
        if real_start not in seen:
            seen = seen | {real_start}
            r2 = probe_and_extract_fresh(llm, state["all_pages"][real_start], real_start + 1, state["total_pages"], trace)
            return {**state, "state": "in_progress", "doc_cursor": first_pending,
                    "toc_page_indices": sorted({real_start, *toc_indices_set}),
                    "toc_tree": _merge_fresh(None, r2["entries"]),
                    "pending_pages": pending, "seen_pages": sorted(seen)}
        return {**state, "state": "in_progress", "doc_cursor": first_pending, "seen_pages": sorted(seen)}

    def node_forward_scan(state: TOCState) -> TOCState:
        doc_idx = state["doc_cursor"]
        seen = set(state["seen_pages"])
        toc_indices_set = set(state["toc_page_indices"])
        pending = list(state.get("pending_pages", []))

        if doc_idx >= state["total_pages"]:
            return {**state, "state": "completed", "found": True}

        if doc_idx in pending:
            result = probe_and_extract_continue(
                llm, state["all_pages"][doc_idx], doc_idx + 1, state["total_pages"],
                state["toc_tree"] or {"entries": []}, trace=trace,
            )
            has_more = result["has_more"]
            updated_tree = _merge_fresh(state["toc_tree"], result.get("entries", []))
            updated_idx = sorted({*toc_indices_set, doc_idx})
            new_pending = [p for p in pending if p != doc_idx]
            seen = seen | {doc_idx}
            next_cursor = new_pending[0] if new_pending else doc_idx + 1
            base = {**state, "toc_page_indices": updated_idx, "toc_tree": updated_tree,
                    "doc_cursor": next_cursor, "pending_pages": new_pending, "seen_pages": sorted(seen)}
            if not new_pending and has_more == "no":
                return {**base, "state": "completed", "found": True}
            return {**base, "state": "in_progress"}

        if doc_idx in seen:
            if doc_idx in toc_indices_set:
                return {**state, "doc_cursor": doc_idx + 1, "state": "in_progress"}
            return {**state, "state": "completed", "found": True}

        seen = seen | {doc_idx}
        result = probe_and_extract_continue(
            llm, state["all_pages"][doc_idx], doc_idx + 1, state["total_pages"],
            state["toc_tree"] or {"entries": []}, trace=trace,
        )
        status, has_more = result["status"], result["has_more"]
        if status == "no_toc":
            return {**state, "state": "completed", "found": True, "seen_pages": sorted(seen)}

        updated_tree = _merge_fresh(state["toc_tree"], result.get("entries", []))
        updated_idx = sorted({*toc_indices_set, doc_idx})
        base = {**state, "toc_page_indices": updated_idx, "toc_tree": updated_tree,
                "doc_cursor": doc_idx + 1, "seen_pages": sorted(seen)}
        return {**base, "state": "completed", "found": True} if has_more == "no" else {**base, "state": "in_progress"}

    def node_exhausted(state: TOCState) -> TOCState:
        return {**state, "found": False}

    g = StateGraph(TOCState)
    g.add_node("probe_first", node_probe_first)
    g.add_node("probe_candidate", node_probe_candidate)
    g.add_node("backtrack", node_backtrack)
    g.add_node("forward_scan", node_forward_scan)
    g.add_node("exhausted", node_exhausted)
    g.set_entry_point("probe_first")

    g.add_conditional_edges("probe_first", route, {
        "probe_candidate": "probe_candidate", "backtrack": "backtrack",
        "forward_scan": "forward_scan", END: END,
    })
    g.add_conditional_edges("probe_candidate", route, {
        "probe_candidate": "probe_candidate", "backtrack": "backtrack",
        "forward_scan": "forward_scan", "exhausted": "exhausted", END: END,
    })
    g.add_conditional_edges("backtrack", route, {
        "forward_scan": "forward_scan", "backtrack": "backtrack", END: END,
    })
    g.add_conditional_edges("forward_scan", route, {"forward_scan": "forward_scan", END: END})
    g.add_edge("exhausted", END)

    return g.compile(checkpointer=MemorySaver())


def run_toc_extraction(
    llm: LLMClient,
    all_pages: List[str],
    ranked_pages: List[dict],
    first_page: dict,
    trace: Optional[TraceRecorder] = None,
) -> Dict[str, Any]:
    graph = build_toc_graph(llm, trace)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial: TOCState = {
        "all_pages": all_pages, "ranked_pages": ranked_pages, "total_pages": len(all_pages),
        "first_page_idx": first_page["original_idx"], "state": "probing_first",
        "ranked_cursor": 0, "doc_cursor": 0, "seen_pages": [], "toc_page_indices": [],
        "toc_tree": None, "found": False, "pending_pages": [], "messages": [],
    }

    logger.info(f"TOC extraction start: {len(all_pages)} pages, first_page={first_page.get('original_idx')}")
    final = graph.invoke(initial, config=config)
    logger.info(f"TOC extraction done: found={final['found']} pages={final['toc_page_indices']}")

    return {
        "found": final["found"],
        "toc_page_indices": final["toc_page_indices"],
        "toc_tree": final["toc_tree"],
        "thread_id": thread_id,
    }
