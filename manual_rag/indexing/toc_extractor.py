"""
indexing/toc_extractor.py
----------------------------
Page-by-page TOC detection pipeline. Ported from the original
`indexing/toc_extractor.py`; the LLM plumbing now goes through the injected
`LLMClient` + `toc_graph.run_toc_extraction`.

Flow:
  1. Compute N = f(pdf_length) candidate pages.
  2. Score + rank the first N pages by TOC heuristics (no elimination).
  3. Select the best first page to probe via select_toc_start_candidate().
  4. Pass all_pages + ranked + first_page to the LLM state machine.
  5. Return TOCResult with found flag, page indices, raw text, toc tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from manual_rag.core.interfaces import LLMClient
from manual_rag.indexing.toc_graph import run_toc_extraction
from manual_rag.logging_config import get_logger
from manual_rag.utils.tracing import TraceRecorder

logger = get_logger(__name__)


# ── 1. Page-count heuristic ──────────────────────────────────────────────────

def compute_candidate_page_count(total_pages: int, cap: int = 10) -> int:
    """How many pages (from the start) to consider as TOC candidates. Scales
    with document length but is capped to avoid over-scanning."""
    if total_pages <= 10:
        return 3
    return cap


# ── 2. Scoring / ranking ──────────────────────────────────────────────────────

_STRONG_PATTERNS: List[tuple] = [
    (r"\b(table of contents|table des mati\u00e8res|plan)\b", 7),  # explicit TOC titles
]

_TOC_PATTERNS: List[tuple] = [
    (r"\b\d{1,4}\s*$", 3),                                # line ends with page number
    (r"^\s*\d+[\.\d]*\s+\w+", 3),                          # line starts with numbered section
    (r"(chapitre|chapter|section|partie|part)\s+\d+", 4),  # explicit keywords
    (r"\.{3,}", 3),                                        # dot leaders
    (r"^\s*[-•#*]\s+\w+", 2),                              # bullet / dash items
    (r"\S.*\s{4,}\d+", 3),                                 # wide gap then number
]


def _score_page(text: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return {"score": 0, "structured_ratio": 0.0}

    total_score = 0
    for pattern, weight in _STRONG_PATTERNS:
        matches = [l for l in lines if re.search(pattern, l, re.IGNORECASE)]
        if matches:
            total_score += weight * len(matches)

    for pattern, weight in _TOC_PATTERNS:
        matches = [l for l in lines if re.search(pattern, l, re.IGNORECASE)]
        if len(matches) >= 2:
            total_score += weight * len(matches)

    structured = sum(
        1 for l in lines if re.search(r"\b\d+\s*$", l) or re.search(r"^\s*\d+[\.\d]*", l)
    )
    ratio = structured / len(lines) if lines else 0.0
    if ratio > 0.4:
        total_score += 5

    return {"score": total_score, "structured_ratio": round(ratio, 2)}


def rank_pages(pages: List[str], offset: int = 0) -> List[dict]:
    ranked = [{"original_idx": offset + i, "text": text, **_score_page(text)} for i, text in enumerate(pages)]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ── 3. Start candidate selection ──────────────────────────────────────────────

def select_toc_start_candidate(ranked: List[dict]) -> dict:
    """From the top-3 ranked pages, pick the earliest by document order if
    they're plausibly consecutive, so we start probing from the real
    beginning of the TOC rather than the middle."""
    top3 = ranked[:3]
    return min(top3, key=lambda p: p["original_idx"])


# ── 4. Result container ───────────────────────────────────────────────────────

@dataclass
class TOCResult:
    found: bool = False
    pages: List[int] = field(default_factory=list)
    raw_text: str = ""
    toc_tree: dict = field(default_factory=dict)
    page_ranking: List[dict] = field(default_factory=list)


# ── 5. Main extractor ─────────────────────────────────────────────────────────

def extract_toc(
    llm: LLMClient,
    all_pages: List[str],
    candidate_page_cap: int = 10,
    trace: TraceRecorder | None = None,
) -> TOCResult:
    n = compute_candidate_page_count(len(all_pages), cap=candidate_page_cap)
    candidate_pages = all_pages[:n]
    ranked = rank_pages(candidate_pages, offset=0)

    result = TOCResult(page_ranking=[
        {
            "original_page": r["original_idx"] + 1,
            "original_idx": r["original_idx"],
            "score": r["score"],
            "structured_ratio": r["structured_ratio"],
        }
        for r in ranked
    ])

    first_page = select_toc_start_candidate(ranked)
    llm_result = run_toc_extraction(llm, all_pages=all_pages, ranked_pages=ranked[:3], first_page=first_page, trace=trace)

    if llm_result["found"]:
        toc_indices = llm_result["toc_page_indices"]
        result.found = True
        result.pages = toc_indices
        result.toc_tree = llm_result["toc_tree"] or {}
        result.raw_text = "\n\n--- PAGE BREAK ---\n\n".join(all_pages[i] for i in toc_indices)

    return result
