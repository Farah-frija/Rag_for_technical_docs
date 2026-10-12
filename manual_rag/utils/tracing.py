"""
utils/tracing.py
-----------------
Lightweight, dependency-free observability helpers shared by indexing and
retrieval:

  * `timed()`      -- a context manager that logs START/END + elapsed time
                       for a named step. Was reimplemented ad hoc in
                       `Answering/retriever.py` (`_timed`); now shared.
  * `TraceRecorder` -- accumulates structured trace entries (one per LLM
                       call or pipeline stage) that both the CLI's
                       `--verbose` output and `eval.py`'s token-usage
                       aggregation read from -- the single source of truth
                       for "what happened during this query/index run".
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@contextmanager
def timed(label: str, logger: logging.Logger) -> Iterator[None]:
    logger.info("start", extra={"stage": label, "event": "start"})
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(
            f"{label} finished in {elapsed:.3f}s",
            extra={"stage": label, "event": "end", "elapsed_sec": round(elapsed, 4)},
        )


@dataclass
class TraceRecorder:
    """Collects trace entries for a single query or indexing run. Every LLM
    call appends one entry here via `record_llm_call`; `eval.py` and the CLI
    both read `.entries` to compute token totals / print a pipeline summary."""

    entries: List[Dict[str, Any]] = field(default_factory=list)
    stage_capture: Dict[str, Any] = field(default_factory=dict)

    def record_llm_call(
        self,
        label: str,
        prompt: str,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
        tokens: Optional[Dict[str, Optional[int]]] = None,
    ) -> None:
        entry: Dict[str, Any] = {"label": label, "prompt": prompt}
        if output is not None:
            entry["output"] = output
        if error is not None:
            entry["error"] = error
        if elapsed_sec is not None:
            entry["elapsed"] = elapsed_sec
        if tokens is not None:
            entry["tokens"] = tokens
        self.entries.append(entry)

    def set_stage(self, key: str, value: Any) -> None:
        self.stage_capture[key] = value

    def total_tokens(self) -> int:
        return sum((e.get("tokens") or {}).get("total_tokens") or 0 for e in self.entries)
