"""
utils/rate_limiter.py
----------------------
A tiny, thread-safe, client-side rate limiter.

The original codebase re-implemented this exact "sleep until
min_call_interval has elapsed since the last call" pattern independently in
`Answering/retriever.py`, `Answering/page_index_approach.py`, and
`indexing/summarize_tree.py` (each with its own `_last_call_time` module
global). Centralised here so it's written, tested, and fixed once.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Blocks `wait()` until at least `min_interval_sec` has elapsed since
    the previous call to `wait()` returned. Safe to share across threads."""

    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = min_interval_sec
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Sleep if necessary, then record the new call time. Returns how
        long it slept (seconds), for logging."""
        with self._lock:
            elapsed = time.monotonic() - self._last_call_time
            sleep_for = max(0.0, self.min_interval_sec - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call_time = time.monotonic()
            return sleep_for
