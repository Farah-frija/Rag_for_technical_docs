"""
llm/base.py
------------
`BaseLLMClient` factors out everything that should NOT differ between
providers: client-side rate limiting, retry-on-failure, and structured
logging of every prompt/output/error/token-usage to the shared
`llm_calls` logger. Concrete providers (`GeminiClient`, `OllamaClient`)
only implement `_call_provider`, which does the actual network request and
returns `(parsed_dict, token_usage)`.

This is what the original codebase was missing: `Answering/retriever.py`,
`Answering/page_index_approach.py`, and `indexing/summarize_tree.py` each
hand-rolled their own near-identical `_rate_limited_gemini_call` /
`_gemini_call`. Any future provider gets rate limiting, retries, and
logging for free just by subclassing this.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from manual_rag.core.exceptions import LLMCallError
from manual_rag.core.interfaces import LLMClient
from manual_rag.logging_config import get_llm_call_logger, get_logger
from manual_rag.utils.rate_limiter import RateLimiter
from manual_rag.utils.tracing import TraceRecorder

logger = get_logger(__name__)


class BaseLLMClient(LLMClient):
    def __init__(self, min_call_interval_sec: float, max_retries: int = 2) -> None:
        self._rate_limiter = RateLimiter(min_call_interval_sec)
        self._max_retries = max_retries
        self._call_logger = get_llm_call_logger()
    @abstractmethod
    def get_context_budget(self) -> int:
        """Character budget available for context in a single LLM call."""
        raise NotImplementedError
    @abstractmethod
    def _call_provider(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str],
        temperature: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Optional[int]]]:
        """Do the actual provider call. Return (parsed_json, token_usage) where
        token_usage has keys prompt_tokens/output_tokens/total_tokens (any
        may be None if the provider doesn't report usage, e.g. local Ollama)."""
        raise NotImplementedError

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        label: str = "unlabeled",
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        trace: Optional[TraceRecorder] = None,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            self._rate_limiter.wait()
            t0 = time.perf_counter()
            try:
                result, tokens = self._call_provider(prompt, schema, system_instruction, temperature)
                elapsed = time.perf_counter() - t0
                self._call_logger.debug(
                    f"{label} ok",
                    extra={
                        "label": label, "model": self.model_name, "prompt": prompt,
                        "output": result, "tokens": tokens, "elapsed_sec": round(elapsed, 3),
                        "attempt": attempt,
                    },
                )
                if trace is not None:
                    trace.record_llm_call(label, prompt, output=result, elapsed_sec=elapsed, tokens=tokens)
                return result
            except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry any failure
                last_error = exc
                elapsed = time.perf_counter() - t0
                self._call_logger.warning(
                    f"{label} failed (attempt {attempt + 1}/{self._max_retries + 1}): {exc}",
                    extra={
                        "label": label, "model": self.model_name, "prompt": prompt,
                        "error": str(exc), "elapsed_sec": round(elapsed, 3), "attempt": attempt,
                    },
                )
                if trace is not None:
                    trace.record_llm_call(label, prompt, error=str(exc), elapsed_sec=elapsed)

        raise LLMCallError(f"LLM call '{label}' failed after {self._max_retries + 1} attempts: {last_error}")
