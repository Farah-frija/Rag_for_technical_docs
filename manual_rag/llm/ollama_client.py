"""
llm/ollama_client.py
----------------------
Local-model implementation of `LLMClient` via Ollama's REST API
(`/api/chat` with `format=<json schema>`), so the whole pipeline can run
against e.g. `llama3.1` or `qwen2.5` with zero cloud dependency.

Ollama's structured-output support (`format` as a JSON schema) requires a
reasonably recent Ollama version; if the installed version only supports
`format: "json"` (no schema enforcement), we still request JSON and rely on
`json.loads` -- prompts already instruct the model to return the exact
schema shape, matching how the original codebase always requested schema'd
JSON but for models that need it spelled out in the prompt too.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import requests

from manual_rag.config import LLMSettings
from manual_rag.llm.base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings.min_call_interval_sec, settings.max_retries)
        self._host = settings.ollama_host.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.request_timeout_sec

    @property
    def model_name(self) -> str:
        return self._model

    def _call_provider(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str],
        temperature: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Optional[int]]]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "{}")
        result = json.loads(content)

        tokens = {
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "total_tokens": (
                (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
                if data.get("prompt_eval_count") is not None or data.get("eval_count") is not None
                else None
            ),
        }
        return result, tokens
    def get_context_budget(self) -> int:
        resp = requests.get(f"{self._host}/api/show", json={"model": self._model}, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        # model_info.parameters contains context_length for most models
        context_length = (
            data.get("model_info", {}).get("llama.context_length")
            or data.get("parameters", {}).get("num_ctx")
            or 8192  # safe fallback
        )
        return context_length * 3