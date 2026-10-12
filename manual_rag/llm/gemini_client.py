"""
llm/gemini_client.py
----------------------
Google Gemini implementation of `LLMClient`, via `google-genai`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from manual_rag.config import LLMSettings
from manual_rag.llm.base import BaseLLMClient
load_dotenv()


class GeminiClient(BaseLLMClient):
    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings.min_call_interval_sec, settings.max_retries)
        from google import genai  # local import: keeps `google-genai` optional

        api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Set it in .env or export it, or "
                "switch MANUAL_RAG_LLM_PROVIDER=ollama."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = settings.gemini_model

    @property
    def model_name(self) -> str:
        return self._model
    def get_context_budget(self) -> int:
        model_info = self._client.models.get(model=self._model)
        input_token_limit = model_info.input_token_limit
        # rough chars-per-token for multilingual content
        return input_token_limit * 3
    def _call_provider(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str],
        temperature: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Optional[int]]]:
        from google.genai import types

        config_kwargs: Dict[str, Any] = dict(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        result = json.loads(response.text)

        usage = getattr(response, "usage_metadata", None)
        tokens = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        }
        return result, tokens
