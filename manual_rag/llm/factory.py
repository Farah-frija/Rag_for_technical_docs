"""
llm/factory.py
---------------
`get_llm_client(settings)` is the ONLY place in the codebase that knows
which concrete `LLMClient` subclass corresponds to a provider name. Every
consumer (indexing pipeline, retrievers) calls this and works against the
`LLMClient` interface only.

To add a new provider (e.g. OpenAI, Anthropic, vLLM): write a class in this
package implementing `LLMClient` (or subclassing `BaseLLMClient`), add one
`elif` branch here, add its settings fields to `config.LLMSettings`. Nothing
else in the codebase changes.
"""

from __future__ import annotations

from manual_rag.config import LLMSettings
from manual_rag.core.interfaces import LLMClient

_client_singleton: dict[str, LLMClient] = {}


def get_llm_client(settings: LLMSettings) -> LLMClient:
    """Return a process-wide singleton LLMClient for the given settings'
    provider (one singleton per provider, so switching providers mid-process
    -- e.g. in tests -- still works)."""
    cache_key = f"{settings.provider}:{settings.gemini_model}:{settings.ollama_model}"
    if cache_key in _client_singleton:
        return _client_singleton[cache_key]

    if settings.provider == "gemini":
        from manual_rag.llm.gemini_client import GeminiClient
        client: LLMClient = GeminiClient(settings)
    elif settings.provider == "ollama":
        from manual_rag.llm.ollama_client import OllamaClient
        client = OllamaClient(settings)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.provider!r}")

    _client_singleton[cache_key] = client
    return client
