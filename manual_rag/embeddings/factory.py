"""embeddings/factory.py -- selects an EmbeddingProvider by config. Add new
providers (OpenAI embeddings, Ollama embeddings, ...) by adding a branch
here plus a settings field in `config.EmbeddingSettings`."""

from __future__ import annotations

from manual_rag.config import EmbeddingSettings
from manual_rag.core.interfaces import EmbeddingProvider

_singleton: dict[str, EmbeddingProvider] = {}


def get_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    key = f"{settings.provider}:{settings.model_name}"
    if key in _singleton:
        return _singleton[key]

    if settings.provider == "sentence_transformers":
        from manual_rag.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )
        provider: EmbeddingProvider = SentenceTransformersProvider(settings)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.provider!r}")

    _singleton[key] = provider
    return provider
