"""rerankers/factory.py -- selects a Reranker by config."""

from __future__ import annotations

from manual_rag.config import RerankerSettings
from manual_rag.core.interfaces import Reranker

_singleton: dict[str, Reranker] = {}


def get_reranker(settings: RerankerSettings) -> Reranker:
    key = f"{settings.provider}:{settings.model_name}"
    if key in _singleton:
        return _singleton[key]

    if settings.provider == "cross_encoder":
        from manual_rag.rerankers.cross_encoder import CrossEncoderReranker
        reranker: Reranker = CrossEncoderReranker(settings)
    elif settings.provider == "none":
        from manual_rag.rerankers.cross_encoder import NoOpReranker
        reranker = NoOpReranker()
    else:
        raise ValueError(f"Unknown reranker provider: {settings.provider!r}")

    _singleton[key] = reranker
    return reranker
