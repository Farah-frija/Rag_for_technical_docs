"""
embeddings/sentence_transformers_provider.py
----------------------------------------------
Wraps a `sentence_transformers.SentenceTransformer` model behind the
`EmbeddingProvider` interface. Loads the model lazily (first `.embed()`
call) and caches it for the process lifetime, exactly like the original
codebase's `_get_embedder()` singletons -- but now there's exactly one of
them, shared by indexing and every retriever, instead of three separate
copies (`retriever.py`, `summarize_tree.py`, `tree_builder.py` each had
their own).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from manual_rag.config import EmbeddingSettings
from manual_rag.core.interfaces import EmbeddingProvider
from manual_rag.logging_config import get_logger

logger = get_logger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, settings: EmbeddingSettings) -> None:
        self._model_name = settings.model_name
        self._device = settings.device
        self._model = None  # lazy

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model '{self._model_name}' ...")
            self._model = SentenceTransformer(self._model_name, device=self._device)
            logger.info("Embedding model ready.")
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        vectors = model.encode(list(texts), convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        model = self._ensure_loaded()
        return int(model.get_sentence_embedding_dimension())
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self._ensure_loaded().tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

    @property
    def max_tokens(self) -> int:
        return int(self._ensure_loaded().max_seq_length)-2