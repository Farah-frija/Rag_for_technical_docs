"""
rerankers/cross_encoder.py
-----------------------------
HuggingFace cross-encoder reranker (default: BAAI/bge-reranker-v2-m3),
matching the original codebase's `_ce_score` but behind the `Reranker`
interface and with the model loaded once, lazily, as a singleton.
"""

from __future__ import annotations

from typing import List

from manual_rag.config import RerankerSettings
from manual_rag.core.interfaces import Reranker
from manual_rag.logging_config import get_logger
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
logger = get_logger(__name__)


class CrossEncoderReranker(Reranker):
    def __init__(self, settings: RerankerSettings) -> None:
        self._model_name = settings.model_name
        self._chunk_overlap_tokens: int = settings.chunk_overlap_tokens
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            

            logger.info(f"Loading cross-encoder reranker '{self._model_name}' ...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.eval()
            self._torch = torch
            logger.info("Cross-encoder reranker ready.")
        return self._tokenizer, self._model

    def score(self, query: str, document: str) -> float:
        tokenizer, model = self._ensure_loaded()
        torch = self._torch
        with torch.no_grad():
            inputs = tokenizer(
                query, document,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            logit = model(**inputs).logits[0, 0]
            return float(torch.sigmoid(logit))

    def count_tokens(self, text: str) -> int:
        tokenizer, _ = self._ensure_loaded()
        return len(tokenizer.encode(text, add_special_tokens=False))

    def get_context_budget(self, query: str) -> int:
        """Token slots available for document text alongside this query:
        model_max_length minus query tokens minus 3 special tokens ([CLS], [SEP], [SEP])."""
        tokenizer, _ = self._ensure_loaded()
        query_tokens = len(tokenizer.encode(query, add_special_tokens=False))
        return tokenizer.model_max_length - query_tokens - 3

    def chunk_for_scoring(self, query: str, document: str) -> List[str]:
        """Split `document` into chunks that each fit within the model's
        context alongside `query`, with token overlap for continuity."""
        tokenizer, _ = self._ensure_loaded()
        available = self.get_context_budget(query)
        overlap = min(self._chunk_overlap_tokens, available // 4)
        step = max(1, available - overlap)

        doc_ids = tokenizer.encode(document, add_special_tokens=False)
        if len(doc_ids) <= available:
            return [document]

        chunks: List[str] = []
        start = 0
        while start < len(doc_ids):
            chunks.append(
                tokenizer.decode(doc_ids[start: start + available], skip_special_tokens=True)
            )
            start += step
        return chunks


class NoOpReranker(Reranker):
    """Used when `MANUAL_RAG_RERANKER_PROVIDER=none` -- e.g. for fast local
    smoke tests without pulling a cross-encoder model. Scores everything
    equally so ranking degrades gracefully to whatever order the recall
    stage produced, rather than failing."""

    def score(self, query: str, document: str) -> float:
        return 0.0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_context_budget(self, query: str) -> int:
        return 512

    def chunk_for_scoring(self, query: str, document: str) -> List[str]:
        return [document]