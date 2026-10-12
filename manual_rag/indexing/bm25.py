"""
indexing/bm25.py
--------------------
Thin wrapper around `rank_bm25.BM25Okapi` that tracks doc_ids alongside the
underlying score matrix, so callers get back {doc_id: score} instead of a
bare positional array. Tokenization itself lives in `manual_rag.utils.text`
(shared with everything else that needs to tokenize/normalise text), so this
module is now genuinely just "the BM25 index data structure".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from manual_rag.utils.text import tokenize


class BM25Index:
    def __init__(self, doc_ids: List[str], tokenized_docs: List[List[str]]) -> None:
        self.doc_ids = doc_ids
        self._bm25 = BM25Okapi(tokenized_docs) if any(tokenized_docs) else None

    @classmethod
    def build(cls, chunks: List[Dict[str, Any]]) -> "BM25Index":
        """chunks: list of dicts with {"id": ..., "content": ...} and
        optionally a pre-cached {"tokens": [...]}. Falls back to tokenizing
        `content` on the fly if `tokens` is missing."""
        doc_ids: List[str] = []
        tokenized: List[List[str]] = []
        for c in chunks:
            toks = c.get("tokens") or tokenize(c.get("content", ""))
            doc_ids.append(c["id"])
            tokenized.append(toks)
        return cls(doc_ids, tokenized)

    def score(self, query: str) -> Dict[str, float]:
        if self._bm25 is None:
            return {doc_id: 0.0 for doc_id in self.doc_ids}
        query_tokens = tokenize(query)
        if not query_tokens:
            return {doc_id: 0.0 for doc_id in self.doc_ids}
        scores = self._bm25.get_scores(query_tokens)
        return dict(zip(self.doc_ids, scores))



def save_bm25_corpus(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_bm25_corpus(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bm25_corpus_path(index_path: str) -> str:
    return str(Path(index_path).with_suffix("")) + "_bm25.json"