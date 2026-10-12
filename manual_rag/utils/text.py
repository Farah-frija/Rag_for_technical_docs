"""
utils/text.py
--------------
Small, pure text-processing helpers reused across indexing and retrieval:

  * `tokenize`           -- BM25 tokenizer (was duplicated as the sole
                             purpose of `indexing/bm25_utils.py`; kept but
                             now just one function among several here).
  * `normalise_title`    -- accent-stripping/whitespace-collapsing used by
                             both the tree builder (matching TOC titles in
                             the markdown body) and eval.py (matching
                             predicted vs ground-truth section names).
  * `make_title_pattern` -- builds a whitespace-tolerant regex for a title.
  * `cosine_similarity`  -- plain-Python cosine, used anywhere a vector
                             store isn't warranted (e.g. scoring a handful
                             of cached section embeddings against a query).
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

# Small, technical-doc-safe stopword list -- deliberately conservative, we'd
# rather keep a borderline word than drop something that matters in a
# technical manual (e.g. unit words, short codes).
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to",
    "in", "on", "for", "and", "or", "with", "as", "at", "by", "this", "that",
    "it", "from", "which", "these", "those", "will", "shall", "can", "may",
}

_WS_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    """Lowercase + regex word split, drop stopwords and single-char tokens."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def normalise_title(text: str) -> str:
    """Strip accents, collapse whitespace, lowercase -- used both when
    locating a TOC title inside converted markdown, and when comparing a
    predicted section name against ground truth in eval."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = _WS_RE.sub(" ", text.strip().lower())
    return text.rstrip(".")


def make_title_pattern(title: str) -> re.Pattern:
    """A regex matching `title`'s words in order, tolerant of the exact
    whitespace/newline between them (titles get reflowed across a PDF->MD
    conversion)."""
    words = normalise_title(title).split()
    sep = r"(?:[^\S\r\n]+|\s*\r?\n\s*)"
    pattern = sep.join(re.escape(w) for w in words)
    return re.compile(r"(?i)" + pattern)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
