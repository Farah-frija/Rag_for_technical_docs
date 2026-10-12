"""core/exceptions.py -- domain-specific exceptions, so callers can catch
something more meaningful than a bare Exception."""


class ManualRagError(Exception):
    """Base class for all manual_rag errors."""


class LLMCallError(ManualRagError):
    """Raised when an LLM provider call fails after retries."""


class IndexNotFoundError(ManualRagError):
    """Raised when a ManualIndex JSON path doesn't exist or can't be parsed."""


class NoTOCDetectedError(ManualRagError):
    """Raised when `force_mode='toc'` is requested but no TOC was detected."""


class RetrieverConfigError(ManualRagError):
    """Raised when a retrieval strategy is misconfigured (bad name, missing
    dependency, incompatible index shape, ...)."""


class VectorStoreError(ManualRagError):
    """Raised on vector store connection / query / upsert failures."""
