"""RAG and information-extraction engine aggregation toolkit."""

from __future__ import annotations

from ragfold.engines.base import (
    DocumentChunk,
    EngineCapabilities,
    RagAnswer,
    RagEngine,
    RetrievalResult,
    RetrievedPassage,
)
from ragfold.engines.router import EngineRouter
from ragfold.fusion import reciprocal_rank_fusion

__all__ = [
    "DocumentChunk",
    "EngineCapabilities",
    "EngineRouter",
    "RagAnswer",
    "RagEngine",
    "RetrievedPassage",
    "RetrievalResult",
    "reciprocal_rank_fusion",
]

__version__ = "0.1.0"
