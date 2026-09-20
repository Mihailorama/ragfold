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
from ragfold.fusion_tuning import TuneResult, tune_rrf_weights

__all__ = [
    "DocumentChunk",
    "EngineCapabilities",
    "EngineRouter",
    "RagAnswer",
    "RagEngine",
    "RetrievedPassage",
    "RetrievalResult",
    "TuneResult",
    "reciprocal_rank_fusion",
    "tune_rrf_weights",
]

__version__ = "0.1.0"
