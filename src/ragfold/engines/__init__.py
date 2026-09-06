"""Engine exports and default registry helpers."""

from __future__ import annotations

from ragfold.engines.base import (
    DocumentChunk,
    EngineCapabilities,
    RagAnswer,
    RagEngine,
    RetrievalResult,
    RetrievedPassage,
)
from ragfold.engines.bm25 import BM25Engine
from ragfold.engines.dense import (
    CohereEmbedEngine,
    OpenAIEmbedEngine,
    SentenceTransformersEngine,
    VoyageEmbedEngine,
)
from ragfold.engines.frameworks import HaystackEngine, LlamaIndexEngine, TxtAIEngine
from ragfold.engines.github_rag import AgenticFileSearchEngine, LightRAGEngine, RAGAnythingEngine
from ragfold.engines.router import EngineRouter, default_engine_factories
from ragfold.engines.text_rag import TextRagEngine
from ragfold.engines.visual import ColPaliEngine, ColQwen2Engine, DSEEngine, PixelRAGEngine
from ragfold.engines.wemm import WeMMEmbeddingEngine

DEFAULT_ENGINE_NAMES = list(default_engine_factories())


def build_default_router() -> EngineRouter:
    return EngineRouter.from_engine_names(DEFAULT_ENGINE_NAMES)


__all__ = [
    "BM25Engine",
    "CohereEmbedEngine",
    "ColPaliEngine",
    "ColQwen2Engine",
    "DEFAULT_ENGINE_NAMES",
    "DSEEngine",
    "DocumentChunk",
    "EngineCapabilities",
    "EngineRouter",
    "AgenticFileSearchEngine",
    "HaystackEngine",
    "LightRAGEngine",
    "LlamaIndexEngine",
    "OpenAIEmbedEngine",
    "PixelRAGEngine",
    "RAGAnythingEngine",
    "RagAnswer",
    "RagEngine",
    "RetrievedPassage",
    "RetrievalResult",
    "SentenceTransformersEngine",
    "TextRagEngine",
    "TxtAIEngine",
    "VoyageEmbedEngine",
    "WeMMEmbeddingEngine",
    "build_default_router",
]
