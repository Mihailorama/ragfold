"""Gated adapters for RAG-related repositories forked under Mihailorama."""

from __future__ import annotations

from typing import Any

from ragfold.engines.base import CorpusInput, EngineCapabilities, RagEngine, RetrievalResult


class _InjectedClientEngine(RagEngine):
    engine_name = "github-rag"
    extra_name = "github-rag"

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return self.engine_name

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="text",
            retrieval_method="framework",
            ocr_free=True,
            local=True,
            license="project-dependent",
            speed="varies",
            cost="varies",
        )

    def is_available(self) -> bool:
        return self.client is not None

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        raise NotImplementedError(
            f"Engine '{self.name}' unavailable: install ragfold[{self.extra_name}] and inject "
            "a configured client for real inference."
        )


class LightRAGEngine(_InjectedClientEngine):
    """Framework adapter for HKUDS LightRAG."""

    engine_name = "lightrag"
    extra_name = "lightrag"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="text",
            retrieval_method="graph-hybrid",
            ocr_free=True,
            local=True,
            license="MIT",
            speed="medium",
            cost="infra/provider-dependent",
        )


class RAGAnythingEngine(_InjectedClientEngine):
    """Multimodal document RAG adapter for HKUDS RAG-Anything."""

    engine_name = "rag-anything"
    extra_name = "rag-anything"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="multimodal",
            retrieval_method="multimodal",
            ocr_free=False,
            local=True,
            vlm=True,
            requires_gpu=True,
            license="MIT",
            speed="slow",
            cost="infra/provider-dependent",
        )


class AgenticFileSearchEngine(_InjectedClientEngine):
    """Agentic document-search adapter inspired by agentic-file-search."""

    engine_name = "agentic-file-search"
    extra_name = "agentic-file-search"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="text",
            retrieval_method="agentic",
            ocr_free=False,
            local=False,
            saas=True,
            answer_generation=True,
            requires_api_key=True,
            license="unknown",
            speed="slow",
            cost="paid tokens",
        )
