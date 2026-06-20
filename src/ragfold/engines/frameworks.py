"""Thin gated adapters for framework retrievers."""

from __future__ import annotations

import importlib.util
from typing import Any

from ragfold.engines.base import CorpusInput, EngineCapabilities, RagEngine, RetrievalResult


class _FrameworkRetrieverEngine(RagEngine):
    engine_name = "framework"
    package_name = ""
    extra_name = ""

    def __init__(self, retriever: Any | None = None) -> None:
        self.retriever = retriever

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
            license="framework-dependent",
            speed="varies",
            cost="varies",
        )

    def is_available(self) -> bool:
        return self.retriever is not None and bool(importlib.util.find_spec(self.package_name))

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        raise NotImplementedError(
            f"Engine '{self.name}' unavailable: install ragfold[{self.extra_name}] and inject a "
            "framework retriever adapter."
        )


class LlamaIndexEngine(_FrameworkRetrieverEngine):
    engine_name = "llamaindex"
    package_name = "llama_index"
    extra_name = "llamaindex"


class HaystackEngine(_FrameworkRetrieverEngine):
    engine_name = "haystack"
    package_name = "haystack"
    extra_name = "haystack"


class TxtAIEngine(_FrameworkRetrieverEngine):
    engine_name = "txtai"
    package_name = "txtai"
    extra_name = "txtai"
