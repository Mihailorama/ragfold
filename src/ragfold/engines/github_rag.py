"""Gated adapters for RAG-related repositories forked under Mihailorama."""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ragfold.engines.base import (
    CorpusInput,
    EngineCapabilities,
    RagAnswer,
    RagEngine,
    RetrievalResult,
    RetrievedPassage,
)


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
        if self.client is None:
            raise NotImplementedError(
                f"Engine '{self.name}' unavailable: install ragfold[{self.extra_name}] and inject "
                "a configured client for real inference."
            )

        start = time.perf_counter()
        raw_result = await self._call_client(corpus=corpus, query=query, top_k=top_k, **kwargs)
        result = self._normalize_result(raw_result, query=query)
        if result.processing_time_ms == 0:
            result.processing_time_ms = int((time.perf_counter() - start) * 1000)
        return result

    async def _call_client(
        self,
        *,
        corpus: CorpusInput,
        query: str,
        top_k: int,
        **kwargs: Any,
    ) -> Any:
        assert self.client is not None
        for method_name in ("retrieve", "query", "search"):
            method = getattr(self.client, method_name, None)
            if method is None:
                continue
            value = method(corpus=corpus, query=query, top_k=top_k, **kwargs)
            if inspect.isawaitable(value):
                return await value
            return value
        raise NotImplementedError(
            f"Injected client for engine '{self.name}' must expose "
            "retrieve(), query(), or search()."
        )

    def _normalize_result(self, raw_result: Any, query: str) -> RetrievalResult:
        if isinstance(raw_result, RetrievalResult):
            raw_result.engine_name = self.name
            return raw_result

        if isinstance(raw_result, Mapping):
            passages = _normalize_passages(
                raw_result.get("passages") or raw_result.get("contexts") or []
            )
            answer_text = raw_result.get("answer")
            answer = None
            if answer_text is not None:
                answer = RagAnswer(
                    answer=str(answer_text),
                    citations=[str(item) for item in raw_result.get("citations", [])],
                    confidence=_optional_float(raw_result.get("confidence")),
                    metadata=dict(raw_result.get("answer_metadata", {})),
                )
            return RetrievalResult(
                engine_name=self.name,
                query=str(raw_result.get("query") or query),
                passages=passages,
                answer=answer,
                metadata=dict(raw_result.get("metadata", {})),
                processing_time_ms=int(raw_result.get("processing_time_ms", 0) or 0),
                token_cost=float(raw_result.get("token_cost", 0.0) or 0.0),
            )

        if isinstance(raw_result, Sequence) and not isinstance(raw_result, str):
            return RetrievalResult(
                engine_name=self.name,
                query=query,
                passages=_normalize_passages(raw_result),
            )

        raise TypeError(
            f"Injected client for engine '{self.name}' returned unsupported result type: "
            f"{type(raw_result).__name__}"
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


def _normalize_passages(raw_passages: Any) -> list[RetrievedPassage]:
    passages: list[RetrievedPassage] = []
    if raw_passages is None:
        return passages
    for rank, item in enumerate(raw_passages, start=1):
        if isinstance(item, RetrievedPassage):
            item.rank = rank
            passages.append(item)
            continue
        if isinstance(item, Mapping):
            passages.append(
                RetrievedPassage(
                    document_id=str(item.get("document_id") or item.get("id") or f"doc-{rank}"),
                    text=str(item.get("text") or item.get("content") or ""),
                    score=float(item.get("score", 0.0) or 0.0),
                    rank=int(item.get("rank", rank) or rank),
                    metadata=dict(item.get("metadata", {})),
                )
            )
            continue
        passages.append(
            RetrievedPassage(
                document_id=f"doc-{rank}",
                text=str(item),
                score=0.0,
                rank=rank,
            )
        )
    return passages


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
