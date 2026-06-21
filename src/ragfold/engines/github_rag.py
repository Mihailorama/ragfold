"""Gated adapters for RAG-related repositories forked under Mihailorama."""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

from ragfold.engines.base import (
    CorpusInput,
    DocumentChunk,
    EngineCapabilities,
    RagAnswer,
    RagEngine,
    RetrievalResult,
    RetrievedPassage,
    normalize_corpus,
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

    def __init__(
        self,
        client: Any | None = None,
        *,
        working_dir: str | None = None,
        llm_model_func: Any | None = None,
        embedding_func: Any | None = None,
        lightrag_cls: Any | None = None,
        query_param_cls: Any | None = None,
        lightrag_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if client is None and any(
            value is not None
            for value in (
                working_dir,
                llm_model_func,
                embedding_func,
                lightrag_cls,
                query_param_cls,
            )
        ):
            client = _LightRAGRuntimeClient(
                working_dir=working_dir,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                lightrag_cls=lightrag_cls,
                query_param_cls=query_param_cls,
                lightrag_kwargs=dict(lightrag_kwargs or {}),
            )
        super().__init__(client=client)

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

    def is_available(self) -> bool:
        if isinstance(self.client, _LightRAGRuntimeClient):
            return self.client.is_available()
        return super().is_available()


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


class _LightRAGRuntimeClient:
    def __init__(
        self,
        *,
        working_dir: str | None,
        llm_model_func: Any | None,
        embedding_func: Any | None,
        lightrag_cls: Any | None,
        query_param_cls: Any | None,
        lightrag_kwargs: dict[str, Any],
    ) -> None:
        self.working_dir = working_dir
        self.llm_model_func = llm_model_func
        self.embedding_func = embedding_func
        self._lightrag_cls = lightrag_cls
        self._query_param_cls = query_param_cls
        self._lightrag_kwargs = lightrag_kwargs
        self._rag: Any | None = None
        self._initialized = False
        self._indexed_signature: tuple[tuple[str, str], ...] | None = None

    def is_available(self) -> bool:
        try:
            self._validate_configuration()
            self._resolve_classes()
        except NotImplementedError:
            return False
        return True

    async def retrieve(
        self,
        *,
        corpus: CorpusInput,
        query: str,
        top_k: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._validate_configuration()
        lightrag_cls, query_param_cls = self._resolve_classes()
        rag = await self._ensure_rag(lightrag_cls)
        chunks = normalize_corpus(corpus)
        await self._index_corpus(rag, chunks)

        mode = str(kwargs.pop("mode", "hybrid"))
        generate_answer = bool(kwargs.pop("generate_answer", False))
        query_param_kwargs = dict(kwargs.pop("query_param_kwargs", {}))

        context_param = self._make_query_param(
            query_param_cls,
            mode=mode,
            top_k=top_k,
            only_need_context=True,
            extra=query_param_kwargs,
        )
        context = await self._query(rag, query, context_param)
        answer = None
        if generate_answer:
            answer_param = self._make_query_param(
                query_param_cls,
                mode=mode,
                top_k=top_k,
                only_need_context=False,
                extra=query_param_kwargs,
            )
            answer = await self._query(rag, query, answer_param)

        return {
            "passages": [
                {
                    "document_id": "lightrag-context",
                    "text": str(context),
                    "score": 1.0,
                    "metadata": {"source": "lightrag", "mode": mode},
                }
            ]
            if context
            else [],
            "answer": str(answer) if answer is not None else None,
            "metadata": {
                "runtime": "lightrag-hku",
                "mode": mode,
                "top_k": top_k,
            },
        }

    def _validate_configuration(self) -> None:
        missing = []
        if not self.working_dir:
            missing.append("working_dir")
        if self.llm_model_func is None:
            missing.append("llm_model_func")
        if self.embedding_func is None:
            missing.append("embedding_func")
        if missing:
            raise NotImplementedError(
                "LightRAG runtime requires "
                f"{', '.join(missing)}. Install ragfold[lightrag] and pass configured "
                "model functions before using this engine."
            )

    def _resolve_classes(self) -> tuple[Any, Any]:
        if self._lightrag_cls is not None and self._query_param_cls is not None:
            return self._lightrag_cls, self._query_param_cls

        try:
            lightrag_module = import_module("lightrag")
            lightrag_cls = self._lightrag_cls or getattr(lightrag_module, "LightRAG")
            query_param_cls = self._query_param_cls or getattr(lightrag_module, "QueryParam")
        except (ImportError, AttributeError):
            try:
                lightrag_cls = self._lightrag_cls or getattr(import_module("lightrag"), "LightRAG")
                query_param_cls = self._query_param_cls or getattr(
                    import_module("lightrag.core.param"), "QueryParam"
                )
            except (ImportError, AttributeError) as exc:
                raise NotImplementedError(
                    "LightRAG runtime is unavailable. Install ragfold[lightrag] or pass "
                    "lightrag_cls and query_param_cls for an injected runtime."
                ) from exc
        return lightrag_cls, query_param_cls

    async def _ensure_rag(self, lightrag_cls: Any) -> Any:
        if self._rag is None:
            self._rag = lightrag_cls(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                embedding_func=self.embedding_func,
                **self._lightrag_kwargs,
            )
        if not self._initialized:
            initialize = getattr(self._rag, "initialize_storages", None)
            if initialize is not None:
                await _maybe_await(initialize())
            self._initialized = True
        return self._rag

    async def _index_corpus(self, rag: Any, chunks: list[DocumentChunk]) -> None:
        signature = tuple((chunk.id, chunk.text) for chunk in chunks)
        if signature == self._indexed_signature:
            return

        texts = [chunk.text for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        insert = getattr(rag, "ainsert", None) or getattr(rag, "insert", None)
        if insert is None:
            raise NotImplementedError("LightRAG runtime must expose ainsert() or insert().")

        await _maybe_await(insert(texts, ids=ids))
        self._indexed_signature = signature

    def _make_query_param(
        self,
        query_param_cls: Any,
        *,
        mode: str,
        top_k: int,
        only_need_context: bool,
        extra: Mapping[str, Any],
    ) -> Any:
        return query_param_cls(
            mode=mode,
            top_k=top_k,
            chunk_top_k=top_k,
            only_need_context=only_need_context,
            **extra,
        )

    async def _query(self, rag: Any, query: str, param: Any) -> Any:
        query_method = getattr(rag, "aquery", None) or getattr(rag, "query", None)
        if query_method is None:
            raise NotImplementedError("LightRAG runtime must expose aquery() or query().")
        return await _maybe_await(query_method(query, param=param))


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


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
