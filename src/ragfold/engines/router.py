"""Engine router for retrieval and extraction sweeps."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragfold.compression import ContextCompressor
from ragfold.engines.base import CorpusInput, RagEngine, RetrievalResult, normalize_corpus
from ragfold.fusion import reciprocal_rank_fusion
from ragfold.preprocessing import Chunker
from ragfold.rerankers import BaseReranker

QueryInput = str | Mapping[str, Any]


@dataclass
class BatchResult:
    results: list[RetrievalResult] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    total_time_ms: int = 0

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


class EngineRouter:
    """Registry, selector, comparer, and batch runner for ragfold engines."""

    def __init__(
        self,
        engines: Sequence[RagEngine] | None = None,
        reranker: BaseReranker | None = None,
        chunker: Chunker | None = None,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self._engines: dict[str, RagEngine] = {}
        self.reranker = reranker
        self.chunker = chunker
        self.compressor = compressor
        for engine in engines or []:
            self.register(engine)

    @classmethod
    def from_engine_names(cls, names: Sequence[str]) -> EngineRouter:
        factories = default_engine_factories()
        engines: list[RagEngine] = []
        unknown = [name for name in names if name not in factories]
        if unknown:
            raise ValueError(f"Unknown engine(s): {', '.join(unknown)}")
        for name in names:
            engines.append(factories[name]())
        return cls(engines)

    @classmethod
    def default(cls) -> EngineRouter:
        return cls.from_engine_names(list(default_engine_factories()))

    def register(self, engine: RagEngine) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> RagEngine | None:
        return self._engines.get(name)

    def select(self, engine_hint: str | None = None) -> RagEngine:
        if engine_hint:
            engine = self._engines.get(engine_hint)
            if engine is None:
                raise ValueError(f"Unknown engine '{engine_hint}'")
            if not engine.is_available():
                raise RuntimeError(f"Engine '{engine_hint}' is registered but unavailable.")
            return engine

        for engine in self._engines.values():
            if engine.is_available():
                return engine
        raise ValueError("No available engines registered.")

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        engine_hint: str | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        engine = self.select(engine_hint=engine_hint)
        return await self._retrieve_with_engine(engine, corpus, query, top_k=top_k, **kwargs)

    async def retrieve_hybrid(
        self,
        corpus: CorpusInput,
        query: str,
        *,
        engines: list[str],
        top_k: int = 5,
        k: int = 60,
        concurrency: int = 4,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Run several engines concurrently and fuse them with RRF.

        This sits above `select()` (which returns a single engine). Unknown
        engine names raise `ValueError`; unavailable engines are skipped and
        recorded in metadata. The corpus is prepared once so every engine sees
        identical document ids, then any configured reranker/compressor runs on
        the fused passages exactly as the single-engine path does.
        """

        if not engines:
            raise ValueError("retrieve_hybrid requires at least one engine name.")

        selected: list[RagEngine] = []
        skipped: list[str] = []
        for name in engines:
            engine = self._engines.get(name)
            if engine is None:
                raise ValueError(f"Unknown engine '{name}'")
            if engine.is_available():
                selected.append(engine)
            else:
                skipped.append(name)

        if not selected:
            raise ValueError(
                "No available engines for hybrid retrieval among: " + ", ".join(engines)
            )

        start = time.perf_counter()
        prepared_corpus, prep_metadata = self._prepare_corpus(corpus)

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _run(engine: RagEngine) -> RetrievalResult:
            async with semaphore:
                return await engine.retrieve(prepared_corpus, query, top_k=top_k, **kwargs)

        engine_results = await asyncio.gather(*[_run(engine) for engine in selected])

        rankings = {
            engine.name: result.passages
            for engine, result in zip(selected, engine_results, strict=True)
        }
        fused = reciprocal_rank_fusion(rankings, k=k)[: max(top_k, 0)]

        ran_names = [engine.name for engine in selected]
        result = RetrievalResult(
            engine_name=f"rrf({','.join(ran_names)})",
            query=query,
            passages=fused,
            metadata={
                **prep_metadata,
                "fusion": {
                    "method": "rrf",
                    "k": k,
                    "engines": ran_names,
                    "skipped_engines": skipped,
                    "per_engine": {
                        engine.name: len(res.passages)
                        for engine, res in zip(selected, engine_results, strict=True)
                    },
                },
            },
            token_cost=sum(res.token_cost for res in engine_results),
        )

        if self.reranker and self.reranker.is_available() and result.passages:
            result.passages = self.reranker.rerank(query, result.passages)
            result.metadata["reranker"] = self.reranker.__class__.__name__
        if self.compressor and self.compressor.is_available() and result.passages:
            result.passages = self.compressor.compress(result.passages, query=query)
            result.metadata["compressor"] = self.compressor.__class__.__name__

        result.processing_time_ms = int((time.perf_counter() - start) * 1000)
        return result

    async def _retrieve_with_engine(
        self,
        engine: RagEngine,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        prepared_corpus, metadata = self._prepare_corpus(corpus)
        result = await engine.retrieve(prepared_corpus, query, top_k=top_k, **kwargs)
        result.metadata.update(metadata)
        if self.reranker and self.reranker.is_available() and result.passages:
            result.passages = self.reranker.rerank(query, result.passages)
            result.metadata["reranker"] = self.reranker.__class__.__name__
        if self.compressor and self.compressor.is_available() and result.passages:
            result.passages = self.compressor.compress(result.passages, query=query)
            result.metadata["compressor"] = self.compressor.__class__.__name__
        return result

    def _prepare_corpus(self, corpus: CorpusInput) -> tuple[CorpusInput, dict[str, Any]]:
        if not self.chunker or not self.chunker.is_available():
            return corpus, {}

        chunked: list[dict[str, Any]] = []
        for document in normalize_corpus(corpus):
            for chunk in self.chunker.chunk(document.text, document_id=document.id):
                chunked.append(
                    {
                        "id": f"{document.id}#chunk-{chunk.index}",
                        "text": chunk.text,
                        "metadata": {
                            **document.metadata,
                            **chunk.metadata,
                            "source_document_id": document.id,
                            "chunk_index": chunk.index,
                        },
                    }
                )
        return chunked, {"chunker": self.chunker.__class__.__name__}

    async def compare(
        self,
        corpus: CorpusInput,
        queries: Sequence[QueryInput],
        engines: Sequence[str] | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> dict[str, list[RetrievalResult]]:
        targets = self._target_engines(engines)
        results: dict[str, list[RetrievalResult]] = {}
        for engine in targets:
            if not engine.is_available():
                continue
            engine_results: list[RetrievalResult] = []
            for query in queries:
                try:
                    engine_results.append(
                        await self._retrieve_with_engine(
                            engine,
                            corpus,
                            _query_text(query),
                            top_k=top_k,
                            **kwargs,
                        )
                    )
                except NotImplementedError:
                    engine_results = []
                    break
            if engine_results:
                results[engine.name] = engine_results
        return results

    async def process_batch(
        self,
        corpus: CorpusInput,
        queries: Sequence[QueryInput],
        concurrency: int = 3,
        engine_hint: str | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> BatchResult:
        start = time.perf_counter()
        semaphore = asyncio.Semaphore(max(1, concurrency))
        batch = BatchResult(total=len(queries))

        async def _one(idx: int, query: QueryInput) -> None:
            query_id = _query_id(query, idx)
            async with semaphore:
                try:
                    result = await self.retrieve(
                        corpus,
                        _query_text(query),
                        top_k=top_k,
                        engine_hint=engine_hint,
                        **kwargs,
                    )
                    batch.results.append(result)
                    batch.succeeded += 1
                except Exception as exc:
                    batch.errors[query_id] = str(exc)
                    batch.failed += 1

        await asyncio.gather(*[_one(idx, query) for idx, query in enumerate(queries)])
        batch.total_time_ms = int((time.perf_counter() - start) * 1000)
        return batch

    def list_engines(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for engine in self._engines.values():
            caps = engine.capabilities
            rows.append(
                {
                    "name": engine.name,
                    "available": engine.is_available(),
                    "modality": caps.modality,
                    "retrieval_method": caps.retrieval_method,
                    "ocr_free": caps.ocr_free,
                    "type": "SaaS" if caps.saas else "VLM" if caps.vlm else "local",
                    "license": caps.license,
                    "rerank": caps.rerank,
                    "speed": caps.speed,
                    "cost": caps.cost,
                    "capabilities": caps.to_dict(),
                }
            )
        return rows

    def _target_engines(self, engines: Sequence[str] | None) -> list[RagEngine]:
        if engines is None:
            return list(self._engines.values())
        targets: list[RagEngine] = []
        for name in engines:
            engine = self._engines.get(name)
            if engine is not None:
                targets.append(engine)
        return targets


def default_engine_factories() -> dict[str, type[RagEngine]]:
    from ragfold.engines.bm25 import BM25Engine
    from ragfold.engines.dense import (
        CohereEmbedEngine,
        OpenAIEmbedEngine,
        SentenceTransformersEngine,
        VoyageEmbedEngine,
    )
    from ragfold.engines.frameworks import HaystackEngine, LlamaIndexEngine, TxtAIEngine
    from ragfold.engines.github_rag import (
        AgenticFileSearchEngine,
        LightRAGEngine,
        RAGAnythingEngine,
    )
    from ragfold.engines.text_rag import TextRagEngine
    from ragfold.engines.visual import ColPaliEngine, ColQwen2Engine, DSEEngine, PixelRAGEngine
    from ragfold.engines.wemm import WeMMEmbeddingEngine

    return {
        "bm25": BM25Engine,
        "text-rag": TextRagEngine,
        "sentence-transformers": SentenceTransformersEngine,
        "openai": OpenAIEmbedEngine,
        "cohere-embed": CohereEmbedEngine,
        "voyage": VoyageEmbedEngine,
        "colpali": ColPaliEngine,
        "colqwen2": ColQwen2Engine,
        "pixelrag": PixelRAGEngine,
        "dse": DSEEngine,
        "wemm": WeMMEmbeddingEngine,
        "lightrag": LightRAGEngine,
        "rag-anything": RAGAnythingEngine,
        "agentic-file-search": AgenticFileSearchEngine,
        "llamaindex": LlamaIndexEngine,
        "haystack": HaystackEngine,
        "txtai": TxtAIEngine,
    }


def _query_text(query: QueryInput) -> str:
    if isinstance(query, str):
        return query
    return str(query.get("query") or query.get("text") or "")


def _query_id(query: QueryInput, idx: int) -> str:
    if isinstance(query, str):
        return f"q{idx + 1}"
    return str(query.get("id") or f"q{idx + 1}")
