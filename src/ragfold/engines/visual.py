"""OCR-free visual and late-interaction retrieval adapters."""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Protocol

from ragfold.engines.base import CorpusInput, EngineCapabilities, RagEngine, RetrievedPassage, RetrievalResult, normalize_corpus
from ragfold.vectorstores import InMemoryVectorStore, VectorRecord


class VisualEmbedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class ColPaliEngine(RagEngine):
    """ColPali adapter with an injected-embedder path for local tests."""

    def __init__(self, embedder: VisualEmbedder | None = None) -> None:
        self.embedder = embedder

    @property
    def name(self) -> str:
        return "colpali"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="visual",
            retrieval_method="late-interaction",
            ocr_free=True,
            local=True,
            requires_gpu=True,
            license="research/model-dependent",
            speed="slow",
            cost="free",
        )

    def is_available(self) -> bool:
        return self.embedder is not None or importlib.util.find_spec("colpali_engine") is not None

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        if self.embedder is None:
            raise NotImplementedError(
                "Engine 'colpali' unavailable: install ragfold[colpali] and inject a loaded embedder."
            )

        start = time.perf_counter()
        chunks = normalize_corpus(corpus)
        doc_vectors = self.embedder.encode([chunk.text for chunk in chunks])
        query_vector = self.embedder.encode([query])[0]
        store = InMemoryVectorStore()
        store.add(
            [
                VectorRecord(id=chunk.id, vector=list(vector), text=chunk.text, metadata=chunk.metadata)
                for chunk, vector in zip(chunks, doc_vectors, strict=True)
            ]
        )
        matches = store.query(list(query_vector), top_k=top_k)
        passages = [
            RetrievedPassage(
                document_id=match.id,
                text=match.text,
                score=match.score,
                rank=rank,
                metadata=match.metadata,
            )
            for rank, match in enumerate(matches, start=1)
        ]
        return RetrievalResult(
            engine_name=self.name,
            query=query,
            passages=passages,
            metadata={"method": "late-interaction", "ocr_free": True},
            processing_time_ms=int((time.perf_counter() - start) * 1000),
        )


class _UnavailableVisualEngine(RagEngine):
    engine_name = "visual"
    package_name = ""
    extra_name = ""

    @property
    def name(self) -> str:
        return self.engine_name

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="visual",
            retrieval_method="late-interaction",
            ocr_free=True,
            local=True,
            requires_gpu=True,
            license="research/model-dependent",
            speed="slow",
            cost="free",
        )

    def is_available(self) -> bool:
        return bool(self.package_name and importlib.util.find_spec(self.package_name))

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        raise NotImplementedError(
            f"Engine '{self.name}' unavailable: install ragfold[{self.extra_name}] and configure "
            "model weights before use."
        )


class ColQwen2Engine(_UnavailableVisualEngine):
    engine_name = "colqwen2"
    package_name = "colpali_engine"
    extra_name = "colqwen2"


class PixelRAGEngine(_UnavailableVisualEngine):
    engine_name = "pixelrag"
    package_name = "pixelrag"
    extra_name = "pixelrag"

    @property
    def capabilities(self) -> EngineCapabilities:
        caps = super().capabilities
        return EngineCapabilities(**{**caps.to_dict(), "vlm": True})


class DSEEngine(_UnavailableVisualEngine):
    engine_name = "dse"
    package_name = "dse"
    extra_name = "dse"
