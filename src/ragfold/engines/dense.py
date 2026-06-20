"""Dense retrieval adapters over the ragfold VectorStore interface."""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Any, Protocol

from ragfold.engines.base import CorpusInput, EngineCapabilities, RagEngine, RetrievedPassage, RetrievalResult, normalize_corpus
from ragfold.vectorstores import InMemoryVectorStore, VectorRecord, VectorStore


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class _DenseVectorEngine(RagEngine):
    engine_name = "dense"
    package_name = ""

    def __init__(
        self,
        model: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.model = model
        self.vector_store = vector_store or InMemoryVectorStore()

    @property
    def name(self) -> str:
        return self.engine_name

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="text",
            retrieval_method="dense",
            ocr_free=True,
            local=True,
            vector_store=True,
            license="varies",
            speed="medium",
            cost="free",
        )

    def is_available(self) -> bool:
        return self.model is not None or bool(
            self.package_name and importlib.util.find_spec(self.package_name)
        )

    def _get_model(self) -> Embedder:
        if self.model is None:
            raise NotImplementedError(
                f"Engine '{self.name}' unavailable: install ragfold[{self.name}] or inject a model."
            )
        return self.model

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        start = time.perf_counter()
        model = self._get_model()
        chunks = normalize_corpus(corpus)
        texts = [chunk.text for chunk in chunks]
        doc_vectors = model.encode(texts)
        query_vector = model.encode([query])[0]

        if hasattr(self.vector_store, "clear"):
            self.vector_store.clear()  # type: ignore[attr-defined]
        self.vector_store.add(
            [
                VectorRecord(
                    id=chunk.id,
                    vector=list(vector),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
                for chunk, vector in zip(chunks, doc_vectors, strict=True)
            ]
        )
        matches = self.vector_store.query(list(query_vector), top_k=top_k)
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
            metadata={"method": "dense", "vector_store": self.vector_store.__class__.__name__},
            processing_time_ms=int((time.perf_counter() - start) * 1000),
        )


class SentenceTransformersEngine(_DenseVectorEngine):
    engine_name = "sentence-transformers"
    package_name = "sentence_transformers"

    def _get_model(self) -> Embedder:
        if self.model is not None:
            return self.model
        raise NotImplementedError(
            "Engine 'sentence-transformers' unavailable: install ragfold[sentence-transformers] "
            "and inject or initialize a local model."
        )


class _SaaSDenseEngine(_DenseVectorEngine):
    env_var = ""
    extra_name = ""

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="text",
            retrieval_method="dense",
            ocr_free=True,
            local=False,
            saas=True,
            vector_store=True,
            requires_api_key=True,
            license="commercial terms",
            speed="fast",
            cost="paid",
        )

    def is_available(self) -> bool:
        return self.model is not None or bool(os.getenv(self.env_var))

    def ensure_available(self) -> None:
        if not self.is_available():
            raise NotImplementedError(
                f"Engine '{self.name}' unavailable: install ragfold[{self.extra_name}] and set "
                f"{self.env_var} or inject a client/model."
            )

    def _get_model(self) -> Embedder:
        if self.model is not None:
            return self.model
        self.ensure_available()
        raise NotImplementedError(
            f"Engine '{self.name}' is gated for real SaaS inference; inject a model/client in tests."
        )


class OpenAIEmbedEngine(_SaaSDenseEngine):
    engine_name = "openai"
    env_var = "OPENAI_API_KEY"
    extra_name = "openai"


class CohereEmbedEngine(_SaaSDenseEngine):
    engine_name = "cohere-embed"
    env_var = "COHERE_API_KEY"
    extra_name = "cohere"


class VoyageEmbedEngine(_SaaSDenseEngine):
    engine_name = "voyage"
    env_var = "VOYAGE_API_KEY"
    extra_name = "voyage"
