"""Pure-Python TF-IDF text retrieval baseline."""

from __future__ import annotations

import time
from typing import Any

from ragfold.engines._text import rank_passages, tfidf_scores
from ragfold.engines.base import CorpusInput, EngineCapabilities, RagEngine, RetrievalResult, normalize_corpus


class TextRagEngine(RagEngine):
    """No-dependency TF-IDF baseline used as an always-available control."""

    @property
    def name(self) -> str:
        return "text-rag"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="text",
            retrieval_method="lexical",
            ocr_free=True,
            local=True,
            license="MIT",
            speed="fast",
            cost="free",
        )

    def is_available(self) -> bool:
        return True

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        start = time.perf_counter()
        chunks = normalize_corpus(corpus)
        passages = rank_passages(chunks, tfidf_scores(chunks, query), top_k)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return RetrievalResult(
            engine_name=self.name,
            query=query,
            passages=passages,
            metadata={"method": "tf-idf"},
            processing_time_ms=elapsed_ms,
        )
