"""BM25 lexical retrieval engine."""

from __future__ import annotations

import importlib.util
import math
import time
from collections import Counter
from typing import Any

from ragfold.engines._text import rank_passages, tokenize
from ragfold.engines.base import CorpusInput, EngineCapabilities, RagEngine, RetrievalResult, normalize_corpus


class BM25Engine(RagEngine):
    """BM25 baseline with pure-Python scoring and optional `rank_bm25` acceleration."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    @property
    def name(self) -> str:
        return "bm25"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="text",
            retrieval_method="lexical",
            ocr_free=True,
            local=True,
            license="Apache-2.0-compatible",
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
        scores = self._scores([chunk.text for chunk in chunks], query)
        passages = rank_passages(chunks, scores, top_k)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return RetrievalResult(
            engine_name=self.name,
            query=query,
            passages=passages,
            metadata={
                "method": "bm25",
                "backend": "rank_bm25" if importlib.util.find_spec("rank_bm25") else "python",
            },
            processing_time_ms=elapsed_ms,
        )

    def _scores(self, documents: list[str], query: str) -> list[float]:
        tokenized_docs = [tokenize(doc) for doc in documents]
        query_terms = tokenize(query)
        if not tokenized_docs:
            return []
        if not query_terms:
            return [0.0 for _ in tokenized_docs]

        if importlib.util.find_spec("rank_bm25") is not None:
            try:
                from rank_bm25 import BM25Okapi

                bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
                return [float(score) for score in bm25.get_scores(query_terms)]
            except Exception:
                pass

        return self._python_scores(tokenized_docs, query_terms)

    def _python_scores(self, tokenized_docs: list[list[str]], query_terms: list[str]) -> list[float]:
        total_docs = len(tokenized_docs)
        avgdl = sum(len(doc) for doc in tokenized_docs) / total_docs if total_docs else 0.0
        doc_freq: Counter[str] = Counter()
        for terms in tokenized_docs:
            doc_freq.update(set(terms))

        scores: list[float] = []
        for terms in tokenized_docs:
            counts = Counter(terms)
            doc_len = len(terms) or 1
            score = 0.0
            for term in query_terms:
                if counts[term] == 0:
                    continue
                idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                numerator = counts[term] * (self.k1 + 1)
                denominator = counts[term] + self.k1 * (1 - self.b + self.b * doc_len / (avgdl or 1))
                score += idf * numerator / denominator
            scores.append(score)
        return scores
