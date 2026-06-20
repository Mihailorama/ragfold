"""Optional reranking adapters."""

from __future__ import annotations

import importlib.util
import os
from abc import ABC, abstractmethod

from ragfold.engines.base import RetrievedPassage


class BaseReranker(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def rerank(self, query: str, passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
        ...


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model: object | None = None) -> None:
        self.model = model

    def is_available(self) -> bool:
        return (
            self.model is not None
            or importlib.util.find_spec("sentence_transformers") is not None
        )

    def rerank(self, query: str, passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
        if self.model is None:
            raise NotImplementedError(
                "CrossEncoderReranker unavailable: install ragfold[rerank-cross-encoder] "
                "and inject a loaded model."
            )
        raise NotImplementedError("CrossEncoderReranker real inference is gated behind slow tests.")


class CohereReranker(BaseReranker):
    def __init__(self, client: object | None = None) -> None:
        self.client = client

    def is_available(self) -> bool:
        return self.client is not None or bool(os.getenv("COHERE_API_KEY"))

    def rerank(self, query: str, passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
        if self.client is None:
            raise NotImplementedError(
                "CohereReranker unavailable: install ragfold[rerank-cohere], set COHERE_API_KEY, "
                "and inject a client for live inference."
            )
        raise NotImplementedError("CohereReranker real inference is gated behind slow tests.")
