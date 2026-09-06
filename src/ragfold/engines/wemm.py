"""WeMM-Embedding adapter — Tencent/WeChat unified multimodal dense embedder.

WeMM-Embedding (arXiv 2608.24053, Apache-2.0) puts text, images, video and visual
documents in one vector space and ships Matryoshka Representation Learning, so a
single model produces nested dimensions that can be truncated at inference without
retraining. This adapter reuses the dense VectorStore retrieval path and adds an
optional ``truncate_dim`` for that Matryoshka behavior. Real weight loading is
gated: inject a loaded model in tests, or install the ``wemm`` extra and provide
one; real inference belongs behind ``@pytest.mark.slow``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ragfold.engines.base import CorpusInput, EngineCapabilities, RetrievalResult
from ragfold.engines.dense import Embedder, _DenseVectorEngine
from ragfold.vectorstores import VectorStore


class _TruncatingEmbedder:
    """Wraps an embedder and slices each vector to ``dim`` (Matryoshka MRL)."""

    def __init__(self, inner: Embedder, dim: int | None) -> None:
        self._inner = inner
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._inner.encode(texts)
        if self._dim is None:
            return [list(vector) for vector in vectors]
        return [list(vector)[: self._dim] for vector in vectors]


class WeMMEmbeddingEngine(_DenseVectorEngine):
    """WeMM-Embedding: local multimodal dense retrieval with optional MRL truncation."""

    engine_name = "wemm"

    def __init__(
        self,
        model: Embedder | None = None,
        vector_store: VectorStore | None = None,
        truncate_dim: int | None = None,
    ) -> None:
        super().__init__(model=model, vector_store=vector_store)
        self.truncate_dim = truncate_dim

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            modality="multimodal",
            retrieval_method="dense",
            ocr_free=True,
            local=True,
            vlm=True,
            vector_store=True,
            requires_gpu=True,
            license="Apache-2.0",
            speed="slow",
            cost="free",
        )

    def is_available(self) -> bool:
        if self.model is not None:
            return True
        find_spec = importlib.util.find_spec
        return bool(find_spec("torch") and find_spec("transformers"))

    def _get_model(self) -> Embedder:
        if self.model is None:
            raise NotImplementedError(
                "Engine 'wemm' unavailable: install ragfold[wemm] and inject or load a "
                "WeMM-Embedding model (e.g. tencent/WeMM-Embedding-2B)."
            )
        return _TruncatingEmbedder(self.model, self.truncate_dim)

    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        result = await super().retrieve(corpus, query, top_k=top_k, **kwargs)
        result.metadata["method"] = "dense-multimodal"
        result.metadata["truncate_dim"] = self.truncate_dim
        return result
