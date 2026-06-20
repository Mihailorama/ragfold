"""Post-retrieval context compression stages."""

from __future__ import annotations

from typing import Any, Protocol

from ragfold.engines.base import RetrievedPassage


class ContextCompressor(Protocol):
    def is_available(self) -> bool: ...

    def compress(
        self,
        passages: list[RetrievedPassage],
        query: str,
        **kwargs: Any,
    ) -> list[RetrievedPassage]: ...


class NoopCompressor:
    def is_available(self) -> bool:
        return True

    def compress(
        self,
        passages: list[RetrievedPassage],
        query: str,
        **kwargs: Any,
    ) -> list[RetrievedPassage]:
        return passages


class HeadroomCompressor:
    """Gated adapter for headroom compression."""

    def __init__(self, compressor: ContextCompressor | None = None) -> None:
        self.compressor = compressor

    def is_available(self) -> bool:
        return self.compressor is not None

    def compress(
        self,
        passages: list[RetrievedPassage],
        query: str,
        **kwargs: Any,
    ) -> list[RetrievedPassage]:
        if self.compressor is None:
            raise NotImplementedError(
                "headroom is unavailable. Install ragfold[headroom] and inject a configured "
                "compressor before use."
            )
        return self.compressor.compress(passages, query=query, **kwargs)
