"""Optional corpus indexing stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ragfold.engines.base import CorpusInput


@dataclass(frozen=True)
class IndexResult:
    index_name: str
    documents_indexed: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Indexer(Protocol):
    def is_available(self) -> bool: ...

    def index(self, corpus: CorpusInput, **kwargs: Any) -> IndexResult: ...


class CocoIndexIndexer:
    """Gated adapter for CocoIndex incremental indexing flows."""

    def __init__(self, indexer: Indexer | None = None) -> None:
        self.indexer = indexer

    def is_available(self) -> bool:
        return self.indexer is not None

    def index(self, corpus: CorpusInput, **kwargs: Any) -> IndexResult:
        if self.indexer is None:
            raise NotImplementedError(
                "cocoindex is unavailable. Install ragfold[cocoindex] and inject a configured "
                "indexer before use."
            )
        return self.indexer.index(corpus, **kwargs)
