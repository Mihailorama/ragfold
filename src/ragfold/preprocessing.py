"""Pre-retrieval document chunking stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    document_id: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(Protocol):
    def is_available(self) -> bool: ...

    def chunk(self, text: str, document_id: str = "doc", **kwargs: Any) -> list[Chunk]: ...


class RecursiveChunker:
    """Small no-dependency word-window chunker for core tests and examples."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def is_available(self) -> bool:
        return True

    def chunk(self, text: str, document_id: str = "doc", **kwargs: Any) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        index = 0
        while start < len(words):
            window = words[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    document_id=document_id,
                    text=" ".join(window),
                    index=index,
                    metadata={"chunk_size": len(window), "strategy": "recursive"},
                )
            )
            index += 1
            start += step
        return chunks


class AdaptiveChunker:
    """Gated adapter for `adaptive-chunking`."""

    def __init__(self, chunker: Chunker | None = None) -> None:
        self.chunker = chunker

    def is_available(self) -> bool:
        return self.chunker is not None

    def chunk(self, text: str, document_id: str = "doc", **kwargs: Any) -> list[Chunk]:
        if self.chunker is None:
            raise NotImplementedError(
                "adaptive-chunking is unavailable. Install ragfold[adaptive-chunking] "
                "and inject a configured chunker before use."
            )
        return self.chunker.chunk(text, document_id=document_id, **kwargs)
