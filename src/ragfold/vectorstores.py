"""Vector store interface and lightweight/gated implementations."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from typing import Any, NoReturn, Protocol


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class VectorStore(Protocol):
    def is_available(self) -> bool: ...

    def add(self, records: list[VectorRecord]) -> None: ...

    def query(self, vector: list[float], top_k: int = 5) -> list[VectorRecord]: ...


class InMemoryVectorStore:
    """Simple cosine-similarity vector store for tests and small corpora."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def is_available(self) -> bool:
        return True

    def clear(self) -> None:
        self._records.clear()

    def add(self, records: list[VectorRecord]) -> None:
        if self._records and records:
            expected = len(self._records[0].vector)
            for record in records:
                if len(record.vector) != expected:
                    raise ValueError("Vector dimension mismatch while adding records")
        self._records.extend(records)

    def query(self, vector: list[float], top_k: int = 5) -> list[VectorRecord]:
        if not self._records:
            return []
        expected = len(self._records[0].vector)
        if len(vector) != expected:
            raise ValueError("Query vector dimension mismatch")

        scored = [
            VectorRecord(
                id=record.id,
                vector=record.vector,
                text=record.text,
                metadata=record.metadata,
                score=_cosine(vector, record.vector),
            )
            for record in self._records
        ]
        return sorted(scored, key=lambda record: record.score, reverse=True)[: max(top_k, 0)]


class _UnavailableVectorStore:
    package_name = ""
    extra_name = ""

    def is_available(self) -> bool:
        return bool(self.package_name and importlib.util.find_spec(self.package_name))

    def add(self, records: list[VectorRecord]) -> None:
        self._raise()

    def query(self, vector: list[float], top_k: int = 5) -> list[VectorRecord]:
        self._raise()

    def _raise(self) -> NoReturn:
        raise NotImplementedError(
            f"Install ragfold[{self.extra_name}] and configure the backing service before use."
        )


class FaissVectorStore(_UnavailableVectorStore):
    package_name = "faiss"
    extra_name = "faiss"


class QdrantVectorStore(_UnavailableVectorStore):
    package_name = "qdrant_client"
    extra_name = "qdrant"


class ChromaVectorStore(_UnavailableVectorStore):
    package_name = "chromadb"
    extra_name = "chroma"


class PGVectorStore(_UnavailableVectorStore):
    package_name = "pgvector"
    extra_name = "pgvector"


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
