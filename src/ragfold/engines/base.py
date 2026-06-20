"""Base public interface for ragfold retrieval engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EngineCapabilities:
    """Capability metadata exposed by every engine adapter."""

    modality: str = "text"
    retrieval_method: str = "lexical"
    ocr_free: bool = True
    local: bool = True
    saas: bool = False
    vlm: bool = False
    rerank: bool = False
    answer_generation: bool = False
    vector_store: bool = False
    requires_api_key: bool = False
    requires_gpu: bool = False
    license: str = "unknown"
    speed: str = "unknown"
    cost: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentChunk:
    """A normalized corpus entry."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedPassage:
    """A ranked passage returned by a retrieval engine."""

    document_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RagAnswer:
    """Optional extracted/generated answer attached to a retrieval result."""

    answer: str
    citations: list[str] = field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Uniform output returned by all ragfold engines."""

    engine_name: str
    query: str
    passages: list[RetrievedPassage]
    answer: RagAnswer | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0
    token_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "query": self.query,
            "passages": [passage.to_dict() for passage in self.passages],
            "answer": self.answer.to_dict() if self.answer else None,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "token_cost": self.token_cost,
        }


CorpusInput = Sequence[str | Mapping[str, Any] | DocumentChunk]


def normalize_corpus(corpus: CorpusInput) -> list[DocumentChunk]:
    """Normalize string, mapping, and `DocumentChunk` corpus entries."""

    chunks: list[DocumentChunk] = []
    for idx, item in enumerate(corpus, start=1):
        if isinstance(item, DocumentChunk):
            chunks.append(item)
        elif isinstance(item, str):
            chunks.append(DocumentChunk(id=f"doc-{idx}", text=item))
        elif isinstance(item, Mapping):
            doc_id = str(item.get("id") or item.get("document_id") or f"doc-{idx}")
            text = str(item.get("text") or item.get("content") or "")
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}
            chunks.append(DocumentChunk(id=doc_id, text=text, metadata=metadata))
        else:
            msg = f"Unsupported corpus item at position {idx}: {type(item).__name__}"
            raise TypeError(msg)
    return chunks


class RagEngine(ABC):
    """Abstract base class for retrieval/extraction adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique lowercase engine identifier."""
        ...

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities()

    @abstractmethod
    def is_available(self) -> bool:
        """Return true when dependencies, credentials, and runtime are ready."""
        ...

    def ensure_available(self) -> None:
        if not self.is_available():
            raise NotImplementedError(
                f"Engine '{self.name}' is unavailable. Install its optional extra and "
                "configure any required credentials before using it."
            )

    @abstractmethod
    async def retrieve(
        self,
        corpus: CorpusInput,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant passages for `query` from `corpus`."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} available={self.is_available()}>"
