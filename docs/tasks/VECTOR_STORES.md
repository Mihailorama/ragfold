---
purpose: "Async VectorStore protocol for stored chunks, with an in-memory reference store"
status: "OPEN"
priority: "P0"
created: "2026-06-20"
---

# Feature: Vector Stores

> Rescoped 2026-11-09 (starter-integration audit). The original scope was a sync
> `VectorStore` plus gated FAISS, Qdrant, Chroma and pgvector stubs. The stubs
> exist and raise `NotImplementedError`. Downstream apps (saas-agent-starter)
> need a real, async, Postgres-first store that runs inside the caller's
> transaction. This task now defines only the protocol and the in-memory
> reference store. pgvector lives in `PGVECTOR_SCHEMA.md`, `PGVECTOR_STORE.md`
> and `PGVECTOR_HYBRID_SEARCH.md`; Qdrant in `QDRANT_STORE.md`.

## Problem
`VectorStore` today is sync (`add`, `query`), keeps only `id/vector/text`, has
no delete, no filters, no content hash and no embedding-model field. An app
cannot do idempotent re-ingestion, delete a document, scope a search, or detect
that chunks were embedded with another model. Every store adapter is a stub.

## Proposed Solution
New module `src/ragfold/stores/` (keep `ragfold.vectorstores` as a re-export
with a `DeprecationWarning` for one minor release).

- `StoredChunk` dataclass: `id`, `document_id`, `chunk_index`, `text`,
  `vector: list[float]`, `content_hash`, `embedding_model`,
  `metadata: dict[str, Any]`.
- `ScoredChunk`: `StoredChunk` fields without `vector`, plus `score` and
  `scores: dict[str, float]` (per-method breakdown, e.g. `vector`, `text`, `rrf`).
- `VectorStore` `typing.Protocol`, all methods `async`:
  - `upsert(chunks: Sequence[StoredChunk]) -> None`
  - `delete_documents(document_ids: Sequence[str]) -> int`
  - `delete_chunks(chunk_ids: Sequence[str]) -> int`
  - `chunk_hashes(document_id: str) -> dict[str, tuple[str, str]]`
    (chunk id -> `(content_hash, embedding_model)`; used by ingestion)
  - `search_vector(vector, *, top_k, filter: Mapping[str, Any] | None) -> list[ScoredChunk]`
- Optional protocol `TextSearchStore` with `search_text(query, *, top_k, filter)`
  and `HybridSearchStore` with `search_hybrid(query, vector, *, top_k, k=60, filter)`.
  `Retriever` (`RETRIEVER.md`) checks for these with `isinstance` on
  `runtime_checkable` protocols and falls back to `ragfold.fusion` when a store
  has vector and text search but no native hybrid.
- `filter` is equality on metadata keys only (`{"source": "upload"}`). No query
  language. Tenancy is the caller's job (RLS, or a filter key), never a store
  default.
- `InMemoryVectorStore` implements all three protocols (text search reuses the
  BM25 scorer from `engines/bm25.py`) and is the CI reference implementation.
- Remove the `FaissVectorStore` and `ChromaVectorStore` stubs after one
  deprecation release: no proposal plans real adapters and they mislead users
  into thinking support exists.

## Affected Files
- `src/ragfold/stores/__init__.py`, `stores/base.py`, `stores/memory.py` - new.
- `src/ragfold/vectorstores.py` - deprecated re-exports.
- `src/ragfold/engines/dense.py`, `engines/wemm.py`, `engines/visual.py` - move to
  the async store (see `DENSE_ENGINES.md`).
- `tests/stores/test_memory_store.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/stores/test_memory_store.py::test_upsert_replaces_chunk_with_same_id`.
- [ ] Upsert of an existing id replaces text, vector, hash and model.
- [ ] `delete_documents` removes every chunk of the document and returns the count.
- [ ] `chunk_hashes` returns `(content_hash, embedding_model)` per chunk id.
- [ ] `search_vector` orders by cosine similarity and respects `top_k` and `filter`.
- [ ] `search_text` ranks a keyword match first; `search_hybrid` fuses with the
      same scores as `ragfold.fusion.reciprocal_rank_fusion` (k=60).
- [ ] Dimension mismatch on upsert or search raises `ValueError` whose message
      contains the dimensions but no chunk text.
- [ ] `InMemoryVectorStore` satisfies `isinstance(store, VectorStore)` and the
      two optional protocols.
- [ ] Importing `ragfold.vectorstores` emits `DeprecationWarning`.

### Integration / E2E Tests
- [ ] Existing dense/wemm/visual engine tests stay green on the new store.

### Test Commands
```bash
pytest tests/stores -m "not slow"
pytest -m "not slow"
```

## Edge Cases
- Empty store; `top_k <= 0`; empty `filter`.
- Chunks of mixed embedding models in one store: search does not filter by
  model; ingestion re-embeds (see `INGESTION_PIPELINE.md`).
- Zero vector (cosine undefined) scores 0.0.

## Out of Scope
- pgvector, Qdrant (own proposals).
- FAISS and Chroma adapters.
- Sparse vectors, multi-vector (late-interaction) storage.
