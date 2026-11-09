---
purpose: "Optional Qdrant VectorStore over a caller-provided AsyncQdrantClient"
status: "OPEN"
priority: "P3"
created: "2026-11-09"
---

# Feature: Qdrant Store

## Problem
Some deployments keep vectors outside Postgres. The `QdrantVectorStore` stub
always raises. Postgres is the default (`PGVECTOR_STORE.md`); Qdrant is the
first optional backend, built only when a user needs it.

## Proposed Solution
- `QdrantStore(client: AsyncQdrantClient, collection: str, *, tenant_key: str | None = None)`
  implementing `VectorStore` (and `TextSearchStore` only if a sparse/full-text
  setup is configured; otherwise `Retriever` hybrid falls back to vector-only
  and records it). The caller creates the client and the collection;
  `collection_config(dimension)` returns the recommended parameters as data.
- Tenancy without RLS: when `tenant_key` is set, every call requires a filter
  value for it, and the store raises if it is missing, so a forgotten filter
  cannot read across tenants.
- Extra `qdrant = ["qdrant-client>=1.20,<2"]`. Verified 2026-11-09: qdrant-client
  1.20.0, Apache-2.0, `class AsyncQdrantClient` in `async_qdrant_client.py`.

## Affected Files
- `src/ragfold/stores/qdrant.py` - new; remove stub from `vectorstores.py`.
- `tests/stores/test_qdrant_store.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/stores/test_qdrant_store.py::test_tenant_key_is_mandatory_on_every_call`
(fake client object).
- [ ] Upsert maps `StoredChunk` to points with payload; ids are stable UUIDs from chunk ids.
- [ ] Search without the tenant filter raises when `tenant_key` is set.
- [ ] Without the package: unavailable, clear `NotImplementedError`.

### Integration / E2E Tests
- [ ] `slow`: qdrant-client local mode (`AsyncQdrantClient(location=":memory:")`) round trip.

### Test Commands
```bash
pytest tests/stores/test_qdrant_store.py -m "not slow"
```

## Edge Cases
- Chunk ids are strings; Qdrant point ids must be unsigned ints or UUIDs:
  derive UUIDv5 from the chunk id and keep the original in the payload.

## Out of Scope
- Managing Qdrant servers; Weaviate, Milvus, Chroma, FAISS.
