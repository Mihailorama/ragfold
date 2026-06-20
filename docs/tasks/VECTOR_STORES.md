---
purpose: "Add vector-store abstraction and gated store adapters"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
---

# Feature: Vector Stores

## Problem
Dense engines need a pluggable storage interface while default CI must avoid
native services and network.

## Proposed Solution
Add `VectorStore`, real `InMemoryVectorStore`, and gated adapters for FAISS,
Qdrant, Chroma, and pgvector. Non-core stores list availability and raise clear
`NotImplementedError` when dependencies are absent.

## Affected Files
- `src/ragfold/vectorstores.py` - store interface and adapters.
- `pyproject.toml` - optional extras.

## Test Plan

### Unit / Functional Tests
- [ ] In-memory store returns cosine-nearest records.
- [ ] Optional stores report unavailable without installed dependencies.
- [ ] Optional stores do not crash construction.

### Integration / E2E Tests
- [ ] Service-backed tests are marked `slow`.

### Test Commands
```bash
pytest tests/engines/test_vectorstores.py -m "not slow"
```

## Edge Cases
- Empty index.
- Dimension mismatch.
- Missing service URL.

## Out of Scope
- Managing external vector-store services.
