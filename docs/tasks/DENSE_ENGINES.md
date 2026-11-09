---
purpose: "Rewire dense benchmark engines onto the async Embedder and VectorStore protocols"
status: "OPEN"
priority: "P2"
created: "2026-06-20"
---

# Feature: Dense Engines

> Rescoped 2026-11-09 (starter-integration audit). The original scope (gated
> sentence-transformers, OpenAI, Cohere and Voyage engines with an injected-
> embedder path) is implemented in `engines/dense.py` and tested. Their real
> paths still raise, and each SaaS vendor has its own engine class. Embedding
> now has one async protocol (`EMBEDDERS.md`) with adapters
> (`EMBEDDER_ADAPTERS.md`), so the engines should become thin consumers of it.

## Problem
`_DenseVectorEngine` uses a sync `Embedder.encode` and the sync in-memory store,
re-embeds the whole corpus on every call, and duplicates vendor logic that the
LiteLLM adapter covers. Benchmarks (`ragfold compare`) cannot measure the same
embedder the app uses in production.

## Proposed Solution
- `DenseEngine(embedder: Embedder, *, store_factory=InMemoryVectorStore)`:
  one engine class over the async `Embedder` (`EMBEDDERS.md`) and the async
  store (`VECTOR_STORES.md`). Corpus vectors are cached per
  `(embedder.model, content_hash)` within the engine instance.
- Keep the registered names `sentence-transformers`, `openai-embed`,
  `cohere-embed`, `voyage-embed` as factories that build `DenseEngine` with the
  matching adapter, so CLI names and `DEFAULT_ENGINE_NAMES` do not change.
- `wemm` keeps its `truncate_dim` and moves to the same base.

## Affected Files
- `src/ragfold/engines/dense.py`, `engines/wemm.py`, `engines/router.py`.
- `tests/engines/test_dense_engines.py`, `tests/engines/test_wemm_engine.py`.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/engines/test_dense_engines.py::test_dense_engine_uses_async_embedder_and_caches_corpus_vectors`.
- [ ] Injected async embedder retrieves the nearest text.
- [ ] A second query over the same corpus does not re-embed documents.
- [ ] Vendor engine names still list, and list as unavailable without the extra or key.
- [ ] Unavailable engines do not crash router sweeps.

### Integration / E2E Tests
- [ ] Real inference tests are marked `slow`.

### Test Commands
```bash
pytest tests/engines/test_dense_engines.py tests/engines/test_wemm_engine.py -m "not slow"
```

## Edge Cases
- Dimension mismatch between corpus and query vectors.
- Missing API keys; installed package without credentials.

## Out of Scope
- Network calls in the default test suite.
- Persistent indexes (the engine is a benchmark adapter; production storage is
  `PGVECTOR_STORE.md`).
