---
purpose: "Add a real optional LightRAG runtime adapter behind the existing gated engine"
status: "DONE"
priority: "P1"
created: "2026-06-21"
---

# Feature: LightRAG Real Adapter

## Problem
`LightRAGEngine` currently works as a safe injected-client adapter, which is
useful for tests and downstream wrappers, but it does not yet configure a real
`lightrag-hku` runtime. Users who install the optional extra should be able to
instantiate a configured LightRAG backend without leaving the `RagEngine` public
surface.

## Proposed Solution
Keep the default engine unavailable in light CI, but let `LightRAGEngine` build
a lazy runtime client when callers provide a LightRAG `working_dir`,
`llm_model_func`, and `embedding_func`. The runtime client should:

- lazily import or accept injected `LightRAG` / `QueryParam` classes;
- initialize LightRAG storage before first use;
- insert the provided ragfold corpus with stable IDs;
- query LightRAG in context mode by default and normalize that context into a
  `RetrievalResult`;
- optionally attach an answer when `generate_answer=True`;
- raise a clear `NotImplementedError` when runtime configuration is incomplete.

## Affected Files
- `src/ragfold/engines/github_rag.py` - LightRAG runtime client and normalization.
- `tests/engines/test_github_rag_engines.py` - RED/GREEN coverage for the new
  runtime path.
- `docs/tasks/LIGHTRAG_REAL_ADAPTER.md` - proposal and test plan.

## Test Plan

### Unit / Functional Tests
- [x] A configured LightRAG runtime client initializes storage, inserts the
      normalized corpus, queries with `QueryParam`, and returns a
      `RetrievalResult`.
- [x] A configured runtime can attach a generated answer when requested.
- [x] Incomplete runtime configuration raises a clear `NotImplementedError` and
      keeps the engine unavailable.
- [x] Existing injected-client behavior remains unchanged.

### Integration / E2E Tests
- [ ] Real `lightrag-hku` inference/download/provider tests remain `slow` and
      outside the default CI path.

### Test Commands
```bash
pytest tests/engines/test_github_rag_engines.py -m "not slow"
pytest -m "not slow"
ruff check src tests
```

## Edge Cases
- LightRAG package installed but no model functions configured.
- LightRAG returns context text instead of structured passages.
- Async and sync LightRAG methods both remain acceptable where possible.
- Existing injected clients continue to return dict/list/`RetrievalResult`
  shapes.

## Out of Scope
- Downloading model weights or calling SaaS providers in default CI.
- Vendoring LightRAG or managing long-lived shared LightRAG indexes.
- Replacing ragfold's public `RagEngine` contract.
