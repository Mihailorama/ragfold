---
purpose: "Make GitHub RAG adapters usable with injected clients without heavy dependencies"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
---

# Feature: GitHub RAG Client Adapters

## Problem
`LightRAGEngine`, `RAGAnythingEngine`, and `AgenticFileSearchEngine` currently
declare capabilities and degrade cleanly, but even with an injected test/client
object they still raise `NotImplementedError`. That blocks deterministic unit
tests and local adapter development without installing the real heavy packages.

## Proposed Solution
Teach the shared GitHub RAG adapter base to call an injected client using a small
adapter convention:

- Prefer `client.retrieve(corpus=..., query=..., top_k=..., **kwargs)`.
- Fall back to `client.query(...)`, then `client.search(...)`.
- Accept sync or async client methods.
- Normalize `RetrievalResult`, dict, or passage-list responses to
  `RetrievalResult`.

Real package clients remain out of scope for default CI and must still be
provided by optional extras or slow tests.

## Affected Files
- `src/ragfold/engines/github_rag.py` - injected-client call and normalization.
- `tests/engines/test_github_rag_engines.py` - behavior tests.

## Test Plan

### Unit / Functional Tests
- [ ] Injected async client result dict becomes `RetrievalResult`.
- [ ] Injected sync client passage-list result becomes `RetrievalResult`.
- [ ] Existing unavailable engines still skip compare sweeps.

### Integration / E2E Tests
- [ ] Real LightRAG/RAG-Anything/agentic-file-search tests stay `slow`.

### Test Commands
```bash
pytest tests/engines/test_github_rag_engines.py -m "not slow"
pytest -m "not slow"
```

## Edge Cases
- Client returns `RetrievalResult` directly.
- Client lacks supported method names.
- Passage dicts use `id` instead of `document_id`.
- Answer citations are omitted.

## Out of Scope
- Importing or configuring real third-party clients.
- Network calls in default CI.
