---
purpose: "Wire chunking and compression lifecycle stages into EngineRouter"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
---

# Feature: Router Pipeline Stages

## Problem
`ragfold` now has chunking and compression stage interfaces, but callers still
have to apply them manually. That makes the lifecycle design theoretical and
prevents consistent compare/batch behavior.

## Proposed Solution
Extend `EngineRouter` with optional `chunker` and `compressor` hooks:

- If a chunker is configured and available, normalize the incoming corpus and
  replace each source document with chunk records before engine retrieval.
- Preserve source document metadata on chunk records.
- If a compressor is configured and available, apply it after retrieval and
  reranking.
- Mark applied stages in `RetrievalResult.metadata`.
- Use the same path for `retrieve()`, `compare()`, and `process_batch()`.

## Affected Files
- `src/ragfold/engines/router.py` - stage hooks.
- `tests/engines/test_router.py` - behavior tests.

## Test Plan

### Unit / Functional Tests
- [ ] Router passes chunked corpus to the selected engine.
- [ ] Router applies compressor after retrieval.
- [ ] `compare()` uses the same pipeline path as `retrieve()`.

### Integration / E2E Tests
- [ ] Existing `ragfold compare examples/...` remains green.

### Test Commands
```bash
pytest tests/engines/test_router.py -m "not slow"
pytest -m "not slow"
```

## Edge Cases
- Chunker configured but unavailable.
- Compressor configured but unavailable.
- Empty corpus.
- Corpus entries with metadata.

## Out of Scope
- Async chunkers/compressors.
- Indexer integration.
