---
purpose: "Add gated dense retrieval adapters over VectorStore"
status: "OPEN"
priority: "P2"
created: "2026-06-20"
---

# Feature: Dense Engines

## Problem
Ragfold should expose common dense embedding backends without requiring model
downloads, SaaS keys, GPU, or network in default CI.

## Proposed Solution
Add adapters for `sentence-transformers`, OpenAI, Cohere Embed, and Voyage over
a shared `VectorStore` interface. Adapters declare capabilities, gate
availability on installed extras and keys or injected test doubles, and raise a
clear `NotImplementedError` when called while unavailable.

## Affected Files
- `src/ragfold/engines/dense.py` - dense adapter classes.
- `src/ragfold/vectorstores.py` - vector-store abstraction.
- `pyproject.toml` - optional extras.

## Test Plan

### Unit / Functional Tests
- [ ] Injected dense embedder can retrieve nearest text.
- [ ] SaaS adapters list as unavailable without keys.
- [ ] Unavailable adapters do not crash router sweeps.

### Integration / E2E Tests
- [ ] Real inference tests are marked `slow`.

### Test Commands
```bash
pytest tests/engines/test_dense_engines.py -m "not slow"
```

## Edge Cases
- Dimension mismatch in vector stores.
- Missing API keys.
- Installed package without credentials.

## Out of Scope
- Network calls in the default test suite.
