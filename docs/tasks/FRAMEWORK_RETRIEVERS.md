---
purpose: "Represent framework retrievers as gated thin adapters"
status: "OPEN"
priority: "P3"
created: "2026-06-20"
---

# Feature: Framework Retrievers

## Problem
Users may already have retrievers in LlamaIndex, Haystack, or txtai. Ragfold
should expose their availability and provide a future integration point without
adding those frameworks to core CI.

## Proposed Solution
Add capability-declaring thin adapters that are unavailable unless the relevant
extra is installed and an adapter object is injected. Without an injected
retriever, calls raise clear `NotImplementedError`.

## Affected Files
- `src/ragfold/engines/frameworks.py` - LlamaIndex, Haystack, txtai adapters.
- `pyproject.toml` - optional extras.

## Test Plan

### Unit / Functional Tests
- [ ] Adapters declare framework retrieval capabilities.
- [ ] Missing frameworks list as unavailable and are skipped in sweeps.

### Integration / E2E Tests
- [ ] Real framework tests are marked `slow`.

### Test Commands
```bash
pytest tests/engines/test_registry_and_stubs.py -m "not slow"
```

## Edge Cases
- Framework installed but no retriever instance provided.
- Framework-specific result shape differences.

## Out of Scope
- Deep framework indexing abstractions.
