---
purpose: "Represent framework retrievers as gated thin adapters"
status: "OBSOLETE"
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

## Audit (2026-11-09)
OBSOLETE. The capability-declaring stubs in `engines/frameworks.py` exist and
their unit tests pass, but the real adapters will not be built: the 1.0 goal is
for ragfold to be the retrieval library itself (see `ROADMAP.md`), and LlamaIndex
and Haystack retrievers make their own model calls, which downstream apps such as
saas-agent-starter cannot route through their gateway. A user who already has a
framework retriever can wrap it with the injected-client convention of
`GITHUB_RAG_CLIENT_ADAPTERS.md`. Reopen only on a concrete user request.
