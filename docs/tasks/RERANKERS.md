---
purpose: "Add optional reranking stage adapters"
status: "OPEN"
priority: "P2"
created: "2026-06-20"
---

# Feature: Rerankers

## Problem
Retrieval sweeps need to model optional reranking without forcing
cross-encoder or SaaS dependencies into CI.

## Proposed Solution
Add a small reranker interface plus cross-encoder and Cohere Rerank adapters.
The router can apply an available reranker after first-stage retrieval.

## Affected Files
- `src/ragfold/rerankers.py` - reranker interface and adapters.
- `src/ragfold/engines/router.py` - optional reranker hook.
- `pyproject.toml` - optional extras.

## Test Plan

### Unit / Functional Tests
- [ ] Injected reranker can reorder passages.
- [ ] Unavailable rerankers raise clear `NotImplementedError`.

### Integration / E2E Tests
- [ ] Real cross-encoder and Cohere tests are marked `slow`.

### Test Commands
```bash
pytest tests/engines/test_router.py -m "not slow"
```

## Edge Cases
- Empty retrieved passages.
- Reranker returns fewer scores than passages.

## Out of Scope
- Training rerankers.
