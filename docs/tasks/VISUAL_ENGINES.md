---
purpose: "Add gated OCR-free visual and late-interaction adapters"
status: "DONE"
priority: "P2"
created: "2026-06-20"
---

# Feature: Visual Engines

## Problem
Ragfold should represent OCR-free and late-interaction RAG engines such as
ColPali, ColQwen2, PixelRAG, and DSE while keeping default CI light.

## Proposed Solution
Add capability-declaring adapters. ColPali supports an injected embedder path for
real local tests; all model-backed adapters are unavailable by default and raise
clear `NotImplementedError` when invoked without dependencies.

## Affected Files
- `src/ragfold/engines/visual.py` - visual/late-interaction adapters.
- `pyproject.toml` - optional extras.

## Test Plan

### Unit / Functional Tests
- [ ] Visual engines declare visual, OCR-free, late-interaction capabilities.
- [ ] Unavailable visual engines are listed and skipped in sweeps.
- [ ] ColPali injected embedder path can retrieve a passage.

### Integration / E2E Tests
- [ ] Real model inference tests are marked `slow`.

### Test Commands
```bash
pytest tests/engines/test_registry_and_stubs.py -m "not slow"
```

## Edge Cases
- No GPU.
- Missing model weights.
- Non-image corpus entries.

## Out of Scope
- Bundling model weights.

## Audit (2026-11-09)
Marked DONE in the starter-integration backlog audit: the scope above is
implemented and `pytest -m "not slow"` is green (90 passed). Evidence: `src/ragfold/engines/visual.py`; `tests/engines/test_registry_and_stubs.py::test_colpali_supports_injected_embedder_path` and the skip-in-sweeps test pass. Real model inference was never in scope and is not on the 1.0 path.
Unchecked `slow` items above stay manual; they are not tracked as open work.
