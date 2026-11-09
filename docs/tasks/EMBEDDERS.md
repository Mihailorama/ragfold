---
purpose: "Async injectable Embedder protocol so model calls stay in the caller"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
plane: "RAGFOLD-5"
---

# Feature: Embedder Protocol

## Problem
Embedding today is a sync `encode(texts)` duck type inside `engines/dense.py`
and `engines/visual.py`. A downstream app that must route every model call
through its own gateway (limits, audit log, BYO keys; saas-agent-starter's
`Gateway.embed`) has no supported way to plug in an async call, and ragfold has
no notion of query vs document inputs, batch size, model id or dimension.

## Proposed Solution
New module `src/ragfold/embeddings.py`, stdlib only:

- `Embedder` `runtime_checkable` Protocol:
  - `model: str` (recorded on every stored chunk), `dimension: int`;
  - `async embed(texts: Sequence[str], *, input_type: Literal["query", "document"]) -> list[list[float]]`.
- `CallableEmbedder(fn, *, model: str, dimension: int, batch_size: int = 64, max_concurrency: int = 1)`:
  wraps any `async (texts, input_type) -> list[list[float]]` callable. It splits
  input into batches, runs at most `max_concurrency` batches at a time, and
  validates that every batch returns one vector per input of length `dimension`.
- `model` and `dimension` are required arguments: ragfold never picks a model.
- No retries, caching or rate limiting: those belong to the caller's gateway.

## Affected Files
- `src/ragfold/embeddings.py` - new.
- `src/ragfold/__init__.py` - export `Embedder`, `CallableEmbedder`.
- `tests/test_embeddings.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_embeddings.py::test_callable_embedder_batches_and_preserves_order`.
- [ ] 130 texts with `batch_size=64` -> 3 calls, output order matches input.
- [ ] `input_type` is passed through to the callable.
- [ ] Wrong vector count or wrong dimension raises `ValueError` with counts only, no text.
- [ ] `max_concurrency=2` never has more than 2 calls in flight.
- [ ] Empty input returns `[]` without calling `fn`.
- [ ] `CallableEmbedder` satisfies `isinstance(x, Embedder)`.

### Integration / E2E Tests
- [ ] None (pure stdlib; real models in `EMBEDDER_ADAPTERS.md`).

### Test Commands
```bash
pytest tests/test_embeddings.py -m "not slow"
```

## Edge Cases
- Empty strings are passed through; the callable decides.
- The callable raises: exception propagates unchanged.
- Matryoshka truncation is the callable's job (or `wemm`'s `truncate_dim`).

## Out of Scope
- Concrete providers (`EMBEDDER_ADAPTERS.md`).
- Sparse and multi-vector embeddings.
