---
purpose: "Async injectable Reranker protocol and an awaited rerank stage"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
plane: "RAGFOLD-16"
---

# Feature: Rerankers

> Rescoped 2026-11-09 (starter-integration audit). The original scope was a sync
> `BaseReranker` with cross-encoder and Cohere stubs. Those stubs exist and
> always raise. Downstream apps must own every model call (saas-agent-starter
> routes rerank through its `Gateway.rerank`, which enforces limits and writes
> the audit log), so the reranker must be an injectable async callable. Concrete
> adapters move to `RERANKER_ADAPTERS.md`.

## Problem
`BaseReranker.rerank` is sync, so an async model call cannot be used without
blocking the event loop. Neither stub can actually rerank, and the router
cannot apply a reranker that lives outside ragfold.

## Proposed Solution
- `ragfold.rerankers.Reranker` `runtime_checkable` Protocol:
  `async rerank(query: str, passages: Sequence[RetrievedPassage], *, top_n: int | None = None) -> list[RetrievedPassage]`.
- `CallableReranker(score_fn, *, model: str)`: wraps any
  `async (query: str, documents: list[str]) -> list[float]` callable, for example
  a gateway method. It calls `score_fn` once, validates one score per passage,
  sorts by score (stable on ties), re-numbers `rank`, keeps the first-stage
  score in `metadata["first_stage_score"]` and records `model` in
  `metadata["reranker"]`.
- `EngineRouter` awaits the reranker in `_retrieve_with_engine` and
  `retrieve_hybrid`. A sync `BaseReranker` still works for one deprecation
  release (run via `asyncio.to_thread`, with a `DeprecationWarning`).
- Mark `CrossEncoderReranker` and `CohereReranker` deprecated here; their
  replacements are specified in `RERANKER_ADAPTERS.md`.

## Affected Files
- `src/ragfold/rerankers.py` - protocol, `CallableReranker`, deprecations.
- `src/ragfold/engines/router.py` - awaited rerank stage.
- `tests/test_rerankers.py` - new; `tests/engines/test_router.py` - update.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_rerankers.py::test_callable_reranker_orders_by_injected_scores`.
- [ ] `CallableReranker` reorders passages by the injected scores and re-ranks 1..n.
- [ ] `top_n` truncates after sorting.
- [ ] A score list of the wrong length raises `ValueError` (message has counts only).
- [ ] Empty passages: `score_fn` is not called.
- [ ] Router applies an async reranker in `retrieve()` and `retrieve_hybrid()`.
- [ ] A legacy sync `BaseReranker` still works and warns.

### Integration / E2E Tests
- [ ] None in default CI; real models are in `RERANKER_ADAPTERS.md` (slow).

### Test Commands
```bash
pytest tests/test_rerankers.py tests/engines/test_router.py -m "not slow"
```

## Edge Cases
- Equal scores keep first-stage order.
- `score_fn` raises: the exception propagates unchanged; ragfold adds no text.
- NaN scores sort last.

## Out of Scope
- Concrete model adapters (`RERANKER_ADAPTERS.md`).
- Training rerankers; LLM-as-reranker prompts.
