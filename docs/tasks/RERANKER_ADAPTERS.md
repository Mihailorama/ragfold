---
purpose: "LiteLLM and sentence-transformers CrossEncoder reranker adapters as optional extras"
status: "OPEN"
priority: "P2"
created: "2026-11-09"
---

# Feature: Reranker Adapters

## Problem
After `RERANKERS.md` there is a working async protocol but no concrete reranker.
The current `CrossEncoderReranker` and `CohereReranker` always raise.

## Proposed Solution
- `LiteLLMReranker(model: str, **litellm_kwargs)` over `litellm.arerank`
  (Cohere, Jina, Voyage, vLLM, Infinity and others). Maps `results[i].index`
  and `relevance_score` back to passages. Extra `litellm` (shared with
  `EMBEDDER_ADAPTERS.md`). Verified 2026-11-09 in the litellm 1.108.3 wheel:
  `async def arerank(model, query, documents, custom_llm_provider=None, top_n=None, ...)`
  in `rerank_api/main.py`.
- `CrossEncoderReranker(model_name: str)` real implementation:
  `CrossEncoder.rank(query, documents, top_k=...)` via `asyncio.to_thread`.
  Extra `rerank-cross-encoder` (sentence-transformers 6.2.0, Apache-2.0; `rank`
  verified in `cross_encoder/model.py`).
- `CohereReranker` removed after one deprecation release (LiteLLM covers Cohere);
  drop the `rerank-cohere` extra then.
- Not planned: an adapter for the `rerankers` library (Apache-2.0, last PyPI
  release 0.10.0 on 2025-05-22). Its `Reranker(model_name, ...)` factory
  silently resolves provider default models, which conflicts with "model always
  explicit"; LiteLLM and CrossEncoder cover the same models.

## Affected Files
- `src/ragfold/rerankers.py` (or `rerankers_adapters/`) - adapters.
- `pyproject.toml` - extras.
- `tests/test_reranker_adapters.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_reranker_adapters.py::test_litellm_reranker_maps_indices_back_to_passages`
(fake `litellm` module).
- [ ] LiteLLM results in a different order are mapped to the right passages.
- [ ] `top_n` is forwarded and respected.
- [ ] CrossEncoder adapter maps `corpus_id`/`score` from a fake model.
- [ ] Missing packages: `is_available()` False, call raises `NotImplementedError` naming the extra.

### Integration / E2E Tests
- [ ] `slow`: real `cross-encoder/ms-marco-MiniLM-L6-v2` rerank.

### Test Commands
```bash
pytest tests/test_reranker_adapters.py -m "not slow"
```

## Edge Cases
- Provider returns fewer results than passages: missing passages go last, keep order.
- Documents longer than the provider limit: caller truncates; adapter does not.

## Out of Scope
- LLM-prompted reranking.
