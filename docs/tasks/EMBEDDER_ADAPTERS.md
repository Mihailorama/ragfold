---
purpose: "LiteLLM and sentence-transformers Embedder adapters as optional extras"
status: "OPEN"
priority: "P2"
created: "2026-11-09"
---

# Feature: Embedder Adapters

## Problem
Users without their own gateway need ready-made embedders. Today each SaaS
vendor has a stub engine class that always raises.

## Proposed Solution
In `src/ragfold/embeddings_adapters/`, each import lazy and gated by
`is_available()`:

- `LiteLLMEmbedder(model: str, dimension: int, **litellm_kwargs)` over
  `litellm.aembedding` (one path to OpenAI, Cohere, Voyage, Jina, vLLM, Ollama
  and others). `model` is required, no default. Extra `litellm = ["litellm>=1.108,<2"]`.
  Verified 2026-11-09: litellm 1.108.3 on PyPI, MIT (the wheel's LICENSE excludes
  only an `enterprise/` directory), `async def aembedding(*args, **kwargs)` in
  `litellm/main.py`.
- `SentenceTransformersEmbedder(model_name: str, dimension: int | None = None)`
  calling `encode_query` / `encode_document` through `asyncio.to_thread`.
  Existing extra `sentence-transformers`. Verified 2026-11-09:
  sentence-transformers 6.2.0, Apache-2.0, `encode_query` and `encode_document`
  in `sentence_transformer/model.py`; it requires torch.
- Remove the per-vendor `openai`, `cohere`, `voyage` extras after one
  deprecation release; LiteLLM replaces them.

## Affected Files
- `src/ragfold/embeddings_adapters/litellm.py`, `.../sentence_transformers.py` - new.
- `pyproject.toml` - `litellm` extra; deprecate vendor extras.
- `tests/test_embedder_adapters.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_embedder_adapters.py::test_litellm_embedder_passes_explicit_model_and_parses_response`
(monkeypatch a fake `litellm` module into `sys.modules`).
- [ ] LiteLLM adapter calls `aembedding(model=..., input=[...])` and maps
      `response.data[i]["embedding"]` in order.
- [ ] Constructing `LiteLLMEmbedder` without `model` is a `TypeError`.
- [ ] Without the package, `is_available()` is False and `embed` raises
      `NotImplementedError` naming the extra.
- [ ] sentence-transformers adapter uses `encode_query` for `input_type="query"`
      (fake model object).

### Integration / E2E Tests
- [ ] `slow`: real `sentence-transformers/all-MiniLM-L6-v2` encode.
- [ ] `slow`: LiteLLM against a provider key from the environment.

### Test Commands
```bash
pytest tests/test_embedder_adapters.py -m "not slow"
```

## Edge Cases
- Provider returns vectors out of order (LiteLLM `index` field): sort by index.
- Provider-specific `input_type` mapping (Cohere `search_query`) passed via kwargs.

## Out of Scope
- Retries, caching, cost tracking.
