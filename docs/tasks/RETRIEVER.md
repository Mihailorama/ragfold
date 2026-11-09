---
purpose: "Production Retriever: embed query, store search (vector/text/hybrid), optional rerank"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
plane: "RAGFOLD-18"
---

# Feature: Retriever

## Problem
`EngineRouter` and the engines re-index a corpus passed in on every call; that
is right for benchmarks but not for an app with a persistent index. There is no
entry point that takes a query and returns passages from a store.

## Proposed Solution
New module `src/ragfold/retriever.py`:

```python
retriever = Retriever(embedder=..., store=..., reranker=None,
                      mode="hybrid", candidates=40)
result = await retriever.search(query, top_k=8, filter=None)  # RetrievalResult
```

- `mode="vector"|"text"|"hybrid"`. Hybrid uses `store.search_hybrid` when the
  store implements `HybridSearchStore`; otherwise it runs vector and text search
  concurrently and fuses with `ragfold.fusion.reciprocal_rank_fusion`. `"text"`
  needs no embedder.
- Embeds the query once with `input_type="query"`.
- Optional async reranker (`RERANKERS.md`) over `candidates`, then `top_k`.
- Returns the public `RetrievalResult` (`engine_name="retriever(<mode>)"`);
  each `RetrievedPassage` carries `chunk_id`, `document_id`, `start`, `end`,
  `heading_path`, `scores` and `embedding_model` in `metadata`, which citations
  use (`CITATIONS.md`). `processing_time_ms` is set; `token_cost` stays 0
  (costs belong to the caller's gateway).
- `StoreEngine(retriever)`: a `RagEngine` adapter, so a persistent index can be
  compared with other engines in `ragfold compare` / `bench`. It ignores the
  `corpus` argument and records that in metadata.

## Affected Files
- `src/ragfold/retriever.py` - new.
- `src/ragfold/__init__.py` - exports.
- `tests/test_retriever.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_retriever.py::test_hybrid_retriever_returns_keyword_and_semantic_hits`.
- [ ] Hybrid over `InMemoryVectorStore` returns both a keyword-only and a semantic-only match.
- [ ] Store with native `search_hybrid` is called once; without it, two calls + fusion.
- [ ] Query is embedded exactly once with `input_type="query"`; `mode="text"` embeds nothing.
- [ ] Reranker sees `candidates` passages and the result has `top_k`.
- [ ] Passage metadata carries `chunk_id`, offsets and `scores`.
- [ ] `filter` is forwarded to the store.
- [ ] `StoreEngine` works in `EngineRouter.compare`.

### Integration / E2E Tests
- [ ] `postgres`: `Retriever` over `PgVectorStore` inside an RLS-scoped transaction.

### Test Commands
```bash
pytest tests/test_retriever.py -m "not slow"
```

## Edge Cases
- Empty query string: return an empty result without calling the embedder.
- Store returns fewer than `top_k`.
- Embedder dimension differs from stored vectors: store raises; propagate.

## Out of Scope
- Answer generation (the app calls its own model; `CITATIONS.md` prepares context).
- Query rewriting / multi-query expansion (needs model calls; a caller hook later).
