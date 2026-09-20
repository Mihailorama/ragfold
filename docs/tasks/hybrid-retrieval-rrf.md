---
purpose: "Add cross-engine hybrid retrieval via Reciprocal Rank Fusion (RRF)"
status: "OPEN"
priority: "P1"
created: "2026-09-20"
---

# Feature: Cross-engine hybrid retrieval with Reciprocal Rank Fusion

## Problem
Ragfold cannot combine the rankings of two or more engines. `retrieve()` runs
exactly one engine (`router.py` `select()` returns a single engine), `compare()`
returns a per-engine dict that is never merged, and the reranker only rescores a
single engine's passages. Running BM25 alongside a dense engine and merging their
rankings - the standard way to beat either alone on mixed keyword/semantic
queries - is impossible today.

Reciprocal Rank Fusion (RRF) is the canonical fix: deterministic, dependency-free
(pure arithmetic), and score-scale agnostic (it fuses ranks, not raw scores, so a
BM25 score and a cosine similarity never need calibration).

## Proposed Solution
Two additions, no mutation of the public contract.

1. **Pure function** `reciprocal_rank_fusion(rankings, *, k=60)` in a new module
   `src/ragfold/fusion.py`.
   - Input: several ranked passage lists, either as a positional
     `Sequence[Sequence[RetrievedPassage]]` (engines auto-named `engine-0`,
     `engine-1`, ...) or a `Mapping[str, Sequence[RetrievedPassage]]` (keys are
     engine names).
   - Fuse by `document_id`. Fused score = sum over engines of `1 / (k + rank)`,
     where `rank` is the 1-based position of the passage in that engine's list.
   - Deterministic tie-break: sort key `(-score, -consensus, min_rank,
     document_id)`.
   - Each fused passage carries an explainable breakdown under
     `metadata["fusion"]`: `method`, `k`, fused `score`, `consensus` (how many
     engines ranked it), and `contributions` (per engine: name, rank, the
     `1/(k+rank)` term), sorted by rank then engine name. The representative
     `text`/base metadata is taken from the occurrence with the best (lowest)
     rank, ties broken by engine order.
   - Returns the full fused `list[RetrievedPassage]` with recomputed 1-based
     `rank`. Truncation to `top_k` is the caller's job.
   - Stdlib only. No engines involved, no I/O.

2. **Router path** `async retrieve_hybrid(corpus, query, *, engines: list[str],
   top_k=5, k=60, **kwargs) -> RetrievalResult`, sitting ABOVE `select()`.
   - Resolves each named engine (unknown name -> `ValueError`), skips
     unavailable ones (recorded in metadata), raises if none are available.
   - Prepares the corpus ONCE via the existing `_prepare_corpus` so every engine
     sees identical `document_id`s (RRF dedups by id).
   - Runs the engines concurrently, reusing the `asyncio.gather` + bounded
     `Semaphore` shape of `process_batch`.
   - Fuses with `reciprocal_rank_fusion`, truncates to `top_k`, then applies any
     configured reranker/compressor exactly as `_retrieve_with_engine` does.
   - `engine_name` marks the fusion, e.g. `rrf(bm25,text-rag)`. `metadata`
     carries the fusion breakdown (k, engines run, engines skipped, per-engine
     passage counts). `token_cost` sums the engines' costs.

Also export `reciprocal_rank_fusion` from `ragfold/__init__.py` for
discoverability.

## Affected Files
- `src/ragfold/fusion.py` - new pure RRF module.
- `src/ragfold/engines/router.py` - add `retrieve_hybrid` (add, do not mutate).
- `src/ragfold/__init__.py` - export `reciprocal_rank_fusion`.
- `tests/test_fusion.py` - new standalone unit tests with hand-computed scores.
- `tests/engines/test_router.py` - hybrid router tests.
- `README.md` - "How to Choose" row for hybrid/RRF.
- `CHANGELOG.md` - Unreleased/Added entry.

## Test Plan

### Unit / Functional Tests
- [ ] Two-engine fusion with hand-computed scores: for `k=60`, an item ranked #1
      by one engine and #2 by the other scores `1/61 + 1/62`; verify exact
      float and consensus=2.
- [ ] Single-list fusion reduces to `1/(k+rank)` per item, order preserved.
- [ ] Dedup by `document_id`: same id from two engines becomes one fused passage
      with consensus=2 and two contributions.
- [ ] `k` changes the score as expected (hand-computed for a non-default `k`).
- [ ] Deterministic tie-break: two items with equal fused score order by
      `(-consensus, min_rank, document_id)`.
- [ ] Empty input -> empty list; a single empty ranking -> empty list.
- [ ] Mapping input labels contributions with the engine names.
- [ ] `contributions` breakdown records each engine's rank and term.

### Integration / E2E Tests
- [ ] `retrieve_hybrid(["bm25","text-rag"])` returns a `RetrievalResult` whose
      `engine_name == "rrf(bm25,text-rag)"` and whose passages carry the fusion
      breakdown; a doc both engines rank highly wins.
- [ ] Unknown engine name raises `ValueError`.
- [ ] Unavailable engine is skipped and recorded; if one of two is unavailable,
      fusion still runs on the available one.
- [ ] Configured reranker/compressor still apply after fusion.
- [ ] Corpus is prepared once (chunker applied) so ids align across engines.

### Test Commands
```bash
pytest tests/ -m "not slow"
ruff check .
mypy src/ragfold/fusion.py src/ragfold/engines/router.py
```

## Edge Cases
- Empty corpus or empty query: engines return no passages -> fused result is
  empty, no crash.
- One engine returns fewer than `top_k` passages: fusion handles ragged lists.
- Same `document_id` appearing twice within one engine's list: first (best-rank)
  occurrence wins for that engine; it is not double-counted.
- All named engines unavailable: `ValueError`.

## Out of Scope
- CLI surface for hybrid retrieval (router API only for now).
- Weighted/learned fusion; per-engine weights.
- Changing `RagEngine` / `RetrievalResult` / `RagAnswer` signatures.
- Adding any dependency (RRF is stdlib arithmetic).
