---
purpose: "Postgres full-text plus pgvector hybrid search with Reciprocal Rank Fusion in one SQL query"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
plane: "RAGFOLD-9"
---

# Feature: pgvector Hybrid Search

## Problem
Vector search alone misses exact terms (product codes, names, error strings);
keyword search alone misses paraphrases. The pgvector README ("Hybrid Search")
recommends combining it with Postgres full-text search and merging with
Reciprocal Rank Fusion, which ragfold already implements in Python
(`ragfold.fusion`). Doing the fusion in SQL keeps it to one round trip inside
the caller's transaction and RLS scope.

## Proposed Solution
Extend `PgVectorStore` (`PGVECTOR_STORE.md`) with the `TextSearchStore` and
`HybridSearchStore` protocols from `VECTOR_STORES.md`:

- `search_text(query, *, top_k, filter)`: `tsv @@ websearch_to_tsquery('<config>', $1)`,
  ordered by `ts_rank_cd(tsv, query)`.
- `search_hybrid(query, vector, *, top_k, k=60, candidates=40, weights=(1.0, 1.0), filter)`:
  one statement with two CTEs, each ranked with `ROW_NUMBER()` and limited to
  `candidates`, then `FULL OUTER JOIN` on `id` and
  `score = w_v / (k + vector_rank) + w_t / (k + text_rank)` (missing rank adds 0).
  Same formula, `k` default and weights semantics as
  `ragfold.fusion.reciprocal_rank_fusion`, so Python and SQL fusion agree.
  Each `ScoredChunk.scores` carries `vector_rank`, `text_rank`, `rrf`.
- The `filter` and `SET LOCAL` rules from `PGVECTOR_STORE.md` apply to both CTEs.

## Affected Files
- `src/ragfold/stores/pgvector.py` - two methods.
- `tests/stores/test_pgvector_store.py`, `tests/stores/test_pgvector_store_pg.py`.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/stores/test_pgvector_store.py::test_hybrid_sql_uses_rrf_with_k_and_weights_as_parameters`.
- [ ] Hybrid SQL contains both CTEs, `websearch_to_tsquery`, `<=>`, and binds
      `k`, weights and `candidates` as parameters.
- [ ] Text search uses the table's `text_search_config`.
- [ ] Filters appear in both CTEs.

### Integration / E2E Tests
- [ ] `postgres`: a fixture corpus where a keyword-only document and a
      semantic-only document both appear in hybrid top-k but not in the other
      single-method top-k.
- [ ] `postgres`: SQL fused order equals `reciprocal_rank_fusion` over the two
      single-method results for the same fixture (parity test).
- [ ] `postgres`: RLS-scoped hybrid search returns only the current org's rows.

### Test Commands
```bash
pytest tests/stores -m "not slow"
RAGFOLD_TEST_PG_DSN=postgresql://... pytest tests/stores -m postgres
```

## Edge Cases
- Query with only stop words or punctuation: text CTE is empty, result equals
  vector ranking.
- Ties in fused score: order by `(score desc, consensus desc, min rank, id)`, the
  same tie-break as `ragfold.fusion`.
- Languages that `simple` handles poorly (no stemming): document; per-document
  regconfig is a follow-up.

## Out of Scope
- True BM25 in Postgres (`pg_textsearch`, ParadeDB) - only behind an eval that
  shows a gain; ParadeDB `pg_search` is AGPL-3.0.
- Reranking (the `Retriever` applies it after this stage).
