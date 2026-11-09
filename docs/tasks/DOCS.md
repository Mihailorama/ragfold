---
purpose: "Reposition docs as a retrieval library plus benchmarks, with tested integration guides"
status: "OPEN"
priority: "P1"
created: "2026-11-09"
---

# Feature: Library Documentation

## Problem
The README presents ragfold as an engine-comparison harness. After the
`ROADMAP.md` milestones it is also a retrieval library, and a downstream app
needs to know: how to inject its own model calls, how to use pgvector inside
its own transaction and RLS, how to apply the schema through its migrations,
and what is stable. Snippets in docs drift unless tested.

## Proposed Solution
- README: lead with "retrieval library" (ingest, retrieve, cite) and keep the
  comparison/benchmark sections (the existing `test_project_docs.py`
  assertions stay).
- Guides in `docs/guides/`:
  - `injecting-models.md` - `CallableEmbedder` / `CallableReranker` over an
    app gateway; why ragfold never picks a model;
  - `postgres.md` - `PgVectorTable`, applying `create_statements` in a
    migration tool, adding a tenant column and RLS policy, running
    `ingest_document` / `Retriever` inside `BEGIN; SELECT set_config(...)`,
    `iterative_scan`;
  - `ingestion.md` - hashes, re-embedding on model change, deletions;
  - `evaluation.md` - recording embeddings and key-free CI.
- Every Python block in `README.md` and `docs/guides/*.md` that is marked
  ```` ```python test ```` is executed by `tests/test_docs_snippets.py` against
  the in-memory store and fake callables.
- `docs/api-stability.md` comes from `PUBLIC_API_AND_PACKAGING.md`.

## Affected Files
- `README.md`, `docs/guides/*.md` - new/updated.
- `tests/test_docs_snippets.py` - new; `tests/test_project_docs.py` - extend.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_project_docs.py::test_guides_exist_and_cover_postgres_rls`.
- [ ] Each guide exists and has its key headings.
- [ ] All `python test` snippets run green offline.

### Integration / E2E Tests
- [ ] None beyond the snippet runner.

### Test Commands
```bash
pytest tests/test_project_docs.py tests/test_docs_snippets.py -m "not slow"
```

## Edge Cases
- Snippets needing Postgres are marked `python postgres` and run only under the
  `postgres` marker.

## Out of Scope
- A hosted documentation site.
