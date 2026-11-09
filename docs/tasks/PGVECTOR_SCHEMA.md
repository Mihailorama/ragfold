---
purpose: "Export pgvector table/index DDL as data the app applies through its own migrations"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
plane: "RAGFOLD-10"
---

# Feature: pgvector Schema Export

## Problem
A pgvector store needs a table with a vector column, a full-text column and two
indexes. Apps with their own migration tool (saas-agent-starter uses Drizzle
migrations, RLS policies and expand/contract rules) cannot let a library run
`CREATE TABLE` at runtime. They also add their own columns (`org_id`) and
policies. ragfold must describe the schema, not apply it.

## Proposed Solution
New module `src/ragfold/stores/pgvector_schema.py`, stdlib only, no I/O:

- `PgVectorTable` frozen dataclass, the single description used by both the DDL
  and the store (`PGVECTOR_STORE.md`):
  `name` (schema-qualified allowed), `dimension`, column names (`id`,
  `document_id`, `chunk_index`, `text`, `embedding`, `tsv`, `content_hash`,
  `embedding_model`, `metadata`) with defaults, `text_search_config="simple"`,
  `distance="cosine"` (`vector_cosine_ops`, operator `<=>`).
  Identifiers are validated against `^[A-Za-z_][A-Za-z0-9_]*$` (and one dot for
  the schema), so they can be inlined safely.
- `create_statements(table, *, extra_columns: Sequence[str] = (), create_extension: bool = False) -> list[str]`:
  `CREATE TABLE IF NOT EXISTS` with
  `tsv tsvector GENERATED ALWAYS AS (to_tsvector('<config>'::regconfig, text)) STORED`,
  an HNSW index (`USING hnsw (embedding vector_cosine_ops)`), a GIN index on
  `tsv`, a btree on `document_id`, and `UNIQUE (document_id, chunk_index)`.
  `extra_columns` are raw column definitions the app owns (e.g.
  `"org_id text NOT NULL"`); ragfold never emits RLS policies.
- `drop_statements(table)` for tests.
- An app that defines the table in its own ORM only needs `PgVectorTable` with
  its column names; the DDL function is a convenience and a reference.

## Affected Files
- `src/ragfold/stores/pgvector_schema.py` - new.
- `tests/stores/test_pgvector_schema.py` - new (golden SQL).

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/stores/test_pgvector_schema.py::test_create_statements_golden_sql`.
- [ ] Default table renders the exact expected SQL (golden string).
- [ ] Custom column names and `text_search_config="english"` appear in DDL.
- [ ] `extra_columns` are inserted verbatim before the constraints.
- [ ] Invalid identifiers (`"chunks; drop"`) raise `ValueError`.
- [ ] `dimension <= 0` raises `ValueError`.
- [ ] `create_extension=True` adds `CREATE EXTENSION IF NOT EXISTS vector` first.

### Integration / E2E Tests
- [ ] `slow` + `postgres` marker: apply the statements to a `pgvector/pgvector`
      Postgres (DSN from `RAGFOLD_TEST_PG_DSN`), re-apply (idempotent), drop.

### Test Commands
```bash
pytest tests/stores/test_pgvector_schema.py -m "not slow"
RAGFOLD_TEST_PG_DSN=postgresql://... pytest -m postgres
```

## Edge Cases
- `halfvec` for dimensions above the 2,000-dimension HNSW limit of `vector`
  (pgvector README, "HNSW"): out of scope here, raise `ValueError` above 2000
  with a pointer to the follow-up.
- Multilingual corpora: `simple` config is the default because it does no
  stemming; per-document language is a follow-up.

## Out of Scope
- Running migrations, creating roles or RLS policies.
- IVFFlat, DiskANN (pgvectorscale), `pg_textsearch` BM25.
