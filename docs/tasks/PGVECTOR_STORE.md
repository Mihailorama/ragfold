---
purpose: "Real pgvector VectorStore over a caller-owned async connection, RLS-friendly"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
---

# Feature: pgvector Store

## Problem
`PGVectorStore` is a stub. The `pgvector` extra pulls `psycopg[binary]`, which
is LGPL-3.0-only (PyPI metadata, psycopg 3.3.8, checked 2026-11-09), so
permissive-only downstream apps cannot install it. More importantly, a
multi-tenant app such as saas-agent-starter runs every tenant query in its own
transaction after `set_config('app.org_id', ...)`, under Postgres RLS. A store
that opens its own connection or transaction would bypass that scoping.

## Proposed Solution
`src/ragfold/stores/pgvector.py`, **no runtime dependency**:

- `AsyncConnection` Protocol with the asyncpg method shapes ragfold uses:
  `execute(query, *args)`, `fetch(query, *args)`, `executemany(query, args)`.
  An asyncpg `Connection` satisfies it as-is (asyncpg 0.32.0, Apache-2.0).
  A psycopg adapter (`%s` placeholders) is a follow-up, not part of this PR.
- `PgVectorStore(conn: AsyncConnection, table: PgVectorTable, *, iterative_scan: Literal["off", "relaxed_order", "strict_order"] = "off", ef_search: int | None = None)`
  implements `VectorStore` from `VECTOR_STORES.md`:
  - `upsert`: one `executemany` of `INSERT ... ON CONFLICT (id) DO UPDATE`.
  - `delete_documents` / `delete_chunks`: `DELETE ... WHERE ... = ANY($1)`,
    returns the row count parsed from the command tag.
  - `chunk_hashes`: `SELECT id, content_hash, embedding_model ... WHERE document_id = $1`.
  - `search_vector`: `ORDER BY embedding <=> $1::text::vector LIMIT $2`, score
    `1 - distance`. Vectors are sent as pgvector text literals (`'[0.1,0.2]'`),
    so the caller does not have to register a codec.
  - When `iterative_scan` or `ef_search` is set, the store issues
    `SET LOCAL hnsw.iterative_scan = ...` / `SET LOCAL hnsw.ef_search = ...`
    before the query. pgvector documents `SET LOCAL` for per-query settings and
    iterative scans (0.8.0+) for filtered queries, which is the RLS case. Outside
    a transaction `SET LOCAL` has no effect; the docs say so.
- Contract, enforced by tests: the store never calls `BEGIN`, `COMMIT`,
  `ROLLBACK`, `set_config`, any DDL, or opens a connection. All values are bind
  parameters; identifiers come only from the validated `PgVectorTable`.
- `filter` keys must be names in an allow-list given to the constructor
  (`filter_columns=("org_id", "source")`) and map to columns, or to
  `metadata->>'key'` when prefixed `metadata.`. Unknown keys raise `ValueError`.
- `pyproject.toml`: the `pgvector` extra becomes empty (kept for discoverability);
  `psycopg` is no longer pulled. Tests that want a driver use `asyncpg` in the
  `test` extra only when `RAGFOLD_TEST_PG_DSN` is set.

## Affected Files
- `src/ragfold/stores/pgvector.py` - new; `src/ragfold/vectorstores.py` - remove stub.
- `pyproject.toml` - `pgvector` extra, `postgres` pytest marker.
- `tests/stores/test_pgvector_store.py` - fake-connection unit tests.
- `tests/stores/test_pgvector_store_pg.py` - real Postgres (`slow`, `postgres`).

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/stores/test_pgvector_store.py::test_store_never_manages_transactions_or_ddl`
(a recording fake connection; assert no statement starts with BEGIN/COMMIT/
ROLLBACK/CREATE/ALTER/DROP and none contains `set_config`).
- [ ] `upsert` sends one `executemany` with bind parameters in column order.
- [ ] `search_vector` SQL uses `<=>` and `LIMIT $n`; vector is a text literal param.
- [ ] `iterative_scan="relaxed_order"` emits `SET LOCAL hnsw.iterative_scan = relaxed_order` first.
- [ ] Filter on an allow-listed column becomes `AND org_id = $n`; unknown key -> `ValueError`.
- [ ] `delete_documents` returns the count from the `DELETE n` tag.

### Integration / E2E Tests
- [ ] `postgres`: round trip upsert -> search -> delete against `pgvector/pgvector`.
- [ ] `postgres`: **RLS proof** - table with `org_id` and policy
      `USING (org_id = current_setting('app.org_id', true))`, a non-owner role,
      two orgs' chunks; inside a transaction with `set_config('app.org_id','a',true)`
      the store returns only org `a` rows, and with `iterative_scan` on still
      returns `top_k` rows when org `a` is a small share of the table.

### Test Commands
```bash
pytest tests/stores -m "not slow"
RAGFOLD_TEST_PG_DSN=postgresql://... pytest tests/stores -m postgres
```

## Edge Cases
- Empty `upsert` / `delete` lists: no statement sent.
- Upsert with a vector of the wrong dimension: Postgres raises; ragfold adds no text.
- `top_k` larger than the row count.
- Chunks from another embedding model in the same table (search still works;
  ingestion re-embeds).

## Out of Scope
- Full-text and hybrid search (`PGVECTOR_HYBRID_SEARCH.md`).
- Connection pools, retries, psycopg/SQLAlchemy adapters.
- A default CI job with a Postgres service (open question in `ROADMAP.md`).
