---
purpose: "Milestones from the 0.1 benchmark harness to a 1.0 retrieval library that saas-agent-starter can depend on"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
---

# Roadmap to 1.0

Goal (owner, 2026-11-09): ragfold becomes a complete retrieval library that
saas-agent-starter depends on, instead of the starter re-implementing
retrieval. Constraints taken from the starter and applied to every task below:

- No model call inside ragfold: embedders and rerankers are injected async
  callables; ragfold never picks a model.
- Storage runs on a caller-provided connection inside the caller's transaction
  and RLS scope; ragfold never opens connections, commits, or runs DDL. Schema
  is exported as data.
- pgvector + Postgres full-text + RRF is the default; other stores are extras.
- Async APIs; empty default dependencies; permissive licenses only; no content
  in logs or exception messages.

Each task is one PR and follows the TDD rule (proposal, failing test, RED,
implementation, GREEN). **S** marks what the starter needs, in the order it
needs it.

## M0 - Ground rules (0.2.0)
1. **S** `NO_CONTENT_LOGGING.md` (P0) - rule + sentinel test harness every later task extends.
2. **S** `PUBLIC_API_AND_PACKAGING.md` (P0) - public-name snapshot, deprecation
   policy, drop LGPL `psycopg` from extras.

## M1 - Starter core (0.3.0): everything the starter's KB PRs need
3. **S** `EMBEDDERS.md` (P0) - async `Embedder`, `CallableEmbedder`.
4. **S** `VECTOR_STORES.md` (P0, rescoped) - async store protocol, in-memory reference.
5. **S** `PGVECTOR_SCHEMA.md` (P0) - `PgVectorTable` + DDL as data.
6. **S** `PGVECTOR_STORE.md` (P0) - store over a caller connection, RLS proof test.
7. **S** `PGVECTOR_HYBRID_SEARCH.md` (P0) - FTS + vector + RRF in one SQL query.
8. **S** `PRODUCTION_CHUNKING.md` (P0) - Markdown/token-aware chunks over
   langchain-text-splitters, offsets and hashes.
9. **S** `INGESTION_PIPELINE.md` (P0) - idempotent ingest/delete by content hash.
10. **S** `RETRIEVER.md` (P0) - query -> store -> optional rerank -> `RetrievalResult`.

3, 5 and 8 are independent and can run in parallel; 4 precedes 6; 6 precedes 7;
9 needs 3, 4, 8; 10 needs 3, 4 (and 7 for native hybrid).

## M2 - Answers and evals (0.4.0)
11. **S** `RERANKERS.md` (P1, rescoped) - async `Reranker`, `CallableReranker`, awaited router stage.
12. **S** `CITATIONS.md` (P1) - numbered context and marker resolution.
13. **S** `RECORDED_EMBEDDINGS_EVAL.md` (P1) - key-free recall@k / MRR in CI.

## M3 - Convenience adapters (0.5.0, Beta)
14. `EMBEDDER_ADAPTERS.md` (P2) - LiteLLM, sentence-transformers.
15. `RERANKER_ADAPTERS.md` (P2) - LiteLLM, CrossEncoder.
16. `DOCFOLD_PARSE.md` (P2) - file -> Markdown via docfold.
17. `DENSE_ENGINES.md` (P2, rescoped) - benchmark engines on the new protocols.

## M4 - 1.0.0
18. `DOCS.md` (P1) - library-first README, tested guides.
19. API freeze: remove everything deprecated in 0.2-0.5, snapshot becomes the
    1.0 contract, `Development Status :: 5`.

## After 1.0 (on demand)
- `QDRANT_STORE.md` (P3).
- psycopg / SQLAlchemy connection adapters for `PgVectorStore`.
- `pg_textsearch` BM25, per-document text-search language, `halfvec` > 2,000 dims.

## Audit of the earlier backlog (2026-11-09)
- DONE (implemented, tests green): `BM25_ENGINE`, `TEXT_RAG_ENGINE`,
  `RAGFOLD_BASELINE`, `GITHUB_RAG_INTEGRATIONS`, `GITHUB_RAG_CLIENT_ADAPTERS`,
  `ROUTER_PIPELINE_STAGES`, `hybrid-retrieval-rrf`, `WEMM_MULTIMODAL_EMBEDDING`,
  `VISUAL_ENGINES`, `LIGHTRAG_REAL_ADAPTER` (already DONE).
- Rescoped in place: `VECTOR_STORES` (P0), `RERANKERS` (P1), `DENSE_ENGINES` (P2).
- OBSOLETE: `FRAMEWORK_RETRIEVERS`.

## Open questions for the owner
1. **Postgres in CI.** The golden rules keep default CI service-free. Proposal:
   a separate `postgres` job (ubuntu only, `pgvector/pgvector` service
   container) that runs `-m postgres`, outside the default `pytest -m "not slow"`
   path. Acceptable, or keep Postgres tests manual?
2. **langchain-core's dependency tree.** `langchain-text-splitters` is MIT, but
   it pulls `langchain-core` and with it `langsmith`, `pydantic`, `httpx`,
   `uuid-utils`, `langchain-protocol` and more. If the starter's license gate or
   image size rejects that, the fallback is `chonkie` (MIT). Which is preferred?
3. **asyncpg-first.** `PgVectorStore` targets the asyncpg call shape (the
   starter's driver). psycopg support is a later adapter. OK?
4. **Status flips.** Nine older proposals were marked DONE from code and green
   tests, not from an owner sign-off. Revert any you want to re-review.
