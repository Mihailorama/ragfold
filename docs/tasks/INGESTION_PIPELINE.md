---
purpose: "Idempotent ingestion primitives: chunk, hash, embed only what changed, upsert, delete"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
---

# Feature: Ingestion Pipeline

## Problem
Every app re-implements the same loop: chunk a document, embed, write, and on
re-upload work out what changed. Done naively it re-embeds everything (cost),
leaves stale chunks behind, or mixes embedding models after a model change.

## Proposed Solution
New module `src/ragfold/ingest.py`, stdlib only, composed of the protocols in
`PRODUCTION_CHUNKING.md`, `EMBEDDERS.md` and `VECTOR_STORES.md`:

```python
report = await ingest_document(
    document_id, text, *, chunker, embedder, store,
    metadata=None, force=False,
)
removed = await delete_document(document_id, *, store)
```

- Steps: chunk -> `store.chunk_hashes(document_id)` -> classify each chunk as
  unchanged (same id, hash and `embedder.model`), changed or new -> embed only
  changed/new with `input_type="document"` -> `store.upsert` -> `store.delete_chunks`
  for ids no longer produced.
- `force=True` re-embeds every chunk (embedding-model migration).
- `IngestReport`: counts only (`chunks_total`, `embedded`, `unchanged`,
  `deleted`) plus `embedding_model`. No text.
- Transactions are the caller's: the store already holds the caller's
  connection, so the app wraps `ingest_document` in its own transaction (and
  RLS scope). ragfold never commits.
- Parsing is not here; pass text from docfold or anywhere (`DOCFOLD_PARSE.md`).
- `metadata` is copied onto every chunk.

## Affected Files
- `src/ragfold/ingest.py` - new.
- `src/ragfold/__init__.py` - exports.
- `tests/test_ingest.py` - new (in-memory store, counting fake embedder).

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_ingest.py::test_reingesting_unchanged_document_embeds_nothing`.
- [ ] First ingest embeds every chunk; report counts match.
- [ ] Re-ingest of identical text: `embedded == 0`, `unchanged == chunks_total`.
- [ ] Editing one section re-embeds only that section's chunks.
- [ ] A shorter new version deletes the trailing chunks.
- [ ] Changing `embedder.model` re-embeds all chunks; `force=True` does too.
- [ ] `delete_document` removes all chunks and returns the count.
- [ ] Embedder failure leaves the store unchanged (embedding happens before any write).
- [ ] Report and exception messages contain no document text (sentinel check).

### Integration / E2E Tests
- [ ] `postgres`: the same scenarios against `PgVectorStore` inside one transaction,
      rolled back on embedder failure by the test's own transaction.

### Test Commands
```bash
pytest tests/test_ingest.py -m "not slow"
```

## Edge Cases
- Empty text: deletes all existing chunks of the document.
- Two documents with identical chunks: ids differ by `document_id`, both stored.
- Concurrent ingest of the same document: the caller serialises (documented;
  the starter uses one Temporal workflow per document).

## Out of Scope
- Queues, retries, scheduling, file watching (CocoIndex adapter stays separate).
- Parsing.
