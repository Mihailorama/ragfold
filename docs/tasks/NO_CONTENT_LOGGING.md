---
purpose: "Guarantee ragfold never puts document, chunk or query text in logs or exception messages"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
---

# Feature: No Content in Logs

## Problem
Downstream apps hold user documents under privacy rules (saas-agent-starter:
"never log prompt or completion content or personal data; log ids and
counts"). As ragfold gains ingestion and stores, one careless
`logger.debug(chunk.text)` or `ValueError(f"bad chunk {text}")` would leak
content into the app's logs and error tracker. Nothing prevents that today.

## Proposed Solution
- Golden rule in `docs/conventions/golden-rules.md`: "No content in logs:
  ragfold logs and raises with ids, counts, dimensions and hashes only, never
  document, chunk, query or answer text, vectors, or metadata values."
- `ragfold` uses `logging.getLogger("ragfold.<module>")` only (no `print`
  outside `cli.py`), with no handlers configured.
- `tests/test_no_content_logging.py`: a shared `sentinel` fixture
  (`"SENTINEL-<uuid>"`) put into corpus text, queries and metadata; runs the
  router, fusion, chunkers and (as they land) ingest, stores, retriever and
  citations, including error paths (dimension mismatch, wrong score count,
  unknown filter key), with `caplog` at `DEBUG`; asserts the sentinel is absent
  from every log record and every raised exception's `str()`.
- Each later proposal adds its module to this test (already listed in their test plans).

## Affected Files
- `docs/conventions/golden-rules.md`, `AGENTS.md`, `CLAUDE.md` - rule (CLAUDE.md
  imports AGENTS.md; summary line only).
- `tests/test_no_content_logging.py` - new.
- `tests/conftest.py` - `sentinel` fixture.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_no_content_logging.py::test_router_and_stores_never_log_or_raise_with_content`.
- [ ] Router `retrieve`, `retrieve_hybrid`, `compare` with sentinel content: no sentinel in logs.
- [ ] `InMemoryVectorStore` dimension mismatch message contains no sentinel.
- [ ] `grep`-style test: no `print(` in `src/ragfold` except `cli.py`.

### Integration / E2E Tests
- [ ] None.

### Test Commands
```bash
pytest tests/test_no_content_logging.py -m "not slow"
```

## Edge Cases
- Exceptions from injected callables (embedder, reranker, LLM client) are the
  caller's; ragfold does not wrap them with text.
- CLI output is user-facing by design and exempt.

## Out of Scope
- Redacting content inside third-party libraries' logs.
