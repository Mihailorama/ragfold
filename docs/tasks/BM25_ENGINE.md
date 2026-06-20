---
purpose: "Add a real BM25 lexical retrieval engine"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
---

# Feature: BM25 Engine

## Problem
Ragfold needs a stronger lexical baseline than TF-IDF and the CLI acceptance
requires `bm25` to produce real no-network scores.

## Proposed Solution
Add `BM25Engine` with a pure-Python BM25 implementation and optional
`rank_bm25` acceleration when installed. The engine returns `RetrievalResult`,
declares lexical/text capabilities, and is available in core CI.

## Affected Files
- `src/ragfold/engines/bm25.py` - engine implementation.
- `src/ragfold/engines/__init__.py` - export and default registry.
- `pyproject.toml` - `bm25` optional extra for `rank_bm25`.

## Test Plan

### Unit / Functional Tests
- [ ] Ranks a document containing the query terms above unrelated documents.
- [ ] Declares lexical text capabilities.
- [ ] Works without optional dependencies.

### Integration / E2E Tests
- [ ] Appears in `ragfold list-engines`.
- [ ] Produces a row in `ragfold compare examples/corpus.json examples/queries.json`.

### Test Commands
```bash
pytest tests/engines/test_lexical_engines.py -m "not slow"
pytest tests/test_cli.py -m "not slow"
```

## Edge Cases
- Empty corpus.
- Query terms not present in any document.

## Out of Scope
- Language-specific analyzers or stemming.
