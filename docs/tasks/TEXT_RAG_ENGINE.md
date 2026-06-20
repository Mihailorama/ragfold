---
purpose: "Keep a real TF-IDF text-rag baseline"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
---

# Feature: Text Rag Engine

## Problem
Ragfold needs a no-dependency text baseline that is always available in CI and
useful as a control engine.

## Proposed Solution
Add `TextRagEngine` with pure-Python TF-IDF scoring over normalized text
chunks. It returns `RetrievalResult` and declares lexical text capabilities.

## Affected Files
- `src/ragfold/engines/text_rag.py` - engine implementation.
- `src/ragfold/engines/__init__.py` - export and default registry.

## Test Plan

### Unit / Functional Tests
- [ ] Ranks a clearly relevant chunk first.
- [ ] Handles dict, string, and dataclass corpus entries.
- [ ] Is always available in core CI.

### Integration / E2E Tests
- [ ] Produces a row in the CLI compare Markdown table.

### Test Commands
```bash
pytest tests/engines/test_lexical_engines.py -m "not slow"
```

## Edge Cases
- Empty text chunks.
- `top_k` larger than corpus size.

## Out of Scope
- Neural retrieval or query expansion.
