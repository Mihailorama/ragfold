---
purpose: "Numbered context building and citation resolution for answers, with no model call"
status: "OPEN"
priority: "P1"
created: "2026-11-09"
---

# Feature: Citations

## Problem
`RagAnswer.citations` is a bare `list[str]`. Apps need to (1) put retrieved
passages into a prompt with stable markers, within a size budget, and (2) map
the markers in the model's answer back to documents and character ranges, while
dropping markers the model invented. Today every app writes this glue.

## Proposed Solution
New module `src/ragfold/citations.py`, stdlib only, no prompt text:

- `build_context(passages, *, max_units, length_function=len, separator="\n\n") -> Context`.
  Numbers passages `[1]..[n]` in rank order, stops before `max_units` (chars by
  default, tokens with an injected counter), and returns `Context(text, sources)`
  where `Source` = `marker`, `chunk_id`, `document_id`, `start`, `end`,
  `heading_path`. Formatting of each block is a small template string the
  caller can override (`"[{marker}] {text}"`); ragfold ships no instructions to
  the model: prompts stay in the app (saas-agent-starter keeps them versioned
  under `prompts/`).
- `resolve_citations(answer_text, context) -> list[Source]`: parses `[n]` and
  `[n, m]` markers, returns the cited sources in first-mention order, ignores
  markers outside `1..len(sources)`, and reports them in
  `metadata["unknown_markers"]` (numbers only).
- `attach_citations(answer_text, context) -> RagAnswer`: fills `citations`
  with chunk ids (existing field, same type) and `metadata["sources"]` with the
  `Source` dicts. Public contract unchanged.
- Sources map directly to the AI SDK `source-document` stream parts the starter
  plans (id, title, offsets); the mapping is the app's.

## Affected Files
- `src/ragfold/citations.py` - new.
- `src/ragfold/__init__.py` - exports.
- `tests/test_citations.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_citations.py::test_resolve_citations_maps_markers_and_drops_invented_ones`.
- [ ] `build_context` numbers in rank order and respects `max_units`.
- [ ] A passage that alone exceeds the budget is skipped, not truncated mid-sentence.
- [ ] `[2]`, `[1, 3]`, repeated `[2]` -> sources 2, 1, 3 once each.
- [ ] `[9]` with 3 sources is ignored and listed in `unknown_markers`.
- [ ] `attach_citations` returns a `RagAnswer` whose `to_dict()` stays JSON-serialisable.
- [ ] Custom block template is honoured.

### Integration / E2E Tests
- [ ] `Retriever` result -> `build_context` -> fake answer -> `attach_citations`
      round trip keeps offsets from `PRODUCTION_CHUNKING.md`.

### Test Commands
```bash
pytest tests/test_citations.py -m "not slow"
```

## Edge Cases
- No passages: empty context, no sources.
- Markers inside code blocks in the answer (`a[1]`): only bracketed integers
  preceded by start, whitespace or punctuation count.
- Footnote-style `[^1]` markers: not supported, documented.

## Out of Scope
- Generating the answer; faithfulness grading (Promptfoo / the app's eval).
- Sentence-level attribution.
