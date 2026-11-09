---
purpose: "Optional docfold adapter: file to Markdown text for the ingestion pipeline"
status: "OPEN"
priority: "P2"
created: "2026-11-09"
plane: "RAGFOLD-3"
---

# Feature: docfold Parse Adapter

## Problem
`ingest_document` takes text. Users of ragfold alone still need to turn PDFs and
Office files into Markdown. docfold (sibling project, MIT, 0.8.1 on PyPI
2026-11-04, zero core dependencies) already routes 20+ parsers behind one async
`EngineRouter.process(file_path, output_format=OutputFormat.MARKDOWN, engine_hint=None)`
that returns `EngineResult(content, format, engine_name, metadata, pages, ...)`
(read in the local clone, docfold 0.8.1).

## Proposed Solution
- `ragfold.parsing.ParsedDocument(text, format, engine, pages, metadata)`.
- `async parse_file(path, *, router=None, engine_hint=None) -> ParsedDocument`:
  uses a caller-supplied docfold `EngineRouter` (or builds the default one),
  asks for Markdown, maps the `EngineResult`. No bytes or text are logged.
- Extra `docfold = ["docfold>=0.8.1,<1"]`; gated by `is_available()`.
- Apps that already call docfold (saas-agent-starter does, in its own parse
  activity) skip this and pass text straight to `ingest_document`.

## Affected Files
- `src/ragfold/parsing.py` - new.
- `pyproject.toml` - `docfold` extra.
- `tests/test_parsing.py` - new (fake router object).

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_parsing.py::test_parse_file_maps_docfold_result_to_parsed_document`.
- [ ] A fake router returning an `EngineResult`-shaped object maps to `ParsedDocument`.
- [ ] `engine_hint` is forwarded.
- [ ] Without docfold installed and no router passed: `NotImplementedError` naming the extra.

### Integration / E2E Tests
- [ ] `slow`: real docfold MarkItDown engine on a small DOCX fixture created in the test.

### Test Commands
```bash
pytest tests/test_parsing.py -m "not slow"
```

## Edge Cases
- docfold returns `TEXT` instead of Markdown (engine without Markdown output):
  passed through with `format="text"`; the chunker handles both.
- Parser failure: docfold's exception propagates.

## Out of Scope
- Choosing or installing docfold engines (docfold's job); AGPL/GPL engines are
  never enabled by ragfold.
