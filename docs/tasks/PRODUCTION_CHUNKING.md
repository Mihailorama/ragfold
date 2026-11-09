---
purpose: "Structure- and token-aware Markdown chunking as a thin wrapper over langchain-text-splitters"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
plane: "RAGFOLD-12"
---

# Feature: Production Chunking

## Problem
`RecursiveChunker` is a word-window demo: it ignores headings, tables and code
blocks, counts words instead of tokens, and returns no offsets or hashes.
Citations need offsets, idempotent ingestion needs a stable hash per chunk, and
embedding models have token limits.

## Proposed Solution
Wrap an established splitter instead of writing one (owner rule: mainstream
libraries over home-grown code).

- Pick: `langchain-text-splitters` (verified 2026-11-09 on PyPI: 1.1.3, MIT,
  Python >=3.10, one direct dependency `langchain-core>=1.4.7,<2`; no model
  calls). Two passes: `MarkdownHeaderTextSplitter(headers_to_split_on=..., strip_headers=False)`
  for structure, then `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, length_function=...)`
  inside each section. Both classes and the `length_function`/`add_start_index`
  arguments were checked in the 1.1.3 wheel.
- `MarkdownChunker(chunk_size=512, chunk_overlap=64, *, length_function: Callable[[str], int] = len, headers=("#","##","###"))`.
  Token-aware by injection: the caller passes its tokenizer's counter (e.g.
  `lambda s: len(enc.encode(s))`). ragfold does not pick a tokenizer or model.
- `Chunk` gains optional, defaulted fields (backward compatible): `start`,
  `end` (character offsets into the input, `None` if the splitter changed the
  text so it cannot be located), `content_hash` (sha256 of the chunk text
  after whitespace normalisation), `heading_path: tuple[str, ...]`.
- Chunk ids are deterministic: `f"{document_id}:{index}"`; the hash, not the id,
  decides whether a chunk changed.
- Extra `chunking = ["langchain-text-splitters>=1.1,<2"]`; `is_available()`
  gates it. `RecursiveChunker` stays as the zero-dependency fallback.
- Fallback if `langchain-core`'s transitive tree is unacceptable to a
  downstream license gate (it pulls `langsmith`, `pydantic`, `httpx`,
  `uuid-utils`, `langchain-protocol` and others): `chonkie` (1.7.0, MIT) behind
  the same `Chunker` protocol. Decide in review, not in code.

## Affected Files
- `src/ragfold/preprocessing.py` - `Chunk` fields, `MarkdownChunker`.
- `pyproject.toml` - `chunking` extra.
- `tests/test_chunking.py` - new.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_chunking.py::test_markdown_chunker_keeps_heading_path_and_offsets`
(skips with a clear reason when the extra is missing; CI installs it via `dev`).
- [ ] Chunks never cross a `##` heading; `heading_path` is `("Title", "Section")`.
- [ ] No chunk exceeds `chunk_size` by the injected `length_function`.
- [ ] `text[start:end] == chunk.text` whenever offsets are set.
- [ ] Same input -> same ids and hashes; editing one section changes only its chunks' hashes.
- [ ] A fenced code block shorter than `chunk_size` is not split.
- [ ] Empty or whitespace-only input -> `[]`.
- [ ] `RecursiveChunker` gains `content_hash` too (no extra needed).

### Integration / E2E Tests
- [ ] `EngineRouter(chunker=MarkdownChunker())` compare run on `examples/` stays green.

### Test Commands
```bash
pip install -e ".[dev,chunking]"
pytest tests/test_chunking.py -m "not slow"
```

## Edge Cases
- Markdown without headings: one section, recursive split only.
- A single token run longer than `chunk_size` (base64, minified JSON): hard split.
- Tables: kept whole when they fit; otherwise split on rows by the recursive separators.
- Non-Markdown input (plain text from docfold `TEXT`): same path, no headings.

## Out of Scope
- Semantic (embedding-based) chunking: it needs model calls; the caller can
  inject a chunker that does it.
- PDF/Office parsing (`DOCFOLD_PARSE.md`).
