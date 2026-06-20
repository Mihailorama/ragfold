# Repository Guidelines

@/Users/m/.codex/RTK.md

## Project Structure

- `src/ragfold/` - package source.
- `tests/` - pytest tests mirroring package structure.
- `docs/tasks/` - proposal documents required before implementation.
- `docs/conventions/golden-rules.md` - non-negotiable project rules.
- `examples/` - tiny offline corpus/query fixtures for CLI smoke tests.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"` - install package with dev tooling.
- `pytest -m "not slow"` - required lightweight test suite.
- `ragfold list-engines` - inspect registered engines and availability.
- `ragfold compare examples/corpus.json examples/queries.json` - CLI smoke test.

## Coding Style

- Python 3.10+.
- PEP 8, 4-space indentation, type hints on public APIs.
- Keep default dependencies empty.
- Use async engine APIs for retrieval work.

## TDD Workflow

Every feature or fix must follow this sequence:

1. Proposal in `docs/tasks/NAME.md` using `_TEMPLATE.md`.
2. Failing pytest written before implementation.
3. RED run confirmed.
4. Minimal implementation.
5. GREEN run with `pytest -m "not slow"`.

## Engine Rules

- Preserve `RagEngine` -> `RetrievalResult` / `RagAnswer`.
- Declare `capabilities` for every engine.
- Gate optional engines with `is_available()`.
- Mark real model/cloud/GPU/service tests as `slow`.
- Unavailable engines must list and skip cleanly.

## Git Rules

Never push unless the user explicitly asks. Do not push from routine agent work.
