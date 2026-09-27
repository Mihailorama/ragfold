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

## Durable lessons and owner corrections

- Before finishing substantial work, check whether a recurring mistake, a verified workaround, or an owner correction yields a reusable rule.
- Check existing instructions and skills first. Update the matching rule in place, reconcile contradictions, and avoid duplicate or single-session skills.
- Keep project-specific guidance in this repository; put a reusable workflow in the existing relevant skill. Keep shared policies in AGENTS.md and make them available to Claude Code through CLAUDE.md.
- Turn a fixed behavioral bug into the smallest meaningful regression check, reusing the existing tests or evals. Instruction-only edits need a diff and consistency review, not an artificial test suite.
- Record verified guidance and when it applies. Keep secrets and private client material out of shared instructions and global skills; preserve existing access, approval, and data-storage boundaries.

## Shared instruction alignment

Read this AGENTS.md and the project guidance in [CLAUDE.md](CLAUDE.md) once before working. Shared policies apply to every agent; commands, models, and tools explicitly tied to one runtime apply only to that runtime. CLAUDE.md imports AGENTS.md so both agents receive the same shared rules. When changing a shared policy, update its existing source and reconcile the counterpart in the same change. Do not recursively reload these files.
