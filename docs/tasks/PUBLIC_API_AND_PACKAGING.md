---
purpose: "Stable typed public API, semver/deprecation policy and clean permissive extras for downstream use"
status: "OPEN"
priority: "P0"
created: "2026-11-09"
---

# Feature: Public API and Packaging

## Problem
ragfold is about to become a runtime dependency of other projects
(saas-agent-starter first). Today:
- the public surface is whatever happens to be importable; `__all__` lists
  engine types only, and `__version__` is hard-coded next to `pyproject.toml`;
- `py.typed` ships, but `mypy` runs without `strict` and nothing checks that
  public signatures are fully typed;
- there is no written semver or deprecation policy, yet the backlog renames and
  removes things (`vectorstores` -> `stores`, sync `BaseReranker`, stub stores);
- the `pgvector` extra pulls `psycopg[binary]`, LGPL-3.0-only (PyPI, psycopg
  3.3.8, 2026-11-09). Every other declared extra was checked the same day and is
  MIT or Apache-2.0;
- several extras are empty placeholders (`pixelrag`, `dse`, `adaptive-chunking`,
  `agentic-file-search`, `evaluation`), and `promptfoo` installs a PyPI wrapper
  (0.2.0, pulls `posthog`) that ragfold never imports.

## Proposed Solution
- `docs/api-stability.md`: what is public (names in `ragfold.__all__` and in
  `__all__` of `ragfold.stores`, `ragfold.embeddings`, `ragfold.rerankers`,
  `ragfold.ingest`, `ragfold.retriever`, `ragfold.citations`,
  `ragfold.evaluation`); 0.x: breaking changes only in a minor release, after a
  `DeprecationWarning` in the previous minor; 1.0: semver, deprecations live one
  minor before removal; underscore modules are private.
- `tests/test_public_api.py`: snapshot of every public name per module; a diff
  fails until the snapshot and `CHANGELOG.md` are updated together.
- `mypy --strict` on the public modules (CI job step), not on engine adapters.
- `__version__` from `importlib.metadata.version("ragfold")`.
- Extras: drop `psycopg` from `pgvector` (see `PGVECTOR_STORE.md`); remove empty
  placeholder extras and the `promptfoo` extra after one release with a
  CHANGELOG note; add `tests/test_optional_extras.py::test_no_copyleft_packages_in_extras`
  with a small deny-list (`psycopg`, `psycopg2`, `pymupdf`, `PyMuPDF`, `mineru`,
  `surya-ocr`).
- Classifiers: add Python 3.13, set `Development Status :: 4 - Beta` at 0.5,
  `5 - Production/Stable` at 1.0.

## Affected Files
- `src/ragfold/__init__.py` and public modules - `__all__`.
- `pyproject.toml` - extras, classifiers, mypy config.
- `docs/api-stability.md` - new; `CHANGELOG.md`.
- `tests/test_public_api.py` - new; `tests/test_optional_extras.py` - extend.
- `.github/workflows/ci.yml` - `mypy --strict` step, Python 3.13 in the matrix.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/test_optional_extras.py::test_no_copyleft_packages_in_extras`
(fails today on `psycopg`).
- [ ] Public-name snapshot test passes and fails on an added/removed name.
- [ ] `ragfold.__version__` equals the installed distribution version.
- [ ] `test_ci_matrix_is_light_and_cross_platform` extended for 3.13.

### Integration / E2E Tests
- [ ] `python -m build && twine check dist/*` in CI (no upload).
- [ ] A downstream type-check smoke: a file using the public API passes
      `mypy --strict` with the installed wheel (proves `py.typed` works).

### Test Commands
```bash
pytest tests/test_public_api.py tests/test_optional_extras.py -m "not slow"
mypy --strict src/ragfold/__init__.py src/ragfold/stores src/ragfold/embeddings.py
```

## Edge Cases
- Removing an extra breaks `pip install ragfold[promptfoo]` for existing users:
  keep it as an empty extra for one release, then remove.
- `all` extra must be kept in sync (test asserts it lists every non-dev extra).

## Out of Scope
- Generated API reference site (`DOCS.md`).
- Changing the `RagEngine` contract.
