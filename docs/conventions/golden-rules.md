---
purpose: "Non-negotiable engineering rules for AI and humans working in ragfold"
updated: "2026-06-20"
---

# Golden Rules

These rules are absolute.

## Rule: TDD Before Implementation

- Write a proposal in `docs/tasks/NAME.md` using `docs/tasks/_TEMPLATE.md`.
- Write the failing test before production code.
- Run the test and confirm RED.
- Implement the smallest behavior that makes it GREEN.
- Run `pytest -m "not slow"` before committing.

Do not write implementation first and backfill tests.

## Rule: Never Push Without Explicit Approval

Do not push to GitHub unless the user explicitly asks. Local commits are fine
when requested by the task; remote pushes are not.

## Rule: Preserve the Public Contract

`RagEngine` -> `RetrievalResult` / `RagAnswer` is the public surface. Extend it
carefully and keep backward compatibility.

## Rule: Keep CI Light

The default CI path is:

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

It must pass with no GPU, no network, no API keys, no model downloads, and no
external vector-store service.

## Rule: Gate Heavy Engines

Every cloud, GPU, model, framework, service, and heavyweight backend must be an
optional extra in `pyproject.toml`, guarded by `is_available()`, and covered by
tests showing graceful degradation when unavailable.

## Rule: Metric Convention

Metrics take predicted value first and reference value second. Retrieval and
answer metrics are higher-is-better. Costs are reported separately.
