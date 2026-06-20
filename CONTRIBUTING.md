# Contributing to Ragfold

Ragfold follows the same development discipline as docfold: TDD first, light CI,
and no hidden network/model work in the default path.

## Local Setup

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

## TDD Workflow

1. Write a proposal in `docs/tasks/NAME.md` using `docs/tasks/_TEMPLATE.md`.
2. Write a failing pytest that captures the behavior.
3. Run the test and confirm RED.
4. Implement the smallest change that makes it GREEN.
5. Run `pytest -m "not slow"` before committing.

## Engine Rules

- `RagEngine`, `RetrievalResult`, and `RagAnswer` are the public surface.
- Extend the contract; do not rewrite it casually.
- Every engine must declare `capabilities`.
- Heavy, cloud, GPU, external-service, and model backends must be optional extras.
- Real inference tests for those paths must be marked `slow`.
- Unavailable engines must list as unavailable and never crash a sweep.

## Metrics

All metric functions take predicted value first and reference value second.
Higher is better for retrieval and answer metrics. Costs are reported separately.

## Release Hygiene

Never push from an agent session. Do not push without explicit user approval.
Keep commits small and logical, with green non-slow tests.
