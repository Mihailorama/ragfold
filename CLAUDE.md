# ragfold

Open-source Python RAG and information-extraction engine aggregator. Ragfold is
the sibling of `docfold`: it provides one interface for retrieval/extraction
engines plus built-in benchmarks.

## Quick Start

```bash
pip install -e ".[dev]"
pytest -m "not slow"
ragfold compare examples/corpus.json examples/queries.json
```

## Key Files

| File/Dir | Purpose |
|---|---|
| `src/ragfold/engines/base.py` | Public `RagEngine`, `RetrievalResult`, `RagAnswer` contract |
| `src/ragfold/engines/router.py` | Registry, auto-select, compare, batch |
| `src/ragfold/evaluation/` | Metrics, runner, dataset adapters |
| `docs/tasks/_TEMPLATE.md` | Required proposal template |
| `docs/conventions/golden-rules.md` | Rules you MUST follow |

## Golden Rules Summary

1. TDD is mandatory: proposal, failing test, RED, implementation, GREEN.
2. Never push to GitHub unless explicitly asked.
3. Preserve the public contract.
4. Keep CI light: `pip install -e ".[dev]"` and `pytest -m "not slow"` must run
   without GPU, network, API keys, external services, or model downloads.
5. Metrics take predicted value first and reference value second; higher is
   better, with costs reported separately.

Full rules: `docs/conventions/golden-rules.md`.
