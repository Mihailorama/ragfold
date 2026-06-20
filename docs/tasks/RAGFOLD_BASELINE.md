---
purpose: "Bring ragfold to a docfold-grade RAG/extraction aggregator baseline"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
---

# Feature: Ragfold Baseline

## Problem
`ragfold` needs the same shape and engineering standard as sibling `docfold`,
but for RAG and information-extraction engines: one public engine contract,
light CI, built-in metrics, runner, router, CLI, docs, and optional heavy
adapters that degrade cleanly when dependencies or keys are absent.

## Proposed Solution
Create a lightweight Python package with:
- `RagEngine`, `RetrievalResult`, and `RagAnswer` as the public surface.
- Real no-network lexical engines: `text-rag` and `bm25`.
- Gated dense, visual, reranking, vector-store, and framework adapters.
- A router with registration, `from_engine_names`, auto-select, compare, and
  bounded batch processing.
- Evaluation metrics/reporting and dataset adapters.
- CLI commands: `list-engines`, `compare`, and `bench`.
- Docfold-grade README, benchmark methodology, contribution docs, changelog,
  agent rules, examples, and CI.

## Affected Files
- `src/ragfold/engines/base.py` - public engine/result dataclasses.
- `src/ragfold/engines/router.py` - registry, selection, compare, batch.
- `src/ragfold/evaluation/` - metrics, runner, datasets.
- `src/ragfold/cli.py` - command line interface.
- `pyproject.toml` - package metadata, extras, pytest config.
- `README.md`, `docs/benchmarks.md`, `CONTRIBUTING.md`, `CHANGELOG.md`,
  `AGENTS.md`, `CLAUDE.md`, `docs/conventions/golden-rules.md` - docs.
- `.github/workflows/ci.yml` - no-network CI matrix.

## Test Plan

### Unit / Functional Tests
- [ ] Public result dataclasses serialize stable fields.
- [ ] `text-rag` ranks relevant passages with no optional dependencies.
- [ ] `bm25` ranks relevant passages with no optional dependencies.
- [ ] Every declared engine appears in `list_engines`.
- [ ] Unavailable engines are listed but skipped in compare/bench sweeps.
- [ ] In-memory vector store returns nearest vectors.
- [ ] Optional vector stores and model/API engines are gated by `is_available`.
- [ ] Router can auto-select, compare, construct from names, and process batches.
- [ ] Metrics follow predicted-first/reference-second convention.
- [ ] Evaluation reports include accuracy, latency, and token-cost columns.
- [ ] CLI commands print populated Markdown tables.

### Integration / E2E Tests
- [ ] `ragfold compare examples/corpus.json examples/queries.json` prints rows for
  `text-rag` and `bm25`.
- [ ] `pip install -e ".[dev]" && pytest -m "not slow"` passes without GPU or
  network.

### Test Commands
```bash
pytest tests/ -m "not slow"
ragfold compare examples/corpus.json examples/queries.json
```

## Edge Cases
- Empty corpus and empty query sets.
- Unknown engine names.
- Optional dependencies missing.
- SaaS engines without API keys.
- Engines raising `NotImplementedError` during a sweep.

## Out of Scope
- Downloading real ViDoRe/UNIDOC assets during CI.
- Real inference for GPU/model/SaaS adapters without explicit extras and keys.
