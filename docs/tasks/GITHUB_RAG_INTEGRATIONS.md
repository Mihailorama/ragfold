---
purpose: "Plan and stage integrations from Mihailorama GitHub RAG-related forks"
status: "OPEN"
priority: "P1"
created: "2026-06-20"
---

# Feature: GitHub RAG Integrations

## Problem
Mihailorama has several RAG-related GitHub forks and adjacent projects that can
make `ragfold` more useful, but they are not all the same kind of integration.
Some are retrievers, some are chunking/indexing/compression stages, and some are
evaluation tools. Pulling them all in as `RagEngine` adapters would blur the
public contract and make CI heavy.

## Proposed Solution
Add the first integration layer:

- `LightRAGEngine` - gated framework retriever adapter for HKUDS LightRAG.
- `RAGAnythingEngine` - gated multimodal/document RAG adapter for RAG-Anything.
- `AgenticFileSearchEngine` - gated agentic retrieval adapter.
- `Chunker` interface with `RecursiveChunker` and gated `AdaptiveChunker`.
- `ContextCompressor` interface with `NoopCompressor` and gated
  `HeadroomCompressor`.
- `Indexer` interface with gated `CocoIndexIndexer`.
- Promptfoo export helper for benchmark/eval configs.

All third-party integrations must be optional extras, unavailable by default,
and safe in no-network CI.

## Affected Files
- `src/ragfold/engines/github_rag.py` - gated engine adapters.
- `src/ragfold/preprocessing.py` - chunking interface and adapters.
- `src/ragfold/compression.py` - context compression interface and adapters.
- `src/ragfold/indexing.py` - indexing interface and adapters.
- `src/ragfold/evaluation/promptfoo.py` - promptfoo export.
- `src/ragfold/engines/router.py` and `src/ragfold/engines/__init__.py` -
  registry updates.
- `pyproject.toml` - optional extras.
- `README.md`, `docs/benchmarks.md`, `CHANGELOG.md` - docs updates.

## Test Plan

### Unit / Functional Tests
- [ ] New GitHub RAG engines appear in `build_default_router().list_engines()`.
- [ ] New engines are unavailable by default and skipped by compare sweeps.
- [ ] `RecursiveChunker` chunks text locally without optional dependencies.
- [ ] `AdaptiveChunker`, `HeadroomCompressor`, and `CocoIndexIndexer` degrade
  cleanly when extras are missing.
- [ ] Promptfoo export serializes a minimal benchmark config.
- [ ] Optional extras are declared in `pyproject.toml`.

### Integration / E2E Tests
- [ ] Real LightRAG/RAG-Anything/agentic-file-search calls are marked `slow`.
- [ ] Real adaptive-chunking/headroom/cocoindex tests are marked `slow`.

### Test Commands
```bash
pytest tests/ -m "not slow"
ruff check src tests
```

## Edge Cases
- Third-party package installed but no model/API/provider configured.
- Chunker creates empty chunks from whitespace-only text.
- Promptfoo export receives dataset queries without gold answers.
- Compare sweep includes unavailable integrations mixed with available engines.

## Out of Scope
- Vendoring external repositories.
- Running model downloads, SaaS APIs, Gemini, LightRAG storage, or CocoIndex
  services in default CI.
