---
purpose: "Add WeMM-Embedding (Tencent/WeChat) as a gated local multimodal dense engine"
status: "OPEN"
priority: "P1"
created: "2026-09-06"
---

# Feature: WeMM-Embedding engine adapter

## Problem
Ragfold has visual late-interaction engines (colpali, colqwen2) and text dense
engines (sentence-transformers, voyage, ...), but no adapter for a **unified
multimodal dense** embedder that puts text, images, video and visual documents
in one vector space. WeMM-Embedding (Tencent/WeChat Vision, Apache-2.0, arXiv
2608.24053) is exactly that, and it ships Matryoshka Representation Learning:
one model yields nested dimensions (2B: 64–2048) so we can truncate at inference
(the 2B model keeps 98.7% of full image+video MMEB-v2 score at 256 dims) with no
retraining — directly useful for cheaper vector stores. We want to benchmark it
against colqwen2 / voyage-multimodal on our own corpora via `ragfold compare`.

## Proposed Solution
Add `WeMMEmbeddingEngine`, a gated local-VLM dense engine reusing the existing
`_DenseVectorEngine` retrieve/VectorStore path, with WeMM-specific capabilities
(`modality="multimodal"`, `retrieval_method="dense"`, `vlm=True`,
`requires_gpu=True`, `license="Apache-2.0"`) and an optional `truncate_dim`
(Matryoshka) that slices embeddings before indexing. Real weight loading stays
gated (injected model in tests, real inference marked `slow`); default CI stays
light. Register it as `wemm`, add the `wemm` optional extra, and document it.

## Affected Files
- `src/ragfold/engines/wemm.py` - new `WeMMEmbeddingEngine` + `truncate_dim`.
- `src/ragfold/engines/router.py` - register `"wemm"` factory.
- `src/ragfold/engines/__init__.py` - export + `__all__` + DEFAULT_ENGINE_NAMES.
- `pyproject.toml` - `wemm` optional extra (torch, transformers, pillow).
- `README.md` - Engine Comparison + How to Choose + Install Extras rows.
- `CHANGELOG.md` - entry.
- `tests/engines/test_wemm_engine.py` - new tests.

## Test Plan

### Unit / Functional Tests
- [ ] Injected embedder path: `retrieve` ranks the relevant doc first; capabilities are multimodal/dense/vlm/Apache-2.0.
- [ ] `truncate_dim=1` slices vectors to 1 dimension (Matryoshka behavior) via the injected embedder.
- [ ] Unavailable without an injected model and without torch+transformers: `is_available() is False`, `ensure_available()` raises `NotImplementedError`.
- [ ] `wemm` is registered in the default router and listed in `DEFAULT_ENGINE_NAMES`.
- [ ] `wemm` optional extra is declared in `pyproject.toml`.

### Integration / E2E Tests
- [ ] (slow, not in default CI) real `tencent/WeMM-Embedding-2B` load + encode — out of scope for this task, added later behind `@pytest.mark.slow`.

### Test Commands
```bash
pytest tests/ -m "not slow"
```

## Edge Cases
- `truncate_dim` larger than the model dimension → return the full vector (no padding).
- `truncate_dim=None` → full-dimensional behavior (default).
- Package present but no weights/GPU → engine reports available but real inference is gated and marked slow.

## Out of Scope
- Real weight download / GPU inference in default CI.
- Video/image ingestion plumbing beyond the text-vector `encode` contract (corpus stays text-first; multimodal inputs are injected-model territory for now).
- Reranking.
