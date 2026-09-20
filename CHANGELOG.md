# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows semantic versioning once published.

## [Unreleased]

### Added

- Cross-engine hybrid retrieval with Reciprocal Rank Fusion: pure
  `ragfold.fusion.reciprocal_rank_fusion(rankings, *, k=60, weights=None)`
  (deterministic, stdlib-only, with an explainable per-engine contribution and
  `consensus` breakdown) and `EngineRouter.retrieve_hybrid(..., engines=[...],
  weights=None)` which runs multiple engines concurrently over one prepared
  corpus and fuses them (`engine_name` like `rrf(bm25,text-rag)`). Public
  contract unchanged.
- Weighted fusion: optional per-engine `weights` (mapping or positional
  sequence) scale each engine's RRF contribution.
- Learnable weights: `ragfold.fusion_tuning.tune_rrf_weights(...)` grid-searches
  per-engine weights against a labelled query set to maximise a retrieval
  metric (deterministic, CI-light, uniform baseline always evaluated).
- CLI `ragfold hybrid <corpus> <query> --engines ... [--k --top-k --weights]`.
- MCP server (optional `mcp` extra, `ragfold-mcp` script) exposing
  `ragfold_list_engines`, `ragfold_retrieve`, `ragfold_hybrid_search`, and
  `ragfold_compare` so API, CLI, and MCP share one router and fusion. Works with
  the mcp 1.x `FastMCP` and 2.x `MCPServer` class names.
- Public `RagEngine` contract with `RetrievalResult` and `RagAnswer`.
- Offline lexical engines: `text-rag` TF-IDF and `bm25`.
- Gated dense adapters for sentence-transformers, OpenAI, Cohere Embed, and Voyage.
- Gated OCR-free visual adapters for ColPali, ColQwen2, PixelRAG, and DSE.
- Gated `wemm` adapter for WeMM-Embedding (Tencent/WeChat, Apache-2.0): unified multimodal dense retrieval with optional Matryoshka `truncate_dim`.
- In-memory vector store plus gated FAISS, Qdrant, Chroma, and pgvector adapters.
- Optional reranker interfaces for cross-encoder and Cohere Rerank.
- Framework stubs for LlamaIndex, Haystack, and txtai.
- Gated GitHub RAG fork adapters for LightRAG, RAG-Anything, and agentic-file-search.
- Optional LightRAG runtime path for configured `lightrag-hku` clients.
- Lifecycle stages for chunking, context compression, indexing, and promptfoo export.
- Engine router with auto-select, compare, and bounded batch processing.
- Evaluation metrics, dataset adapters, report generation, examples, and CLI.
- TDD proposals, docs, and light CI matrix.
- Release hygiene note: Do not push unless the user explicitly asks.
- PyPI publish workflow using trusted publishing, plus `docs/releasing.md`.

### Changed

- Replaced placeholder README with docfold-grade package documentation.

### Fixed

- `mypy` clean across the whole package: annotated
  `_UnavailableVectorStore._raise` as `NoReturn` so the gated vector-store
  `add`/`query` methods type-check.
