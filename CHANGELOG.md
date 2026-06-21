# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows semantic versioning once published.

## [Unreleased]

### Added

- Public `RagEngine` contract with `RetrievalResult` and `RagAnswer`.
- Offline lexical engines: `text-rag` TF-IDF and `bm25`.
- Gated dense adapters for sentence-transformers, OpenAI, Cohere Embed, and Voyage.
- Gated OCR-free visual adapters for ColPali, ColQwen2, PixelRAG, and DSE.
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

### Changed

- Replaced placeholder README with docfold-grade package documentation.

### Fixed

- No known fixes yet.
