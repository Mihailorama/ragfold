# Ragfold

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/mihailorama/ragfold/actions/workflows/ci.yml/badge.svg)](https://github.com/mihailorama/ragfold/actions/workflows/ci.yml)

**Compare RAG and information-extraction engines behind one interface.** Ragfold
does for retrieval/extraction tools what `docfold` does for document parsers:
standard adapters, clean fallbacks, built-in benchmarks, and a light default CI
that needs no GPU, model downloads, API keys, or network.

```bash
pip install -e ".[dev]"
ragfold list-engines
ragfold compare examples/corpus.json examples/queries.json --engines text-rag,bm25 --top-k 3
```

## Engine Comparison

> Research-based estimates from public docs, package metadata, and adapter
> behavior. See [docs/benchmarks.md](docs/benchmarks.md). Costs are estimates,
> not procurement quotes.

| Engine | ragfold | Modality | OCR-free | Type | License | Retrieval method | Rerank | Speed | Cost |
|---|:---:|---|:---:|---|---|---|:---:|---|---|
| `text-rag` | yes | text | yes | local | MIT | lexical TF-IDF | no | fast | free |
| `bm25` | yes | text | yes | local | MIT adapter / optional package license | lexical BM25 | no | fast | free |
| `sentence-transformers` | gated | text | yes | local | Apache-2.0 library, model-dependent | dense single-vector | optional | medium | free infra cost |
| `openai` | gated | text | yes | SaaS | commercial terms | dense single-vector | no | fast | paid tokens |
| `cohere-embed` | gated | text | yes | SaaS | commercial terms | dense single-vector | optional via Cohere Rerank | fast | paid tokens/search |
| `voyage` | gated | text | yes | SaaS | commercial terms | dense single-vector | no | fast | paid tokens |
| `colpali` | gated | visual | yes | local VLM | code/model-dependent | late-interaction | no | slow | free infra cost |
| `colqwen2` | gated | visual | yes | local VLM | code/model-dependent | late-interaction | no | slow | free infra cost |
| `pixelrag` | gated | visual | yes | local/VLM | research/model-dependent | visual embedding | no | slow | free infra cost |
| `dse` | gated | visual | yes | local | research/model-dependent | screenshot embedding | no | medium | free infra cost |
| `llamaindex` | gated | text | yes | framework | framework-dependent | framework retriever | framework-dependent | varies | varies |
| `haystack` | gated | text | yes | framework | framework-dependent | framework retriever | framework-dependent | varies | varies |
| `txtai` | gated | text | yes | framework | framework-dependent | framework retriever | framework-dependent | varies | varies |

`text-rag` and `bm25` run in core. Every model, cloud, GPU, visual, service, and
framework adapter is optional and reports `available=no` until its extra,
credentials, and runtime are configured.

## How to Choose

| Situation | Start with |
|---|---|
| Need a no-dependency baseline for CI or unit tests | `text-rag` |
| Keyword-heavy corpora, legal clauses, IDs, exact phrases | `bm25` |
| Semantic text retrieval on local hardware | `sentence-transformers` |
| Managed embeddings with low operational overhead | `openai`, `cohere-embed`, or `voyage` |
| Need reranking after first-stage retrieval | Cross-encoder or Cohere Rerank adapters |
| PDFs where layout matters and OCR should be avoided | `colpali` or `colqwen2` |
| Screenshot/page-image retrieval experiments | `pixelrag` or `dse` |
| Already invested in a RAG framework | `llamaindex`, `haystack`, or `txtai` |
| Need persistent vector search | FAISS, Qdrant, Chroma, or pgvector vector-store extras |

## Why Ragfold

| Challenge | Without Ragfold | With Ragfold |
|---|---|---|
| Try a new retriever | Rewrite indexing and result parsing | Swap an engine name |
| Keep CI light | Mock the whole RAG layer | Core lexical engines run offline |
| Compare quality | Build ad hoc notebooks | `ragfold compare` gives a Markdown report |
| Track costs | Token math lives in spreadsheets | Reports include token-cost columns |
| Handle missing extras | Import failures crash sweeps | Unavailable engines list and skip cleanly |
| Batch queries | Write concurrency glue | `EngineRouter.process_batch(..., concurrency=N)` |

## Install Extras

| Extra | Installs | Use when |
|---|---|---|
| `bm25` | `rank-bm25` | You want the external BM25 implementation instead of pure Python fallback |
| `sentence-transformers` | local embedding/reranking stack | You can download/load local models |
| `openai` | OpenAI SDK | You have `OPENAI_API_KEY` and want OpenAI embeddings |
| `cohere` | Cohere SDK | You have `COHERE_API_KEY` for Embed/Rerank |
| `voyage` | Voyage AI SDK | You have `VOYAGE_API_KEY` |
| `colpali`, `colqwen2` | ColPali engine stack | You can run visual document retrievers |
| `faiss`, `qdrant`, `chroma`, `pgvector` | vector-store clients | You need persistent or accelerated vector search |
| `llamaindex`, `haystack`, `txtai` | framework clients | You want thin wrappers over existing retrievers |
| `dev` | pytest, ruff, mypy | Local development and CI |

Examples:

```bash
pip install ragfold
pip install "ragfold[bm25,sentence-transformers,faiss]"
pip install "ragfold[openai,cohere,voyage]"
```

## Python API

```python
import asyncio

from ragfold import EngineRouter
from ragfold.engines.bm25 import BM25Engine
from ragfold.engines.text_rag import TextRagEngine


async def main():
    corpus = [
        {"id": "policy", "text": "Refunds are available for 30 days."},
        {"id": "security", "text": "Accounts require multi-factor authentication."},
    ]
    router = EngineRouter([BM25Engine(), TextRagEngine()])

    result = await router.retrieve(corpus, "refund policy", engine_hint="bm25", top_k=1)
    print(result.passages[0].document_id)

    comparison = await router.compare(
        corpus,
        [{"id": "q1", "query": "refund policy", "relevant_ids": ["policy"]}],
        top_k=1,
    )
    print(comparison.keys())


asyncio.run(main())
```

## CLI

```bash
ragfold list-engines
ragfold compare examples/corpus.json examples/queries.json --engines text-rag,bm25 --top-k 1
ragfold bench examples --engines text-rag,bm25
```

`corpus.json` is a list of objects with `id` and `text`. `queries.json` is a list
of objects with `id`, `query`, `relevant_ids`, and optional `answers`.

## Evaluation

Ragfold reports retrieval and answer metrics using the same convention as
docfold: predicted value first, reference value second, and higher is better.

| Metric | What it measures |
|---|---|
| Recall@k | Share of gold documents found in top-k |
| Precision@k | Share of top-k results that are gold |
| Hit@k | Whether any gold document appears in top-k |
| MRR | Reciprocal rank of first relevant result |
| nDCG@k | Ranking quality with early hits rewarded |
| MAP | Mean average precision over queries |
| Answer EM/F1 | End-to-end exact match and token overlap when gold answers exist |
| Token cost | Provider-reported or adapter-estimated cost, reported separately |

Reference-free faithfulness is intentionally a gated `slow` hook because it
requires an LLM judge.

## Architecture

```
Corpus + Queries
      |
      v
EngineRouter  -- optional reranker --> RetrievalResult / RagAnswer
      |
      +-- lexical: text-rag, bm25
      +-- dense: sentence-transformers, OpenAI, Cohere, Voyage
      +-- visual: ColPali, ColQwen2, PixelRAG, DSE
      +-- framework: LlamaIndex, Haystack, txtai
      +-- vector stores: in-memory, FAISS, Qdrant, Chroma, pgvector
```

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

All heavy/cloud/GPU/model tests must be marked `slow` and must not run in the
default CI path.
