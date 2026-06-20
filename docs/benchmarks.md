---
purpose: "Benchmark methodology and research-based engine profiles for ragfold"
updated: "2026-06-20"
---

# Ragfold Benchmarks

This document defines how ragfold compares retrieval and information-extraction
engines. Values below are a research-based estimate from public documentation,
package behavior, and common deployment patterns. They are starting points for
planning, not replacement measurements for your own corpus.

## Methodology

1. Normalize each corpus entry to `{id, text, metadata}`.
2. Run each available engine over the same query set with the same `top_k`.
3. Skip unavailable engines and record the reason instead of failing the sweep.
4. Compute metrics with predicted IDs first and reference IDs second.
5. Report quality, latency, and token cost separately.
6. Keep default CI offline: no downloads, network, SaaS calls, external vector
   services, model weights, or GPU.

## Metrics

| Metric | Direction | Notes |
|---|---:|---|
| Recall@k | higher is better | Best for coverage-sensitive retrieval |
| Precision@k | higher is better | Best when top-k is passed to a small context window |
| Hit@k | higher is better | Binary per query, useful for dashboards |
| MRR | higher is better | Rewards the first relevant result |
| nDCG@k | higher is better | Rewards ranking relevant docs early |
| MAP | higher is better | Stable summary across query sets |
| Answer EM/F1 | higher is better | Only when gold answers exist |
| Faithfulness | higher is better | LLM-judge hook, `slow`, reference-free |
| Token cost | reported separately | Not mixed into quality scores |

## Dataset Matrix

| Dataset | Modality | Default CI | Purpose | Loader |
|---|---|:---:|---|---|
| `examples` | text | yes | Smoke test for CLI and docs | local JSON |
| Custom JSON | text | yes | User-owned corpora | `load_json_dataset` |
| ViDoRe slice | visual/text | no, `slow` | OCR-free visual retrieval | `load_vidore_slice(download=True)` |
| UNIDOC-BENCH slice | visual/text | no, `slow` | multi-format document QA | future gated loader |

## Per-Engine Profiles

| Engine | Profile | Strength | Risk |
|---|---|---|---|
| `text-rag` | Pure-Python TF-IDF | Deterministic offline baseline | Weak semantic matching |
| `bm25` | Pure-Python BM25 with optional `rank-bm25` | Strong lexical baseline | No semantic matching |
| `sentence-transformers` | Local embedding model | Semantic retrieval, no SaaS | Model downloads and memory |
| `openai` | Hosted embeddings | Low ops, strong general embeddings | API cost, data egress policy |
| `cohere-embed` | Hosted embeddings | Enterprise RAG ecosystem | API cost and key management |
| `voyage` | Hosted embeddings | Retrieval-focused embedding family | API cost and key management |
| `colpali` | Visual late interaction | OCR-free page retrieval | GPU/model weight requirements |
| `colqwen2` | Visual late interaction | OCR-free page retrieval | GPU/model weight requirements |
| `pixelrag` | Visual embedding/VLM research | Screenshot-first retrieval | Research maturity |
| `dse` | Document Screenshot Embedding | OCR-free screenshots | Implementation variance |
| `llamaindex` | Framework adapter | Reuse existing retrievers | Framework-specific result shapes |
| `haystack` | Framework adapter | Reuse production pipelines | Framework-specific result shapes |
| `txtai` | Framework adapter | Lightweight app integration | Framework-specific result shapes |

## Hardware Requirements

| Engine family | CPU | RAM | GPU | Network |
|---|---:|---:|---:|---:|
| `text-rag`, `bm25` | yes | low | no | no |
| Local dense | yes | medium/high | optional | only for downloads |
| SaaS dense/rerank | yes | low | no | yes |
| Visual late-interaction | yes | high | recommended/required | only for downloads |
| Vector services | yes | service-dependent | no | maybe |

## Cost per 1K Queries

These are research-based estimate bands for planning. Tokenized SaaS costs vary
with chunk size, query length, rerank depth, region, and committed-use discounts.

| Engine family | Research-based estimate per 1K queries | Cost driver |
|---|---:|---|
| `text-rag`, `bm25` | $0 | local CPU |
| Local dense | $0 API cost, infra only | CPU/GPU time and storage |
| OpenAI embeddings | often cents for small corpora | input tokens |
| Cohere Embed | often cents for small corpora | input tokens |
| Cohere Rerank | can dominate at high candidate counts | searches/chunks ranked |
| Voyage embeddings | often cents for small corpora | input tokens |
| Visual local models | infra cost only | GPU time, page images, storage |
| Vector stores | $0 local to managed-service fees | storage, QPS, replicas |

## Slow Tests

Real inference tests for model, SaaS, GPU, dataset download, and external
service paths must be marked:

```python
@pytest.mark.slow
```

The required lightweight CI command is:

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

## References

- OpenAI embedding pricing and model cards: https://developers.openai.com/api/docs/models/text-embedding-3-small
- Cohere pricing model and rerank search-unit notes: https://cohere.com/pricing
- Voyage AI embeddings and pricing docs: https://docs.voyageai.com/docs/embeddings
- Sentence Transformers license and retrieval/reranking docs: https://github.com/huggingface/sentence-transformers
- ColPali repository and visual retrieval paper implementation: https://github.com/illuin-tech/colpali
- FAISS license and similarity-search docs: https://github.com/facebookresearch/faiss
