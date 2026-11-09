---
purpose: "Key-free retrieval evals in CI from recorded embeddings"
status: "OPEN"
priority: "P1"
created: "2026-11-09"
---

# Feature: Recorded Embeddings for Evaluation

## Problem
Retrieval quality (recall@k, MRR) can only be measured with real embeddings,
but ragfold's CI and downstream CIs (saas-agent-starter runs `pnpm eval
--fake-llm`) must run with no keys, network or model downloads. A fake
embedder makes the metrics meaningless.

## Proposed Solution
In `src/ragfold/evaluation/recorded.py`, stdlib only:

- `RecordingEmbedder(inner: Embedder, path)`: delegates to a real embedder and
  appends `{"model", "input_type", "sha256", "vector"}` per input to a JSONL
  file. Run once on a developer machine (`slow`).
- `RecordedEmbedder(path, *, model)`: an `Embedder` that answers only from the
  file, keyed by `(model, input_type, sha256(text))`. A missing key raises
  `KeyError` naming the hash, never the text. The file holds no text, only
  hashes and vectors.
- `evaluate_retrieval(retriever, dataset, *, k_values=(1, 5, 10)) -> RetrievalReport`:
  runs each query through a `Retriever` (`RETRIEVER.md`) and scores with the
  existing `ragfold.evaluation.metrics` (predicted first, reference second).
  Datasets reuse `evaluation/datasets.py` (query, relevant chunk or document ids).
- CLI: `ragfold eval-retrieval <corpus> <queries> --recorded embeddings.jsonl --model <id> [--mode hybrid]`
  printing the usual Markdown table; exit code 1 below `--min-recall`.
- Recorded files are generated artefacts for the user's repo; ragfold ships one
  tiny fixture for its own tests.

## Affected Files
- `src/ragfold/evaluation/recorded.py` - new.
- `src/ragfold/cli.py` - `eval-retrieval`.
- `tests/evaluation/test_recorded.py` - new; `examples/recorded_embeddings.jsonl` - fixture.

## Test Plan

### Unit / Functional Tests
Write first (RED): `tests/evaluation/test_recorded.py::test_recorded_embedder_replays_vectors_by_hash`.
- [ ] Recording then replaying gives identical vectors and no inner calls.
- [ ] Unknown input raises `KeyError` whose message has the hash and no text.
- [ ] Recorded file contains no input text (sentinel check).
- [ ] Different `model` or `input_type` is a different key.
- [ ] `evaluate_retrieval` on the fixture reports the hand-computed recall@k and MRR.
- [ ] CLI exits 1 below `--min-recall`.

### Integration / E2E Tests
- [ ] `slow`: record with `SentenceTransformersEmbedder` and replay in a fresh process.

### Test Commands
```bash
pytest tests/evaluation -m "not slow"
ragfold eval-retrieval examples/corpus.json examples/queries.json --recorded examples/recorded_embeddings.jsonl --model fixture
```

## Edge Cases
- Corpus edited after recording: the changed chunk misses -> clear error telling
  the user to re-record.
- Duplicate lines in the JSONL: last one wins.
- Large files: streamed line by line into a dict; no numpy.

## Out of Scope
- LLM-judged answer metrics (faithfulness): the app's Promptfoo harness.
- Recording rerank scores (follow-up if rerank evals are needed).
