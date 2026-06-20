import json

import pytest

from ragfold.evaluation.datasets import load_json_dataset, load_vidore_slice


def test_load_json_dataset_reads_corpus_and_queries(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    queries_path = tmp_path / "queries.json"
    corpus_path.write_text(json.dumps([{"id": "doc", "text": "Example text"}]), encoding="utf-8")
    queries_path.write_text(
        json.dumps([{"id": "q1", "query": "example", "relevant_ids": ["doc"]}]),
        encoding="utf-8",
    )

    dataset = load_json_dataset(corpus_path, queries_path, name="fixture")

    assert dataset.name == "fixture"
    assert dataset.queries[0].relevant_ids == ["doc"]


def test_vidore_loader_is_download_gated_by_default(tmp_path):
    with pytest.raises(RuntimeError, match="slow"):
        load_vidore_slice(tmp_path, download=False)
