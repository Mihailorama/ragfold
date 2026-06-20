"""Dataset adapters and gated public-dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path

from ragfold.evaluation.runner import Dataset, QueryExample


def load_json_dataset(corpus_path: str | Path, queries_path: str | Path, name: str = "json") -> Dataset:
    corpus_data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    query_data = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    queries = [QueryExample.from_mapping(item) for item in query_data]
    return Dataset(name=name, corpus=corpus_data, queries=queries)


def load_vidore_slice(cache_dir: str | Path, download: bool = False) -> Dataset:
    if not download:
        raise RuntimeError(
            "ViDoRe slice download is gated as slow. Pass download=True only in slow tests or "
            "explicit benchmark runs."
        )
    raise NotImplementedError(
        "ViDoRe download/normalization is reserved for slow integration tests to keep CI offline."
    )
