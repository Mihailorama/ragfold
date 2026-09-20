import json

import pytest

from ragfold.cli import _build_router, main
from ragfold.engines.router import EngineRouter


def test_build_router_returns_router():
    assert isinstance(_build_router(), EngineRouter)


def test_list_engines_prints_name_modality_and_availability(capsys):
    main(["list-engines"])

    output = capsys.readouterr().out
    assert "| name | modality | available |" in output
    assert "text-rag" in output
    assert "bm25" in output


def test_compare_prints_populated_markdown_table(tmp_path, capsys):
    corpus = tmp_path / "corpus.json"
    queries = tmp_path / "queries.json"
    corpus.write_text(
        json.dumps([
            {"id": "paris", "text": "Paris is the capital of France."},
            {"id": "pip", "text": "pip installs Python packages."},
        ]),
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps([
            {"id": "q1", "query": "France capital", "relevant_ids": ["paris"], "answers": ["Paris"]}
        ]),
        encoding="utf-8",
    )

    main(["compare", str(corpus), str(queries), "--engines", "text-rag,bm25", "--top-k", "1"])

    output = capsys.readouterr().out
    assert "| Engine | Queries | Recall@1 | Precision@1 |" in output
    assert "text-rag" in output
    assert "bm25" in output


def test_bench_accepts_examples_dataset(capsys):
    main(["bench", "examples", "--engines", "text-rag", "--top-k", "1"])

    output = capsys.readouterr().out
    assert "text-rag" in output


def test_hybrid_prints_fused_ranking(tmp_path, capsys):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps([
            {"id": "paris", "text": "Paris is the capital of France."},
            {"id": "pip", "text": "pip installs Python packages."},
        ]),
        encoding="utf-8",
    )

    main(["hybrid", str(corpus), "France capital", "--engines", "text-rag,bm25", "--top-k", "1"])

    output = capsys.readouterr().out
    assert "rrf(text-rag,bm25)" in output
    assert "paris" in output


def test_hybrid_accepts_weights(tmp_path, capsys):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps([
            {"id": "paris", "text": "Paris is the capital of France."},
            {"id": "pip", "text": "pip installs Python packages."},
        ]),
        encoding="utf-8",
    )

    main([
        "hybrid", str(corpus), "France capital",
        "--engines", "text-rag,bm25", "--top-k", "1", "--weights", "text-rag=2,bm25=1",
    ])

    output = capsys.readouterr().out
    assert "paris" in output


def test_no_args_prints_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 0
    assert "list-engines" in capsys.readouterr().out
