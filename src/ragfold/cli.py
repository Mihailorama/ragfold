"""ragfold CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ragfold.engines import build_default_router
from ragfold.engines.router import EngineRouter
from ragfold.evaluation.datasets import load_json_dataset
from ragfold.evaluation.runner import EvaluationRunner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ragfold",
        description="Compare RAG and information-extraction engines.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list-engines", help="List registered engines and availability")

    compare_p = sub.add_parser("compare", help="Compare engines on a corpus/query JSON pair")
    compare_p.add_argument("corpus", help="Path to corpus JSON")
    compare_p.add_argument("queries", help="Path to queries JSON")
    compare_p.add_argument("--engines", help="Comma-separated engine names")
    compare_p.add_argument("--top-k", type=int, default=5)

    bench_p = sub.add_parser("bench", help="Run a named or path-based benchmark")
    bench_p.add_argument(
        "dataset",
        help="'examples' or a directory containing corpus.json/queries.json",
    )
    bench_p.add_argument("--engines", help="Comma-separated engine names")
    bench_p.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(0)
    if args.command == "list-engines":
        _cmd_list_engines()
    elif args.command == "compare":
        asyncio.run(_cmd_compare(args))
    elif args.command == "bench":
        asyncio.run(_cmd_bench(args))


def _build_router() -> EngineRouter:
    return build_default_router()


def _cmd_list_engines() -> None:
    router = _build_router()
    print("| name | modality | available | method | type | license | speed | cost |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for engine in router.list_engines():
        available = "yes" if engine["available"] else "no"
        print(
            f"| {engine['name']} | {engine['modality']} | {available} | "
            f"{engine['retrieval_method']} | {engine['type']} | {engine['license']} | "
            f"{engine['speed']} | {engine['cost']} |"
        )


async def _cmd_compare(args: argparse.Namespace) -> None:
    dataset = load_json_dataset(args.corpus, args.queries, name="compare")
    router = _build_router()
    engines = _split_engines(args.engines)
    report = await EvaluationRunner(router).run(dataset, engines=engines, top_k=args.top_k)
    print(report.to_markdown())


async def _cmd_bench(args: argparse.Namespace) -> None:
    corpus_path, queries_path = _dataset_paths(args.dataset)
    dataset = load_json_dataset(corpus_path, queries_path, name=Path(args.dataset).name)
    router = _build_router()
    engines = _split_engines(args.engines)
    report = await EvaluationRunner(router).run(dataset, engines=engines, top_k=args.top_k)
    print(report.to_markdown())


def _split_engines(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _dataset_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "examples":
        root = Path(__file__).resolve().parents[2] / "examples"
    else:
        root = Path(dataset)
    return root / "corpus.json", root / "queries.json"


if __name__ == "__main__":
    main(sys.argv[1:])
