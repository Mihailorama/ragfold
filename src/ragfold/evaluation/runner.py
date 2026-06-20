"""Evaluation runner for retrieval and answer benchmarks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ragfold.engines.base import CorpusInput, RetrievalResult
from ragfold.engines.router import EngineRouter
from ragfold.evaluation.metrics import (
    answer_exact_match,
    answer_f1,
    average_precision,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass
class QueryExample:
    id: str
    query: str
    relevant_ids: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> QueryExample:
        return cls(
            id=str(data.get("id") or data.get("query_id") or data.get("query")),
            query=str(data.get("query") or data.get("text") or ""),
            relevant_ids=[str(item) for item in data.get("relevant_ids", [])],
            answers=[str(item) for item in data.get("answers", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Dataset:
    name: str
    corpus: CorpusInput
    queries: list[QueryExample]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRow:
    engine_name: str
    queries: int
    k: int
    recall_at_k: float
    precision_at_k: float
    map: float
    hit_at_k: float
    mrr: float
    ndcg_at_k: float
    answer_em: float | None = None
    answer_f1: float | None = None
    latency_ms: float = 0.0
    token_cost: float = 0.0
    errors: int = 0


@dataclass
class EvaluationReport:
    dataset_name: str
    rows: list[EvaluationRow] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "rows": [asdict(row) for row in self.rows],
            "skipped": self.skipped,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        headers = [
            "Engine",
            "Queries",
            f"Recall@{self.rows[0].k if self.rows else 'k'}",
            f"Precision@{self.rows[0].k if self.rows else 'k'}",
            "MAP",
            f"Hit@{self.rows[0].k if self.rows else 'k'}",
            "MRR",
            f"nDCG@{self.rows[0].k if self.rows else 'k'}",
            "Answer EM",
            "Answer F1",
            "Latency ms",
            "Token cost",
            "Errors",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in self.rows:
            values = [
                row.engine_name,
                str(row.queries),
                _fmt(row.recall_at_k),
                _fmt(row.precision_at_k),
                _fmt(row.map),
                _fmt(row.hit_at_k),
                _fmt(row.mrr),
                _fmt(row.ndcg_at_k),
                _fmt(row.answer_em),
                _fmt(row.answer_f1),
                f"{row.latency_ms:.1f}",
                f"{row.token_cost:.4f}",
                str(row.errors),
            ]
            lines.append("| " + " | ".join(values) + " |")
        if self.skipped:
            lines.append("")
            lines.append("| Skipped engine | Reason |")
            lines.append("| --- | --- |")
            for name, reason in self.skipped.items():
                lines.append(f"| {name} | {reason} |")
        return "\n".join(lines)


class EvaluationRunner:
    def __init__(self, router: EngineRouter) -> None:
        self.router = router

    async def run(
        self,
        dataset: Dataset,
        engines: list[str] | None = None,
        top_k: int = 5,
    ) -> EvaluationReport:
        report = EvaluationReport(dataset_name=dataset.name)
        engine_names = engines or [row["name"] for row in self.router.list_engines()]

        for engine_name in engine_names:
            engine = self.router.get(engine_name)
            if engine is None:
                report.skipped[engine_name] = "unknown"
                continue
            if not engine.is_available():
                report.skipped[engine_name] = "unavailable"
                continue

            results: list[RetrievalResult] = []
            errors = 0
            for query in dataset.queries:
                try:
                    results.append(
                        await engine.retrieve(dataset.corpus, query.query, top_k=top_k)
                    )
                except Exception:
                    errors += 1
            if not results:
                if errors:
                    report.skipped[engine_name] = "errors"
                continue
            report.rows.append(_summarize(engine_name, dataset.queries, results, top_k, errors))
        return report


def _summarize(
    engine_name: str,
    queries: list[QueryExample],
    results: list[RetrievalResult],
    top_k: int,
    errors: int,
) -> EvaluationRow:
    predicted_ids = [
        [passage.document_id for passage in result.passages[:top_k]]
        for result in results
    ]
    references = [query.relevant_ids for query in queries[: len(results)]]
    answer_refs = [query.answers for query in queries[: len(results)]]

    answer_em_scores: list[float] = []
    answer_f1_scores: list[float] = []
    for result, refs in zip(results, answer_refs, strict=True):
        if not refs:
            continue
        predicted_answer = result.answer.answer if result.answer else ""
        answer_em_scores.append(max(answer_exact_match(predicted_answer, ref) for ref in refs))
        answer_f1_scores.append(max(answer_f1(predicted_answer, ref) for ref in refs))

    count = len(results)
    return EvaluationRow(
        engine_name=engine_name,
        queries=count,
        k=top_k,
        recall_at_k=sum(recall_at_k(pred, ref, top_k) for pred, ref in zip(predicted_ids, references, strict=True)) / count,
        precision_at_k=sum(precision_at_k(pred, ref, top_k) for pred, ref in zip(predicted_ids, references, strict=True)) / count,
        map=sum(average_precision(pred, ref, top_k) for pred, ref in zip(predicted_ids, references, strict=True)) / count,
        hit_at_k=sum(hit_at_k(pred, ref, top_k) for pred, ref in zip(predicted_ids, references, strict=True)) / count,
        mrr=sum(mean_reciprocal_rank(pred, ref) for pred, ref in zip(predicted_ids, references, strict=True)) / count,
        ndcg_at_k=sum(ndcg_at_k(pred, ref, top_k) for pred, ref in zip(predicted_ids, references, strict=True)) / count,
        answer_em=(sum(answer_em_scores) / len(answer_em_scores) if answer_em_scores else None),
        answer_f1=(sum(answer_f1_scores) / len(answer_f1_scores) if answer_f1_scores else None),
        latency_ms=sum(result.processing_time_ms for result in results) / count,
        token_cost=sum(result.token_cost for result in results),
        errors=errors,
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
