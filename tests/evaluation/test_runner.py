import pytest

from ragfold.engines.router import EngineRouter
from ragfold.engines.text_rag import TextRagEngine
from ragfold.evaluation.runner import Dataset, EvaluationReport, EvaluationRunner, QueryExample


@pytest.fixture
def dataset():
    return Dataset(
        name="mini",
        corpus=[
            {"id": "paris", "text": "Paris is the capital of France."},
            {"id": "berlin", "text": "Berlin is the capital of Germany."},
        ],
        queries=[
            QueryExample(
                id="q1",
                query="capital of France",
                relevant_ids=["paris"],
                answers=["Paris"],
            )
        ],
    )


@pytest.mark.asyncio
async def test_runner_report_contains_accuracy_latency_and_cost_columns(dataset):
    router = EngineRouter([TextRagEngine()])
    runner = EvaluationRunner(router)

    report = await runner.run(dataset, engines=["text-rag"], top_k=1)

    assert isinstance(report, EvaluationReport)
    assert report.rows[0].engine_name == "text-rag"
    assert report.rows[0].recall_at_k == 1.0
    assert report.rows[0].latency_ms >= 0
    assert report.rows[0].token_cost == 0.0

    markdown = report.to_markdown()
    assert "| Engine | Queries | Recall@1 | Precision@1 | MAP | Hit@1 | MRR | nDCG@1 |"
    assert "text-rag" in markdown


@pytest.mark.asyncio
async def test_runner_skips_unavailable_engines_without_crashing(dataset):
    router = EngineRouter.from_engine_names(["text-rag", "openai"])
    runner = EvaluationRunner(router)

    report = await runner.run(dataset, engines=["text-rag", "openai"], top_k=1)

    assert [row.engine_name for row in report.rows] == ["text-rag"]
    assert report.skipped["openai"] == "unavailable"
