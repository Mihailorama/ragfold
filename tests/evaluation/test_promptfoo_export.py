from ragfold.evaluation.runner import Dataset, QueryExample


def test_promptfoo_export_contains_dataset_queries_and_expected_answers():
    from ragfold.evaluation.promptfoo import to_promptfoo_config

    dataset = Dataset(
        name="mini",
        corpus=[{"id": "doc", "text": "Paris is the capital of France."}],
        queries=[
            QueryExample(
                id="q1",
                query="capital of France",
                relevant_ids=["doc"],
                answers=["Paris"],
            )
        ],
    )

    config = to_promptfoo_config(dataset, provider="python:ragfold")

    assert "providers:" in config
    assert "python:ragfold" in config
    assert "capital of France" in config
    assert "Paris" in config
