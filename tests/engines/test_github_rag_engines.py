import pytest

from ragfold.engines import DEFAULT_ENGINE_NAMES, build_default_router


def test_github_rag_engines_are_registered_and_gated():
    router = build_default_router()
    engine_names = {row["name"] for row in router.list_engines()}

    assert {"lightrag", "rag-anything", "agentic-file-search"} <= engine_names
    assert {"lightrag", "rag-anything", "agentic-file-search"} <= set(DEFAULT_ENGINE_NAMES)

    by_name = {row["name"]: row for row in router.list_engines()}
    assert by_name["lightrag"]["available"] is False
    assert by_name["rag-anything"]["modality"] == "multimodal"
    assert by_name["agentic-file-search"]["retrieval_method"] == "agentic"


@pytest.mark.asyncio
async def test_github_rag_engines_skip_cleanly_in_compare_sweeps():
    router = build_default_router()

    results = await router.compare(
        [{"id": "doc", "text": "Ragfold integrates retrieval engines."}],
        [{"id": "q1", "query": "retrieval engines", "relevant_ids": ["doc"]}],
        engines=["text-rag", "lightrag", "rag-anything", "agentic-file-search"],
        top_k=1,
    )

    assert set(results) == {"text-rag"}
    assert results["text-rag"][0].passages[0].document_id == "doc"
