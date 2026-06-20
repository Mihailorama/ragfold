import pytest

from ragfold.engines import DEFAULT_ENGINE_NAMES, build_default_router
from ragfold.engines.base import RetrievalResult
from ragfold.engines.github_rag import AgenticFileSearchEngine, LightRAGEngine, RAGAnythingEngine


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


class AsyncDictClient:
    def __init__(self):
        self.calls = []

    async def retrieve(self, *, corpus, query, top_k, **kwargs):
        self.calls.append({"corpus": corpus, "query": query, "top_k": top_k, "kwargs": kwargs})
        return {
            "answer": "LightRAG found doc-1",
            "citations": ["doc-1"],
            "passages": [
                {"id": "doc-1", "text": "graph context", "score": 0.8, "metadata": {"kind": "kg"}}
            ],
            "token_cost": 0.012,
        }


@pytest.mark.asyncio
async def test_lightrag_injected_async_client_dict_normalizes_to_retrieval_result():
    client = AsyncDictClient()
    engine = LightRAGEngine(client=client)

    result = await engine.retrieve(
        [{"id": "doc-1", "text": "graph context"}],
        "graph query",
        top_k=1,
        mode="hybrid",
    )

    assert isinstance(result, RetrievalResult)
    assert result.engine_name == "lightrag"
    assert result.passages[0].document_id == "doc-1"
    assert result.passages[0].metadata["kind"] == "kg"
    assert result.answer.answer == "LightRAG found doc-1"
    assert result.answer.citations == ["doc-1"]
    assert result.token_cost == 0.012
    assert client.calls[0]["kwargs"]["mode"] == "hybrid"


class SyncListClient:
    def search(self, *, corpus, query, top_k, **kwargs):
        return [
            {"document_id": "doc-a", "text": "agentic hit", "score": 1.0},
            {"document_id": "doc-b", "text": "agentic backup", "score": 0.5},
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_cls", [RAGAnythingEngine, AgenticFileSearchEngine])
async def test_github_rag_injected_sync_client_list_normalizes_passages(engine_cls):
    engine = engine_cls(client=SyncListClient())

    result = await engine.retrieve([{"id": "doc-a", "text": "agentic hit"}], "agentic", top_k=2)

    assert result.engine_name == engine.name
    assert [passage.document_id for passage in result.passages] == ["doc-a", "doc-b"]
    assert [passage.rank for passage in result.passages] == [1, 2]
