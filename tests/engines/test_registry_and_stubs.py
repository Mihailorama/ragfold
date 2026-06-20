import pytest

from ragfold.engines import DEFAULT_ENGINE_NAMES, build_default_router
from ragfold.engines.visual import ColPaliEngine


def test_default_router_lists_every_declared_engine_without_crashing():
    router = build_default_router()

    listed = {engine["name"] for engine in router.list_engines()}

    assert set(DEFAULT_ENGINE_NAMES) <= listed
    assert "text-rag" in listed
    assert "bm25" in listed
    assert "colpali" in listed
    assert "llamaindex" in listed


@pytest.mark.asyncio
async def test_unavailable_stub_engines_are_skipped_in_compare_sweeps(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = build_default_router()

    results = await router.compare(
        [{"id": "doc", "text": "Paris is in France."}],
        [{"id": "q1", "query": "France city", "relevant_ids": ["doc"]}],
        engines=["text-rag", "openai", "colqwen2"],
        top_k=1,
    )

    assert set(results) == {"text-rag"}
    assert results["text-rag"][0].passages[0].document_id == "doc"


class ToyVisualEmbedder:
    def encode(self, texts):
        return [
            [float("diagram" in text.lower()), float("table" in text.lower())]
            for text in texts
        ]


@pytest.mark.asyncio
async def test_colpali_supports_injected_embedder_path():
    engine = ColPaliEngine(embedder=ToyVisualEmbedder())

    result = await engine.retrieve(
        [{"id": "visual", "text": "A screenshot with a diagram"}],
        "find diagram",
        top_k=1,
    )

    assert engine.is_available() is True
    assert result.passages[0].document_id == "visual"
    assert engine.capabilities.ocr_free is True
    assert engine.capabilities.retrieval_method == "late-interaction"
