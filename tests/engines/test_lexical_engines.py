import pytest

from ragfold.engines.bm25 import BM25Engine
from ragfold.engines.text_rag import TextRagEngine

CORPUS = [
    {"id": "paris", "text": "Paris is the capital of France and has the Louvre."},
    {"id": "berlin", "text": "Berlin is the capital of Germany."},
    {"id": "python", "text": "Python packages can be installed with pip."},
]


@pytest.mark.asyncio
async def test_text_rag_retrieves_relevant_passage_without_optional_deps():
    engine = TextRagEngine()

    result = await engine.retrieve(CORPUS, "which city has the Louvre?", top_k=2)

    assert engine.is_available() is True
    assert result.engine_name == "text-rag"
    assert result.passages[0].document_id == "paris"
    assert result.passages[0].rank == 1
    assert result.passages[0].score > 0
    assert engine.capabilities.retrieval_method == "lexical"


@pytest.mark.asyncio
async def test_bm25_retrieves_relevant_passage_without_optional_deps():
    engine = BM25Engine()

    result = await engine.retrieve(CORPUS, "France capital Louvre", top_k=2)

    assert engine.is_available() is True
    assert result.engine_name == "bm25"
    assert result.passages[0].document_id == "paris"
    assert result.passages[0].score > result.passages[1].score
    assert engine.capabilities.retrieval_method == "lexical"


@pytest.mark.asyncio
async def test_lexical_engines_handle_empty_corpus():
    for engine in [TextRagEngine(), BM25Engine()]:
        result = await engine.retrieve([], "anything", top_k=3)
        assert result.passages == []
        assert result.processing_time_ms >= 0
