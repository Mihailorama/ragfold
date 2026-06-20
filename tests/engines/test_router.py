import asyncio

import pytest

from ragfold.engines.base import RetrievalResult
from ragfold.engines.bm25 import BM25Engine
from ragfold.engines.router import EngineRouter
from ragfold.engines.text_rag import TextRagEngine
from ragfold.preprocessing import RecursiveChunker
from ragfold.rerankers import BaseReranker

CORPUS = [
    {"id": "a", "text": "Alpha invoice payment"},
    {"id": "b", "text": "Beta contract renewal"},
]


def test_from_engine_names_builds_router_with_known_engines():
    router = EngineRouter.from_engine_names(["bm25", "text-rag"])

    assert [engine["name"] for engine in router.list_engines()] == ["bm25", "text-rag"]


def test_from_engine_names_rejects_unknown_engine():
    with pytest.raises(ValueError, match="Unknown engine"):
        EngineRouter.from_engine_names(["missing"])


def test_auto_select_prefers_first_available_engine():
    router = EngineRouter([BM25Engine(), TextRagEngine()])

    assert router.select().name == "bm25"


@pytest.mark.asyncio
async def test_compare_runs_available_engines_for_all_queries():
    router = EngineRouter([TextRagEngine(), BM25Engine()])
    queries = [
        {"id": "q1", "query": "invoice", "relevant_ids": ["a"]},
        {"id": "q2", "query": "contract", "relevant_ids": ["b"]},
    ]

    results = await router.compare(CORPUS, queries, top_k=1)

    assert set(results) == {"text-rag", "bm25"}
    assert len(results["text-rag"]) == 2
    assert all(isinstance(item, RetrievalResult) for item in results["bm25"])


class SlowEngine(TextRagEngine):
    @property
    def name(self):
        return "slow-text"

    async def retrieve(self, corpus, query, top_k=5, **kwargs):
        await asyncio.sleep(0.01)
        return await super().retrieve(corpus, query, top_k=top_k, **kwargs)


@pytest.mark.asyncio
async def test_process_batch_uses_bounded_concurrency():
    router = EngineRouter([SlowEngine()])
    queries = [{"id": f"q{i}", "query": "invoice", "relevant_ids": ["a"]} for i in range(4)]

    batch = await router.process_batch(CORPUS, queries, concurrency=2, engine_hint="slow-text")

    assert batch.total == 4
    assert batch.succeeded == 4
    assert not batch.errors


class ReverseReranker(BaseReranker):
    def is_available(self):
        return True

    def rerank(self, query, passages):
        reranked = list(reversed(passages))
        for idx, passage in enumerate(reranked, start=1):
            passage.rank = idx
        return reranked


@pytest.mark.asyncio
async def test_router_applies_available_reranker():
    router = EngineRouter([TextRagEngine()], reranker=ReverseReranker())

    result = await router.retrieve(CORPUS, "invoice contract", top_k=2, engine_hint="text-rag")

    assert [passage.rank for passage in result.passages] == [1, 2]
    assert result.metadata["reranker"] == "ReverseReranker"


class RecordingEngine(TextRagEngine):
    @property
    def name(self):
        return "recording"

    async def retrieve(self, corpus, query, top_k=5, **kwargs):
        self.last_corpus = corpus
        return await super().retrieve(corpus, query, top_k=top_k, **kwargs)


@pytest.mark.asyncio
async def test_router_applies_chunker_before_retrieval():
    engine = RecordingEngine()
    router = EngineRouter(
        [engine],
        chunker=RecursiveChunker(chunk_size=2, chunk_overlap=0),
    )

    result = await router.retrieve(
        [{"id": "doc", "text": "alpha beta gamma", "metadata": {"source": "fixture"}}],
        "gamma",
        top_k=3,
        engine_hint="recording",
    )

    assert [item["id"] for item in engine.last_corpus] == ["doc#chunk-0", "doc#chunk-1"]
    assert engine.last_corpus[0]["metadata"]["source_document_id"] == "doc"
    assert engine.last_corpus[0]["metadata"]["source"] == "fixture"
    assert result.metadata["chunker"] == "RecursiveChunker"


class SuffixCompressor:
    def is_available(self):
        return True

    def compress(self, passages, query, **kwargs):
        for passage in passages:
            passage.text = f"{passage.text} [compressed for {query}]"
        return passages


@pytest.mark.asyncio
async def test_router_applies_compressor_after_retrieval():
    router = EngineRouter([TextRagEngine()], compressor=SuffixCompressor())

    result = await router.retrieve(CORPUS, "invoice", top_k=1, engine_hint="text-rag")

    assert result.passages[0].text.endswith("[compressed for invoice]")
    assert result.metadata["compressor"] == "SuffixCompressor"


@pytest.mark.asyncio
async def test_compare_uses_router_pipeline_stages():
    router = EngineRouter(
        [TextRagEngine()],
        chunker=RecursiveChunker(chunk_size=2, chunk_overlap=0),
        compressor=SuffixCompressor(),
    )

    results = await router.compare(
        [{"id": "doc", "text": "alpha beta gamma"}],
        [{"id": "q1", "query": "gamma", "relevant_ids": ["doc#chunk-1"]}],
        engines=["text-rag"],
        top_k=1,
    )

    result = results["text-rag"][0]
    assert result.metadata["chunker"] == "RecursiveChunker"
    assert result.metadata["compressor"] == "SuffixCompressor"
    assert result.passages[0].text.endswith("[compressed for gamma]")
