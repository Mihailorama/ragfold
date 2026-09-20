"""Tests for deterministic, CI-light RRF weight tuning."""

from __future__ import annotations

import pytest

from ragfold.engines.bm25 import BM25Engine
from ragfold.engines.router import EngineRouter
from ragfold.engines.text_rag import TextRagEngine
from ragfold.fusion_tuning import tune_rrf_weights

CORPUS = [
    {"id": "a", "text": "invoice invoice payment due"},
    {"id": "b", "text": "beta contract renewal terms"},
    {"id": "c", "text": "gamma shipping logistics note"},
]
QUERIES = [
    {"id": "q1", "query": "invoice payment", "relevant_ids": ["a"]},
    {"id": "q2", "query": "contract renewal", "relevant_ids": ["b"]},
]


@pytest.mark.asyncio
async def test_tune_returns_a_weight_for_each_engine_and_a_score():
    router = EngineRouter([BM25Engine(), TextRagEngine()])

    result = await tune_rrf_weights(
        router, CORPUS, QUERIES, engines=["bm25", "text-rag"], top_k=2
    )

    assert set(result.weights) == {"bm25", "text-rag"}
    assert all(isinstance(w, float) for w in result.weights.values())
    assert 0.0 <= result.score <= 1.0
    assert result.trials  # every grid point evaluated


@pytest.mark.asyncio
async def test_tuning_is_deterministic():
    router = EngineRouter([BM25Engine(), TextRagEngine()])

    first = await tune_rrf_weights(router, CORPUS, QUERIES, engines=["bm25", "text-rag"], top_k=2)
    second = await tune_rrf_weights(router, CORPUS, QUERIES, engines=["bm25", "text-rag"], top_k=2)

    assert first.weights == second.weights
    assert first.score == second.score


@pytest.mark.asyncio
async def test_best_score_is_at_least_uniform_weights():
    # Uniform weights (all 1.0) are always in the grid, so the winner can never
    # be worse than the unweighted baseline.
    router = EngineRouter([BM25Engine(), TextRagEngine()])

    result = await tune_rrf_weights(
        router,
        CORPUS,
        QUERIES,
        engines=["bm25", "text-rag"],
        top_k=2,
        candidate_weights=(0.5, 1.0, 2.0),
    )
    uniform = next(
        score for weights, score in result.trials if set(weights.values()) == {1.0}
    )

    assert result.score >= uniform


@pytest.mark.asyncio
async def test_tuned_weights_feed_retrieve_hybrid():
    router = EngineRouter([BM25Engine(), TextRagEngine()])

    result = await tune_rrf_weights(router, CORPUS, QUERIES, engines=["bm25", "text-rag"], top_k=2)
    retrieval = await router.retrieve_hybrid(
        CORPUS, "invoice payment", engines=["bm25", "text-rag"], top_k=2, weights=result.weights
    )

    assert retrieval.passages[0].document_id == "a"
    assert retrieval.metadata["fusion"]["weights"] == result.weights
