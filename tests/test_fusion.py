"""Standalone unit tests for Reciprocal Rank Fusion.

No engines are involved. Every expected score is written as the closed-form
RRF arithmetic (sum of 1/(k + rank)) so the assertions are hand-verifiable and
never copied back from the implementation.
"""

from __future__ import annotations

import pytest

from ragfold.engines.base import RetrievedPassage
from ragfold.fusion import reciprocal_rank_fusion


def passage(document_id: str, rank: int, *, text: str | None = None) -> RetrievedPassage:
    """Build a passage. Its position in the input list is what RRF ranks on."""

    return RetrievedPassage(
        document_id=document_id,
        text=text if text is not None else f"text-of-{document_id}",
        score=1.0 / rank,
        rank=rank,
    )


def test_two_engine_fusion_matches_hand_computed_scores():
    bm25 = [passage("d1", 1), passage("d2", 2), passage("d3", 3)]
    dense = [passage("d2", 1), passage("d3", 2), passage("d4", 3)]

    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, k=60)

    # d2: bm25 rank 2 + dense rank 1 = 1/62 + 1/61
    # d3: bm25 rank 3 + dense rank 2 = 1/63 + 1/62
    # d1: bm25 rank 1            = 1/61
    # d4: dense rank 3           = 1/63
    by_id = {p.document_id: p for p in fused}
    assert by_id["d2"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["d3"].score == pytest.approx(1 / 63 + 1 / 62)
    assert by_id["d1"].score == pytest.approx(1 / 61)
    assert by_id["d4"].score == pytest.approx(1 / 63)

    # Highest fused score first, ranks recomputed 1..n.
    assert [p.document_id for p in fused] == ["d2", "d3", "d1", "d4"]
    assert [p.rank for p in fused] == [1, 2, 3, 4]


def test_consensus_counts_engines_that_ranked_each_passage():
    bm25 = [passage("d1", 1), passage("d2", 2)]
    dense = [passage("d2", 1), passage("d4", 2)]

    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
    by_id = {p.document_id: p for p in fused}

    assert by_id["d2"].metadata["fusion"]["consensus"] == 2
    assert by_id["d1"].metadata["fusion"]["consensus"] == 1
    assert by_id["d4"].metadata["fusion"]["consensus"] == 1


def test_contributions_record_engine_name_rank_and_term():
    bm25 = [passage("d1", 1), passage("d2", 2)]
    dense = [passage("d2", 1)]

    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, k=60)
    d2 = next(p for p in fused if p.document_id == "d2")

    contributions = d2.metadata["fusion"]["contributions"]
    # Sorted by rank then engine name: dense (rank 1) before bm25 (rank 2).
    assert contributions == [
        {"engine": "dense", "rank": 1, "weight": 1.0, "score": pytest.approx(1 / 61)},
        {"engine": "bm25", "rank": 2, "weight": 1.0, "score": pytest.approx(1 / 62)},
    ]
    assert d2.metadata["fusion"]["method"] == "rrf"
    assert d2.metadata["fusion"]["k"] == 60


def test_single_ranking_reduces_to_reciprocal_rank():
    ranking = [passage("x", 1), passage("y", 2), passage("z", 3)]

    fused = reciprocal_rank_fusion([ranking], k=60)

    assert [p.document_id for p in fused] == ["x", "y", "z"]
    assert [p.score for p in fused] == [
        pytest.approx(1 / 61),
        pytest.approx(1 / 62),
        pytest.approx(1 / 63),
    ]
    # Positional (list) input auto-labels the engine.
    assert fused[0].metadata["fusion"]["contributions"][0]["engine"] == "engine-0"
    assert all(p.metadata["fusion"]["consensus"] == 1 for p in fused)


def test_k_parameter_changes_scores():
    engine_a = [passage("p", 1), passage("q", 2)]
    engine_b = [passage("q", 1)]

    fused = reciprocal_rank_fusion([engine_a, engine_b], k=10)
    by_id = {p.document_id: p for p in fused}

    # q: engine_a rank 2 + engine_b rank 1 = 1/12 + 1/11 ; p: 1/11
    assert by_id["q"].score == pytest.approx(1 / 12 + 1 / 11)
    assert by_id["p"].score == pytest.approx(1 / 11)
    assert [p.document_id for p in fused] == ["q", "p"]


def test_tie_break_is_deterministic_by_document_id():
    # Symmetric ranks -> identical fused scores, equal consensus, equal min rank.
    # The final tie-break falls to document_id ascending.
    engine_a = [passage("beta", 1), passage("alpha", 2)]
    engine_b = [passage("alpha", 1), passage("beta", 2)]

    fused = reciprocal_rank_fusion([engine_a, engine_b])

    assert fused[0].score == pytest.approx(fused[1].score)
    assert [p.document_id for p in fused] == ["alpha", "beta"]


def test_representative_text_comes_from_best_rank_occurrence():
    # RRF ranks on list position: bm25 puts d1 at position 3, dense at position 1.
    bm25 = [passage("x", 1), passage("y", 2), passage("d1", 3, text="worse")]
    dense = [passage("d1", 1, text="better")]

    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
    d1 = next(p for p in fused if p.document_id == "d1")

    assert d1.text == "better"  # dense ranked it best (position 1)


def test_empty_inputs_return_empty_list():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion({"bm25": [], "dense": []}) == []


def test_weights_mapping_scales_each_engine_contribution():
    bm25 = [passage("d1", 1), passage("d2", 2)]
    dense = [passage("d2", 1)]

    fused = reciprocal_rank_fusion(
        {"bm25": bm25, "dense": dense}, k=60, weights={"bm25": 2.0, "dense": 1.0}
    )
    by_id = {p.document_id: p for p in fused}

    # d1: bm25 rank 1 * 2.0                 = 2/61
    # d2: bm25 rank 2 * 2.0 + dense rank 1  = 2/62 + 1/61
    assert by_id["d1"].score == pytest.approx(2 / 61)
    assert by_id["d2"].score == pytest.approx(2 / 62 + 1 / 61)
    assert [p.document_id for p in fused] == ["d2", "d1"]
    # Contribution records the applied weight and the weighted term.
    d1_contrib = by_id["d1"].metadata["fusion"]["contributions"][0]
    assert d1_contrib["weight"] == 2.0
    assert d1_contrib["score"] == pytest.approx(2 / 61)


def test_weights_sequence_is_positional():
    engine_a = [passage("p", 1)]
    engine_b = [passage("q", 1)]

    fused = reciprocal_rank_fusion([engine_a, engine_b], weights=[3.0, 1.0])
    by_id = {p.document_id: p for p in fused}

    assert by_id["p"].score == pytest.approx(3 / 61)
    assert by_id["q"].score == pytest.approx(1 / 61)


def test_missing_engine_weight_defaults_to_one():
    bm25 = [passage("d1", 1)]
    dense = [passage("d2", 1)]

    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, weights={"bm25": 5.0})
    by_id = {p.document_id: p for p in fused}

    assert by_id["d1"].score == pytest.approx(5 / 61)
    assert by_id["d2"].score == pytest.approx(1 / 61)  # dense defaults to weight 1.0


def test_weights_sequence_length_mismatch_raises():
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[passage("d1", 1)], [passage("d2", 1)]], weights=[1.0])


def test_duplicate_document_within_one_engine_is_not_double_counted():
    # A malformed ranking that repeats a document id: only the best (first)
    # position for that engine contributes.
    bm25 = [passage("d1", 1), passage("d1", 2)]

    fused = reciprocal_rank_fusion({"bm25": bm25})

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[0].metadata["fusion"]["consensus"] == 1
