"""Reciprocal Rank Fusion (RRF) for combining engine rankings.

RRF is the canonical way to merge several ranked lists into one without
calibrating their score scales: it fuses *ranks*, not raw scores, so a BM25
score and a cosine similarity never need to be made comparable. The fused score
of a document is the sum over engines of ``1 / (k + rank)``, where ``rank`` is
the document's 1-based position in that engine's list.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009). Pure stdlib
arithmetic - no engines, no I/O, no dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ragfold.engines.base import RetrievedPassage

Rankings = Mapping[str, Sequence[RetrievedPassage]] | Sequence[Sequence[RetrievedPassage]]


def _labelled_rankings(rankings: Rankings) -> list[tuple[str, Sequence[RetrievedPassage]]]:
    """Normalize input to ``[(engine_name, ranking), ...]`` preserving order."""

    if isinstance(rankings, Mapping):
        return [(str(name), ranking) for name, ranking in rankings.items()]
    return [(f"engine-{idx}", ranking) for idx, ranking in enumerate(rankings)]


def reciprocal_rank_fusion(
    rankings: Rankings,
    *,
    k: int = 60,
) -> list[RetrievedPassage]:
    """Fuse several ranked passage lists into one with Reciprocal Rank Fusion.

    Args:
        rankings: Either a mapping of ``engine name -> ranked passages`` or a
            plain sequence of ranked passage lists (auto-labelled ``engine-0``,
            ``engine-1``, ...). Each list must already be ordered best-first;
            the passage's position in the list (1-based) is the rank RRF uses.
        k: The RRF damping constant. Larger ``k`` flattens the contribution of
            top ranks. The canonical default is 60.

    Returns:
        A new list of :class:`RetrievedPassage`, one per unique
        ``document_id``, ordered by fused score (highest first) with ``rank``
        recomputed 1..n. Each fused passage carries an explainable breakdown
        under ``metadata["fusion"]`` (``method``, ``k``, fused ``score``,
        ``consensus`` count, and per-engine ``contributions``). The representative
        ``text`` and base metadata are taken from the occurrence with the best
        (lowest) rank, ties broken by engine order. Truncation to a ``top_k`` is
        left to the caller.

    Tie-break: results are ordered by ``(-score, -consensus, min_rank,
    document_id)``, which is fully deterministic.
    """

    labelled = _labelled_rankings(rankings)

    scores: dict[str, float] = {}
    consensus: dict[str, int] = {}
    min_rank: dict[str, int] = {}
    # Internal contribution tuples: (rank, engine_name, term) - typed so the
    # later sort key is comparable.
    contributions: dict[str, list[tuple[int, str, float]]] = {}
    representative: dict[str, tuple[int, int, RetrievedPassage]] = {}

    for engine_index, (engine_name, ranking) in enumerate(labelled):
        seen: set[str] = set()
        for position, passage in enumerate(ranking, start=1):
            doc_id = passage.document_id
            # A single engine contributes a document at most once (its best rank).
            if doc_id in seen:
                continue
            seen.add(doc_id)

            term = 1.0 / (k + position)
            scores[doc_id] = scores.get(doc_id, 0.0) + term
            consensus[doc_id] = consensus.get(doc_id, 0) + 1
            min_rank[doc_id] = min(min_rank.get(doc_id, position), position)
            contributions.setdefault(doc_id, []).append((position, engine_name, term))

            # Representative view: best (lowest) rank wins; ties -> earlier engine.
            candidate = (position, engine_index, passage)
            current = representative.get(doc_id)
            if current is None or candidate[:2] < current[:2]:
                representative[doc_id] = candidate

    ordered_ids = sorted(
        scores,
        key=lambda doc_id: (
            -scores[doc_id],
            -consensus[doc_id],
            min_rank[doc_id],
            doc_id,
        ),
    )

    fused: list[RetrievedPassage] = []
    for new_rank, doc_id in enumerate(ordered_ids, start=1):
        source = representative[doc_id][2]
        engine_contributions = [
            {"engine": engine_name, "rank": rank, "score": term}
            for rank, engine_name, term in sorted(contributions[doc_id])
        ]
        metadata = dict(source.metadata)
        metadata["fusion"] = {
            "method": "rrf",
            "k": k,
            "score": scores[doc_id],
            "consensus": consensus[doc_id],
            "contributions": engine_contributions,
        }
        fused.append(
            RetrievedPassage(
                document_id=doc_id,
                text=source.text,
                score=scores[doc_id],
                rank=new_rank,
                metadata=metadata,
            )
        )
    return fused
