"""Learn per-engine RRF weights from a labelled dataset.

This is a deterministic, dependency-free grid search: it evaluates every point
on a small candidate-weight grid against a labelled query set and keeps the
combination that maximises a retrieval metric. No gradient training, no model
downloads - it runs under `pytest -m "not slow"` on the core lexical engines.
The uniform (all-1.0) point is always in the grid, so the tuned weights can
never score worse than the unweighted baseline.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragfold.engines.base import CorpusInput
from ragfold.engines.router import EngineRouter
from ragfold.evaluation import metrics

QueryInput = Mapping[str, Any]

_METRICS = {
    "ndcg": lambda pred, ref, k: metrics.ndcg_at_k(pred, ref, k),
    "recall": lambda pred, ref, k: metrics.recall_at_k(pred, ref, k),
    "precision": lambda pred, ref, k: metrics.precision_at_k(pred, ref, k),
    "hit": lambda pred, ref, k: metrics.hit_at_k(pred, ref, k),
    "mrr": lambda pred, ref, k: metrics.mean_reciprocal_rank(pred, ref),
    "map": lambda pred, ref, k: metrics.average_precision(pred, ref, k=k),
}


@dataclass
class TuneResult:
    """Best weights found, its mean metric score, and every trial evaluated."""

    weights: dict[str, float]
    score: float
    metric: str
    trials: list[tuple[dict[str, float], float]] = field(default_factory=list)


def _query_text(query: QueryInput) -> str:
    return str(query.get("query") or query.get("text") or "")


def _relevant_ids(query: QueryInput) -> list[str]:
    relevant = query.get("relevant_ids") or query.get("relevant") or []
    return [str(doc_id) for doc_id in relevant]


async def tune_rrf_weights(
    router: EngineRouter,
    corpus: CorpusInput,
    queries: Sequence[QueryInput],
    *,
    engines: list[str],
    top_k: int = 5,
    k: int = 60,
    candidate_weights: Sequence[float] = (0.5, 1.0, 2.0),
    metric: str = "ndcg",
    metric_k: int | None = None,
) -> TuneResult:
    """Grid-search per-engine RRF weights to maximise a retrieval metric.

    Args:
        router: A router that already has `engines` registered and available.
        corpus: The document collection (same shape as `retrieve_hybrid`).
        queries: Labelled queries; each needs `query`/`text` and
            `relevant_ids`/`relevant`.
        engines: The engine names to fuse.
        top_k: Passages retrieved per query for scoring.
        k: RRF damping constant.
        candidate_weights: The per-engine weight grid. Must include 1.0 so the
            uniform baseline is always evaluated.
        metric: One of ``ndcg``, ``recall``, ``precision``, ``hit``, ``mrr``,
            ``map`` (predicted first, reference second; higher is better).
        metric_k: Cut-off for the metric; defaults to `top_k`.

    Returns:
        A :class:`TuneResult` with the best weights (deterministic tie-break:
        the first grid point reaching the max score), its mean score, and all
        trials.
    """

    if metric not in _METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {sorted(_METRICS)}.")
    if not engines:
        raise ValueError("tune_rrf_weights requires at least one engine name.")

    score_fn = _METRICS[metric]
    cut = top_k if metric_k is None else metric_k
    grid = itertools.product(candidate_weights, repeat=len(engines))

    trials: list[tuple[dict[str, float], float]] = []
    best_weights: dict[str, float] | None = None
    best_score = float("-inf")

    for combo in grid:
        weights = {name: float(value) for name, value in zip(engines, combo, strict=True)}
        per_query: list[float] = []
        for query in queries:
            result = await router.retrieve_hybrid(
                corpus,
                _query_text(query),
                engines=engines,
                top_k=top_k,
                k=k,
                weights=weights,
            )
            predicted = [passage.document_id for passage in result.passages]
            per_query.append(score_fn(predicted, _relevant_ids(query), cut))
        mean_score = sum(per_query) / len(per_query) if per_query else 0.0
        trials.append((weights, mean_score))
        if mean_score > best_score:
            best_score = mean_score
            best_weights = weights

    assert best_weights is not None  # grid is non-empty when engines is non-empty
    return TuneResult(weights=best_weights, score=best_score, metric=metric, trials=trials)
