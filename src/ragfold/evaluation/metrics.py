"""Retrieval and answer metrics.

All functions follow the convention: predicted value first, reference value
second. Higher is better for every score in this module.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence


def recall_at_k(predicted: Sequence[str], reference: Sequence[str], k: int) -> float:
    ref = set(reference)
    if not ref:
        return 1.0
    return len(set(predicted[:k]) & ref) / len(ref)


def precision_at_k(predicted: Sequence[str], reference: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(predicted[:k]) & set(reference)) / k


def hit_at_k(predicted: Sequence[str], reference: Sequence[str], k: int) -> float:
    return 1.0 if set(predicted[:k]) & set(reference) else 0.0


def mean_reciprocal_rank(predicted: Sequence[str], reference: Sequence[str]) -> float:
    ref = set(reference)
    for idx, doc_id in enumerate(predicted, start=1):
        if doc_id in ref:
            return 1 / idx
    return 0.0


def ndcg_at_k(predicted: Sequence[str], reference: Sequence[str], k: int) -> float:
    ref = set(reference)
    if not ref:
        return 1.0
    dcg = 0.0
    for idx, doc_id in enumerate(predicted[:k], start=1):
        if doc_id in ref:
            dcg += 1 / math.log2(idx + 1)
    ideal_hits = min(len(ref), k)
    ideal = sum(1 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def average_precision(
    predicted: Sequence[str],
    reference: Sequence[str],
    k: int | None = None,
) -> float:
    ref = set(reference)
    if not ref:
        return 1.0
    limit = len(predicted) if k is None else k
    hits = 0
    total = 0.0
    for idx, doc_id in enumerate(predicted[:limit], start=1):
        if doc_id in ref:
            hits += 1
            total += hits / idx
    return total / len(ref)


def map_score(
    predicted: Sequence[Sequence[str]],
    reference: Sequence[Sequence[str]],
    k: int | None = None,
) -> float:
    if not predicted:
        return 0.0
    scores = [
        average_precision(pred, ref, k=k)
        for pred, ref in zip(predicted, reference, strict=True)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def answer_exact_match(predicted: str, reference: str) -> float:
    return 1.0 if _normalize_answer(predicted) == _normalize_answer(reference) else 0.0


def answer_f1(predicted: str, reference: str) -> float:
    pred_tokens = _normalize_answer(predicted).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_cost(results: Sequence[float]) -> float:
    return sum(results)


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punct)
    return " ".join(no_articles.split())
