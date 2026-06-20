"""Shared text utilities for lightweight lexical engines."""

from __future__ import annotations

import math
import re
from collections import Counter

from ragfold.engines.base import DocumentChunk, RetrievedPassage

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def rank_passages(chunks: list[DocumentChunk], scores: list[float], top_k: int) -> list[RetrievedPassage]:
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda item: item[1], reverse=True)
    passages: list[RetrievedPassage] = []
    for rank, (chunk, score) in enumerate(ranked[: max(top_k, 0)], start=1):
        passages.append(
            RetrievedPassage(
                document_id=chunk.id,
                text=chunk.text,
                score=float(score),
                rank=rank,
                metadata=chunk.metadata,
            )
        )
    return passages


def tfidf_scores(chunks: list[DocumentChunk], query: str) -> list[float]:
    if not chunks:
        return []
    tokenized_docs = [tokenize(chunk.text) for chunk in chunks]
    query_terms = tokenize(query)
    if not query_terms:
        return [0.0 for _ in chunks]

    doc_freq: Counter[str] = Counter()
    for terms in tokenized_docs:
        doc_freq.update(set(terms))

    query_counts = Counter(query_terms)
    scores: list[float] = []
    total_docs = len(chunks)
    for terms in tokenized_docs:
        term_counts = Counter(terms)
        doc_len = len(terms) or 1
        score = 0.0
        for term, query_weight in query_counts.items():
            if term_counts[term] == 0:
                continue
            tf = term_counts[term] / doc_len
            idf = math.log((total_docs + 1) / (doc_freq[term] + 1)) + 1.0
            score += tf * idf * query_weight
        scores.append(score)
    return scores
