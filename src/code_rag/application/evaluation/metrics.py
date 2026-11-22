"""Ranking and latency metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def reciprocal_rank(relevance: Sequence[int], k: int = 10) -> float:
    """`relevance[i]` > 0 if the i-th retrieved item is relevant."""
    for i, rel in enumerate(relevance[:k], start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(retrieved_gains: Sequence[float], ideal_gains: Sequence[float], k: int = 10) -> float:
    """Graded nDCG@k. `ideal_gains` are the grades of *all* gold items (not only retrieved ones)."""
    ideal = dcg(sorted(ideal_gains, reverse=True), k)
    return dcg(retrieved_gains, k) / ideal if ideal > 0 else 0.0


def recall_at_k(found_gold: int, total_gold: int) -> float:
    return found_gold / total_gold if total_gold else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 50), 1),
        "p95": round(percentile(values, 95), 1),
        "p99": round(percentile(values, 99), 1),
        "max": round(max(values), 1) if values else 0.0,
        "mean": round(sum(values) / len(values), 1) if values else 0.0,
    }


def mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
