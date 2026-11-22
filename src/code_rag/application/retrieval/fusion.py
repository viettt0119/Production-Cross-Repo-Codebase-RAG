"""Reciprocal Rank Fusion across heterogeneous ranked lists with per-list weights."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from code_rag.domain.ports.indexing import ScoredId
from code_rag.domain.query import HitSource
from code_rag.domain.value_objects import SymbolId


@dataclass(frozen=True, slots=True)
class RankedList:
    source: HitSource
    hits: Sequence[ScoredId]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class FusedHit:
    symbol_id: SymbolId
    score: float
    sources: tuple[HitSource, ...]
    best_rank: int


def rrf_fuse(ranked_lists: Sequence[RankedList], k: int = 60, limit: int | None = None) -> list[FusedHit]:
    """score(d) = sum_i w_i / (k + rank_i(d)). Duplicates inside one list count once (best rank)."""
    scores: dict[SymbolId, float] = defaultdict(float)
    sources: dict[SymbolId, list[HitSource]] = defaultdict(list)
    best: dict[SymbolId, int] = {}
    for rl in ranked_lists:
        seen: set[SymbolId] = set()
        rank = 0
        for hit in rl.hits:
            if hit.symbol_id in seen:
                continue
            seen.add(hit.symbol_id)
            rank += 1
            scores[hit.symbol_id] += rl.weight / (k + rank)
            if rl.source not in sources[hit.symbol_id]:
                sources[hit.symbol_id].append(rl.source)
            best[hit.symbol_id] = min(best.get(hit.symbol_id, rank), rank)
    fused = [
        FusedHit(symbol_id=sid, score=s, sources=tuple(sources[sid]), best_rank=best[sid]) for sid, s in scores.items()
    ]
    fused.sort(key=lambda h: (h.score, -h.best_rank), reverse=True)
    return fused[:limit] if limit else fused
