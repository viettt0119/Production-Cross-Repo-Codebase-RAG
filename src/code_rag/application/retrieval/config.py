from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    dense_k: int = 40
    bm25_k: int = 40
    fused_k: int = 50
    rerank_top_k: int = 10
    graph_seeds: int = 8
    graph_limit: int = 16
    graph_min_confidence: float = 0.5
    rrf_k: int = 60
    generated_penalty: float = 0.6
    query_expansion: bool = True
    max_synthesis_attempts: int = 3
    source_weights: dict[str, float] = Field(
        default_factory=lambda: {"dense_code": 1.0, "dense_summary": 0.8, "bm25": 1.0}
    )
