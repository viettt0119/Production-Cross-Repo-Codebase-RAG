"""Query-side value objects."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from code_rag.domain.entities import CodeChunk
from code_rag.domain.value_objects import FrozenModel, Language, SymbolId


class SearchQuery(FrozenModel):
    text: Annotated[str, Field(min_length=3, max_length=4000)]
    repos: tuple[str, ...] = ()
    languages: tuple[Language, ...] = ()
    include_generated: bool = False
    include_tests: bool = True
    top_k: Annotated[int, Field(ge=1, le=50)] = 10


class HitSource(StrEnum):
    DENSE_CODE = "dense_code"
    DENSE_SUMMARY = "dense_summary"
    BM25 = "bm25"
    GRAPH = "graph"


class RetrievalHit(FrozenModel):
    symbol_id: SymbolId
    source: HitSource
    rank: Annotated[int, Field(ge=1)]
    raw_score: float


class SearchFilter(FrozenModel):
    """Store-agnostic filter derived from a SearchQuery."""

    repos: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    include_generated: bool = False
    include_tests: bool = True
    generated_penalty: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6

    @classmethod
    def from_query(cls, q: SearchQuery, generated_penalty: float = 0.6) -> SearchFilter:
        return cls(
            repos=q.repos,
            languages=tuple(lang.value for lang in q.languages),
            include_generated=q.include_generated,
            include_tests=q.include_tests,
            generated_penalty=generated_penalty,
        )


class RankedContext(FrozenModel):
    chunk: CodeChunk
    fused_score: float
    rerank_score: float | None = None
    via_graph_from: SymbolId | None = None
    sources: tuple[HitSource, ...] = ()

    @property
    def symbol_id(self) -> SymbolId:
        return self.chunk.symbol.symbol_id
