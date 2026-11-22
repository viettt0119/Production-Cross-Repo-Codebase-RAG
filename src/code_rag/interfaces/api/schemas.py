from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from code_rag.domain.query import RankedContext, SearchQuery
from code_rag.domain.value_objects import Language


class QueryRequest(BaseModel):
    question: Annotated[str, Field(min_length=3, max_length=4000)]
    repos: list[str] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    include_generated: bool = False
    include_tests: bool = True
    top_k: Annotated[int, Field(ge=1, le=50)] = 10
    stream: bool = False

    def to_query(self) -> SearchQuery:
        return SearchQuery(
            text=self.question,
            repos=tuple(self.repos),
            languages=tuple(self.languages),
            include_generated=self.include_generated,
            include_tests=self.include_tests,
            top_k=self.top_k,
        )


class ContextOut(BaseModel):
    symbol_id: str
    repo: str
    path: str
    start_line: int
    end_line: int
    kind: str
    qualified_name: str
    signature: str
    summary: str | None
    fused_score: float
    rerank_score: float | None
    via_graph: bool
    sources: list[str]
    anchor: str

    @classmethod
    def from_context(cls, c: RankedContext) -> ContextOut:
        s = c.chunk.symbol
        return cls(
            symbol_id=s.symbol_id,
            repo=s.coord.repo,
            path=s.coord.path,
            start_line=s.span.start,
            end_line=s.span.end,
            kind=s.kind.value,
            qualified_name=s.qualified_name,
            signature=s.signature,
            summary=c.chunk.summary,
            fused_score=round(c.fused_score, 6),
            rerank_score=None if c.rerank_score is None else round(c.rerank_score, 6),
            via_graph=c.via_graph_from is not None,
            sources=[x.value for x in c.sources],
            anchor=s.citation_anchor(),
        )


class SearchResponse(BaseModel):
    results: list[ContextOut]


class WebhookAccepted(BaseModel):
    queued: bool
    job_id: str | None = None
    reason: str | None = None
