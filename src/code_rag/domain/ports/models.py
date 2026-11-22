from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from code_rag.domain.entities import SymbolNode
from code_rag.domain.query import RankedContext
from code_rag.domain.value_objects import SymbolId

InputType = Literal["document", "query"]


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class FileToSummarize:
    repo: str
    path: str
    language: str
    file_source: str
    symbols: tuple[SymbolNode, ...]


class SymbolSummarizer(Protocol):
    @property
    def model_name(self) -> str: ...

    async def summarize_file(self, item: FileToSummarize) -> dict[SymbolId, str]:
        """Return one-sentence summaries. Missing ids mean 'no summary' (degraded, not fatal)."""
        ...


class BatchSummarizer(Protocol):
    """Asynchronous, cheaper summarization for bulk indexing (e.g. Message Batches API)."""

    @property
    def model_name(self) -> str: ...

    async def submit(self, items: Sequence[FileToSummarize]) -> str: ...

    async def poll(self, batch_id: str) -> Literal["in_progress", "ended"]: ...

    async def collect(self, batch_id: str) -> dict[SymbolId, str]: ...


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    async def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[RerankResult]: ...


@dataclass(frozen=True, slots=True)
class QueryExpansion:
    rewrites: tuple[str, ...]
    identifiers: tuple[str, ...]


class QueryExpander(Protocol):
    async def expand(self, question: str) -> QueryExpansion: ...


class AnswerSynthesizer(Protocol):
    @property
    def model_name(self) -> str: ...

    def stream(self, question: str, contexts: Sequence[RankedContext], feedback: str | None) -> AsyncIterator[str]: ...


class EntailmentJudge(Protocol):
    """Optional LLM judge for evaluation: does the cited code support the claim?"""

    async def supports(self, claim: str, evidence: Mapping[str, str]) -> bool: ...


@dataclass(frozen=True, slots=True)
class QuestionSeed:
    """A pair of related symbols (often across repos) to turn into an evaluation question."""

    relation: str
    src_repo: str
    src_name: str
    src_summary: str
    src_source: str
    dst_repo: str
    dst_name: str
    dst_summary: str
    dst_source: str


class QuestionWriter(Protocol):
    async def write(self, seed: QuestionSeed) -> str | None:
        """A natural developer question whose answer needs both symbols; None to skip the seed."""
        ...
