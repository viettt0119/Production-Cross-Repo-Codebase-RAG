from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from code_rag.domain.changeset import LedgerEntry
from code_rag.domain.entities import CodeChunk, ParsedFile, RpcLink, SymbolEdge, SymbolNode, SymbolRef
from code_rag.domain.query import SearchFilter
from code_rag.domain.value_objects import SymbolId

VectorName = Literal["code", "summary"]


@dataclass(frozen=True, slots=True)
class ScoredId:
    symbol_id: SymbolId
    score: float


@dataclass(frozen=True, slots=True)
class FileRecord:
    repo: str
    path: str
    language: str
    module_name: str
    blob_sha: str
    line_count: int
    imports: tuple[str, ...]
    commit_sha: str


@dataclass(frozen=True, slots=True)
class GraphNeighbor:
    neighbor_id: SymbolId
    seed_id: SymbolId
    relation: str
    cross_repo: bool


class HashLedger(Protocol):
    """Source of truth for incremental diffing and cross-repo name resolution."""

    def file_entries(self, repo: str, path: str) -> list[LedgerEntry]: ...

    def get_file(self, repo: str, path: str) -> FileRecord | None: ...

    def replace_file(self, parsed: ParsedFile, line_count: int) -> None:
        """Atomically replace file record, its symbols and its unresolved refs."""
        ...

    def delete_file(self, repo: str, path: str) -> list[SymbolId]: ...

    def lookup_by_name(self, name: str) -> list[LedgerEntry]: ...

    def refs_from(self, src_ids: Sequence[SymbolId]) -> list[SymbolRef]: ...

    def refs_targeting(self, names: Sequence[str]) -> list[SymbolRef]:
        """Refs (any repo) whose last target segment is in `names`."""
        ...

    def all_src_ids_with_refs(self, repo: str | None = None) -> list[SymbolId]: ...

    def indexed_sha(self, repo: str) -> str | None: ...

    def set_indexed_sha(self, repo: str, sha: str) -> None: ...

    def symbol_entries(self, ids: Sequence[SymbolId]) -> dict[SymbolId, LedgerEntry]: ...

    def start_run(self, repo: str, from_sha: str | None, to_sha: str) -> str: ...

    def finish_run(self, run_id: str, status: str, stats_json: str, error: str | None = None) -> None: ...

    def repos(self) -> list[str]: ...

    def files_in_repo(self, repo: str) -> list[str]: ...

    def line_count(self, repo: str, path: str) -> int | None: ...

    def add_pending_batch(self, batch_id: str, repo: str) -> None: ...

    def pending_batches(self) -> list[tuple[str, str]]: ...

    def complete_batch(self, batch_id: str, status: str = "done") -> None: ...


class VectorIndex(Protocol):
    async def ensure_ready(self) -> None: ...

    async def upsert(
        self,
        chunks: Sequence[CodeChunk],
        code_vectors: Sequence[Sequence[float]],
        summary_vectors: Sequence[Sequence[float] | None],
    ) -> None: ...

    async def set_summaries(self, items: Sequence[tuple[SymbolId, str, str, Sequence[float]]]) -> None:
        """(symbol_id, summary, summary_model, summary_vector) — used by async enrichment."""
        ...

    async def update_locations(self, symbols: Sequence[SymbolNode]) -> None: ...

    async def delete(self, ids: Sequence[SymbolId]) -> None: ...

    async def search(
        self, vector: Sequence[float], using: VectorName, flt: SearchFilter, limit: int
    ) -> list[ScoredId]: ...

    async def fetch(self, ids: Sequence[SymbolId]) -> dict[SymbolId, CodeChunk]: ...

    async def count(self) -> int: ...


class LexicalIndex(Protocol):
    def upsert(self, chunks: Sequence[CodeChunk]) -> None: ...

    def delete(self, ids: Sequence[SymbolId]) -> None: ...

    def commit(self) -> None: ...

    def search(self, text: str, flt: SearchFilter, limit: int) -> list[ScoredId]: ...


class SymbolGraph(Protocol):
    def upsert_symbols(self, symbols: Sequence[SymbolNode]) -> None: ...

    def delete_symbols(self, ids: Sequence[SymbolId]) -> None: ...

    def replace_outgoing_edges(self, src_ids: Sequence[SymbolId], edges: Sequence[SymbolEdge]) -> None: ...

    def replace_rpc_links(self, symbol_ids: Sequence[SymbolId], links: Sequence[RpcLink]) -> None: ...

    def neighbors(
        self, seed_ids: Sequence[SymbolId], limit: int, min_confidence: float = 0.5
    ) -> list[GraphNeighbor]: ...

    def cross_repo_pairs(self, limit: int) -> list[tuple[SymbolId, SymbolId, str]]:
        """Sample (src, dst, relation) pairs that cross repo boundaries (for eval generation)."""
        ...

    def publish(self) -> None:
        """Make committed writes visible to read-only consumers (snapshot swap)."""
        ...


class FileLineIndex(Protocol):
    """Lookup used by the citation verifier."""

    def line_count(self, repo: str, path: str) -> int | None: ...


SymbolLookup = Mapping[SymbolId, CodeChunk]
