"""Change detection value objects for incremental indexing."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from code_rag.domain.entities import SymbolNode
from code_rag.domain.value_objects import FrozenModel, SymbolId


class ChangeType(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class FileChange(FrozenModel):
    repo: str
    path: str
    old_path: str | None = None
    change: ChangeType


class ChangeSet(FrozenModel):
    repo: str
    from_sha: str | None
    to_sha: str
    files: tuple[FileChange, ...]

    @property
    def is_full(self) -> bool:
        return self.from_sha is None


class LedgerEntry(FrozenModel):
    symbol_id: SymbolId
    repo: str
    path: str
    name: str
    qualified_name: str
    kind: str
    content_hash: str
    start_line: int
    end_line: int
    commit_sha: str


class SymbolDelta(FrozenModel):
    """Result of diffing freshly parsed symbols against the hash ledger."""

    to_embed: tuple[SymbolNode, ...] = ()
    to_relocate: tuple[SymbolNode, ...] = ()
    to_delete: tuple[SymbolId, ...] = ()
    unchanged: tuple[SymbolNode, ...] = Field(default=())

    @property
    def touched(self) -> int:
        return len(self.to_embed) + len(self.to_relocate) + len(self.to_delete)


class IngestStats(FrozenModel):
    repo: str
    from_sha: str | None
    to_sha: str
    files_changed: int
    files_parsed: int
    symbols_embedded: int
    symbols_relocated: int
    symbols_deleted: int
    symbols_unchanged: int
    edges_resolved: int
    summaries_failed: int
    timings_ms: dict[str, float]
