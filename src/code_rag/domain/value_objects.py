"""Immutable value objects. Pure Python + Pydantic, no infrastructure imports."""

from __future__ import annotations

import hashlib
import uuid
from enum import StrEnum
from typing import Annotated, NewType

from pydantic import BaseModel, ConfigDict, Field, model_validator

SymbolId = NewType("SymbolId", str)
"""Stable identity of a symbol: survives line moves, changes on rename/move to another file."""

ContentHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")]
RepoName = Annotated[str, Field(pattern=r"^[\w.-]+/[\w.-]+$")]

# Fixed namespace so point ids are reproducible across processes and machines.
_POINT_NAMESPACE = uuid.UUID("6f1d3a9e-6c0b-4a57-9d8e-2b6f3c1e8a40")


class Language(StrEnum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    GO = "go"
    RUST = "rust"
    PROTO = "proto"


class SymbolKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    STRUCT = "struct"
    INTERFACE = "interface"
    TRAIT = "trait"
    ENUM = "enum"
    TYPE_ALIAS = "type_alias"
    IMPL = "impl"
    MODULE = "module"


class EdgeKind(StrEnum):
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    CALLS_RPC = "CALLS_RPC"
    SERVES_RPC = "SERVES_RPC"
    DECLARES_RPC = "DECLARES_RPC"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LineSpan(FrozenModel):
    """1-based inclusive line range."""

    start: Annotated[int, Field(ge=1)]
    end: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _ordered(self) -> LineSpan:
        if self.end < self.start:
            raise ValueError(f"LineSpan end ({self.end}) < start ({self.start})")
        return self

    def contains(self, other: LineSpan) -> bool:
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: LineSpan) -> bool:
        return self.start <= other.end and other.start <= self.end

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


class RepoCoordinate(FrozenModel):
    repo: RepoName
    path: str = Field(min_length=1)
    commit_sha: CommitSha

    @model_validator(mode="after")
    def _posix_relative(self) -> RepoCoordinate:
        if self.path.startswith("/") or "\\" in self.path or ".." in self.path.split("/"):
            raise ValueError(f"path must be repo-relative posix: {self.path!r}")
        return self


def make_symbol_id(repo: str, path: str, qualified_name: str, kind: SymbolKind, ordinal: int = 0) -> SymbolId:
    raw = f"{repo}|{path}|{qualified_name}|{kind.value}|{ordinal}"
    return SymbolId(hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest())


def symbol_point_id(symbol_id: SymbolId) -> str:
    """Deterministic UUID used as the vector-store point id (idempotent upserts)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, symbol_id))


def sha256_hex(data: str | bytes) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()
