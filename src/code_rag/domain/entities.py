"""Core entities of the code-intelligence domain."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from code_rag.domain.value_objects import (
    ContentHash,
    EdgeKind,
    FrozenModel,
    Language,
    LineSpan,
    RepoCoordinate,
    SymbolId,
    SymbolKind,
)


class SymbolNode(FrozenModel):
    """Pure parser output: one AST-level unit (function/class/struct/...)."""

    symbol_id: SymbolId
    coord: RepoCoordinate
    language: Language
    kind: SymbolKind
    name: str
    qualified_name: str
    parent_id: SymbolId | None = None
    span: LineSpan
    signature: str
    source: str
    content_hash: ContentHash
    is_generated: bool = False
    is_test: bool = False

    @property
    def repo(self) -> str:
        return self.coord.repo

    @property
    def path(self) -> str:
        return self.coord.path

    def citation_anchor(self) -> str:
        return f"({self.coord.repo}/{self.coord.path}:{self.span.start}-{self.span.end})"


class CodeChunk(FrozenModel):
    """Indexed unit: a symbol plus its generated representations' metadata."""

    symbol: SymbolNode
    summary: str | None = None
    summary_model: str | None = None
    embedding_model: str


class SymbolRef(FrozenModel):
    """Unresolved reference captured at parse time (target may live in another repo)."""

    src_id: SymbolId
    src_repo: str
    src_path: str
    kind: EdgeKind
    target_name: str
    import_context: tuple[str, ...] = ()
    at_line: Annotated[int, Field(ge=1)]
    rpc_service: str | None = None
    rpc_method: str | None = None


class SymbolEdge(FrozenModel):
    src_id: SymbolId
    dst_id: SymbolId
    kind: EdgeKind
    cross_repo: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    at_line: Annotated[int, Field(ge=1)]


class RpcEndpoint(FrozenModel):
    service: str
    method: str

    @property
    def endpoint_id(self) -> str:
        return f"{self.service}/{self.method}"


class RpcLink(FrozenModel):
    """Symbol -> RpcEndpoint relation (client call or server handler)."""

    symbol_id: SymbolId
    endpoint: RpcEndpoint
    kind: EdgeKind  # CALLS_RPC | SERVES_RPC | DECLARES_RPC
    at_line: Annotated[int, Field(ge=1)]


class ParsedFile(FrozenModel):
    """Everything extracted from a single file at a single commit."""

    repo: str
    path: str
    commit_sha: str
    language: Language
    blob_sha: str
    module_name: str
    symbols: tuple[SymbolNode, ...]
    refs: tuple[SymbolRef, ...]
    imports: tuple[str, ...]
    rpc_links: tuple[RpcLink, ...] = ()
