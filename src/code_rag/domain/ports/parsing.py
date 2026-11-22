from __future__ import annotations

from typing import Protocol

from code_rag.domain.entities import ParsedFile


class CodeParser(Protocol):
    """Turns a file's bytes into AST-level symbols, unresolved references and RPC links."""

    def supports(self, path: str) -> bool: ...

    def parse(self, repo: str, path: str, commit_sha: str, content: bytes) -> ParsedFile | None:
        """Return None for unsupported/empty/binary files. Raise ParseError on hard failures."""
        ...


class BulkCodeParser(Protocol):
    """Parses many files at once (possibly across processes)."""

    def supports(self, path: str) -> bool: ...

    async def parse_many(
        self, repo: str, commit_sha: str, blobs: dict[str, bytes]
    ) -> tuple[list[ParsedFile], dict[str, str]]:
        """Return parsed files and a `path -> error` map for files that failed."""
        ...
