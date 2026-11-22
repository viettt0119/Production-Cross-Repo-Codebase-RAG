from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from code_rag.domain.changeset import ChangeSet


class SourceRepository(Protocol):
    """Read-only access to repository content at arbitrary commits."""

    async def sync(self, repo: str) -> None:
        """Fetch latest refs from the remote (no-op for purely local repos)."""
        ...

    async def head(self, repo: str, ref: str | None = None) -> str: ...

    async def list_files(self, repo: str, commit_sha: str) -> list[str]: ...

    async def read_blobs(self, repo: str, commit_sha: str, paths: Sequence[str]) -> Mapping[str, bytes]:
        """Missing paths are omitted from the result instead of raising."""
        ...

    async def diff(self, repo: str, from_sha: str, to_sha: str) -> ChangeSet: ...

    async def is_ancestor(self, repo: str, ancestor: str, descendant: str) -> bool: ...

    async def commits_between(self, repo: str, from_sha: str, to_sha: str) -> list[str]:
        """First-parent commits in (from_sha, to_sha], oldest first."""
        ...
