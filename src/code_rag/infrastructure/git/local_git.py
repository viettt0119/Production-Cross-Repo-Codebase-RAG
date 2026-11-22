"""SourceRepository adapter over local git mirrors (bare clones or working copies)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from code_rag.domain.changeset import ChangeSet, ChangeType, FileChange
from code_rag.domain.errors import SourceUnavailableError

log = logging.getLogger(__name__)

_STATUS = {"A": ChangeType.ADDED, "M": ChangeType.MODIFIED, "D": ChangeType.DELETED, "T": ChangeType.MODIFIED}


class LocalGitSource:
    def __init__(
        self,
        repos_root: str | Path,
        remotes: Mapping[str, str] | None = None,
        local_paths: Mapping[str, str] | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self._root = Path(repos_root)
        self._remotes = dict(remotes or {})
        self._local = {k: Path(v) for k, v in (local_paths or {}).items()}
        self._timeout = timeout_s
        self._locks: dict[str, asyncio.Lock] = {}

    def repo_dir(self, repo: str) -> Path:
        if repo in self._local:
            return self._local[repo]
        return self._root / repo

    async def _git(
        self, repo: str | None, *args: str, stdin: bytes | None = None, check: bool = True
    ) -> tuple[int, bytes]:
        cmd = ["git"]
        if repo is not None:
            cmd += ["-C", str(self.repo_dir(repo))]
        cmd += list(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=self._timeout)
        except TimeoutError as exc:
            proc.kill()
            raise SourceUnavailableError(f"git {' '.join(args[:2])} timed out") from exc
        rc = proc.returncode if proc.returncode is not None else -1
        if check and rc != 0:
            raise SourceUnavailableError(f"git {' '.join(args)} failed ({rc}): {err.decode(errors='replace')[:400]}")
        return rc, out

    async def sync(self, repo: str) -> None:
        lock = self._locks.setdefault(repo, asyncio.Lock())
        async with lock:
            path = self.repo_dir(repo)
            remote = self._remotes.get(repo)
            if repo in self._local:
                if not path.exists():
                    raise SourceUnavailableError(f"local repo path missing: {path}")
                return
            if not path.exists():
                if not remote:
                    raise SourceUnavailableError(f"no remote configured for {repo}")
                path.parent.mkdir(parents=True, exist_ok=True)
                await self._git(None, "clone", "--bare", "--quiet", remote, str(path))
                return
            if remote:
                await self._git(repo, "fetch", "--quiet", "--prune", "origin", "+refs/heads/*:refs/heads/*")

    async def head(self, repo: str, ref: str | None = None) -> str:
        _, out = await self._git(repo, "rev-parse", "--verify", f"{ref or 'HEAD'}^{{commit}}")
        return out.decode().strip()

    async def list_files(self, repo: str, commit_sha: str) -> list[str]:
        _, out = await self._git(repo, "ls-tree", "-r", "-z", "--name-only", commit_sha)
        return [p for p in out.decode("utf-8", errors="replace").split("\0") if p]

    async def read_blobs(self, repo: str, commit_sha: str, paths: Sequence[str]) -> dict[str, bytes]:
        if not paths:
            return {}
        request = "".join(f"{commit_sha}:{p}\n" for p in paths).encode("utf-8")
        _, out = await self._git(repo, "cat-file", "--batch", stdin=request)
        result: dict[str, bytes] = {}
        pos = 0
        for path in paths:
            nl = out.index(b"\n", pos)
            header = out[pos:nl].decode("utf-8", errors="replace")
            pos = nl + 1
            parts = header.split(" ")
            if len(parts) == 2 and parts[1] == "missing":
                continue
            if len(parts) != 3:
                raise SourceUnavailableError(f"unexpected cat-file header: {header!r}")
            size = int(parts[2])
            if parts[1] == "blob":
                result[path] = out[pos : pos + size]
            pos += size + 1  # content + trailing LF
        return result

    async def diff(self, repo: str, from_sha: str, to_sha: str) -> ChangeSet:
        _, out = await self._git(repo, "diff", "--name-status", "-z", "-M50%", from_sha, to_sha)
        tokens = out.decode("utf-8", errors="replace").split("\0")
        files: list[FileChange] = []
        i = 0
        while i < len(tokens) and tokens[i]:
            status = tokens[i]
            code = status[0]
            if code in ("R", "C"):
                old, new = tokens[i + 1], tokens[i + 2]
                i += 3
                if code == "R":
                    files.append(FileChange(repo=repo, path=new, old_path=old, change=ChangeType.RENAMED))
                else:
                    files.append(FileChange(repo=repo, path=new, change=ChangeType.ADDED))
                continue
            path = tokens[i + 1]
            i += 2
            change = _STATUS.get(code)
            if change is not None:
                files.append(FileChange(repo=repo, path=path, change=change))
        return ChangeSet(repo=repo, from_sha=from_sha, to_sha=to_sha, files=tuple(files))

    async def is_ancestor(self, repo: str, ancestor: str, descendant: str) -> bool:
        rc, _ = await self._git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
        return rc == 0

    async def commits_between(self, repo: str, from_sha: str, to_sha: str) -> list[str]:
        _, out = await self._git(repo, "rev-list", "--reverse", "--first-parent", f"{from_sha}..{to_sha}")
        return [line for line in out.decode().split("\n") if line]
