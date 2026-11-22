from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from code_rag.infrastructure.container import Container
from code_rag.infrastructure.parsing.treesitter_parser import TreeSitterParser
from code_rag.infrastructure.settings import Settings

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env={**os.environ, **_GIT_ENV}
    )
    return out.stdout.strip()


@dataclass
class CorpusRepos:
    root: Path
    paths: dict[str, Path]
    repos_file: Path

    def commit(self, repo: str, message: str = "change") -> str:
        path = self.paths[repo]
        git(path, "add", "-A")
        git(path, "commit", "-qm", message)
        return git(path, "rev-parse", "HEAD")

    def write(self, repo: str, rel: str, content: str) -> None:
        target = self.paths[repo] / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def read(self, repo: str, rel: str) -> str:
        return (self.paths[repo] / rel).read_text()


@pytest.fixture
def corpus(tmp_path: Path) -> CorpusRepos:
    paths: dict[str, Path] = {}
    for d in sorted(CORPUS.iterdir()):
        name = d.name.replace("__", "/")
        dst = tmp_path / "src" / d.name
        shutil.copytree(d, dst)
        git(dst, "init", "-q", "-b", "main")
        git(dst, "add", "-A")
        git(dst, "commit", "-qm", "initial")
        paths[name] = dst
    repos_file = tmp_path / "repos.yaml"
    repos_file.write_text(yaml.safe_dump({"repos": [{"name": n, "path": str(p)} for n, p in paths.items()]}))
    return CorpusRepos(root=tmp_path, paths=paths, repos_file=repos_file)


@pytest.fixture
async def container(corpus: CorpusRepos, tmp_path: Path) -> AsyncIterator[Container]:
    settings = Settings(profile="offline", data_dir=tmp_path / "data", repos_file=corpus.repos_file)
    c = Container(settings)
    yield c
    await c.aclose()


@pytest.fixture
async def indexed(container: Container, corpus: CorpusRepos) -> Container:
    for repo in corpus.paths:
        await container.ingest_workflow.run(repo)
    return container


@pytest.fixture(scope="session")
def parser() -> Iterator[TreeSitterParser]:
    return TreeSitterParser()
