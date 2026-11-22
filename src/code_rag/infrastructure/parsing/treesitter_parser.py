"""Multi-language AST parser (tree-sitter) implementing the CodeParser port."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import tree_sitter as ts
import tree_sitter_go
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript

from code_rag.domain.entities import ParsedFile
from code_rag.domain.errors import ParseError
from code_rag.domain.value_objects import Language
from code_rag.infrastructure.parsing.classify import (
    detect_language,
    is_excluded,
    is_generated,
    is_proto,
    is_test,
    looks_binary,
)
from code_rag.infrastructure.parsing.extractors.base import BaseExtractor, FileContext
from code_rag.infrastructure.parsing.extractors.golang import GoExtractor
from code_rag.infrastructure.parsing.extractors.python import PythonExtractor
from code_rag.infrastructure.parsing.extractors.rust import RustExtractor
from code_rag.infrastructure.parsing.extractors.typescript import TsxExtractor, TypeScriptExtractor
from code_rag.infrastructure.parsing.proto_parser import parse_proto

MAX_FILE_BYTES = 1_500_000

_GRAMMARS: dict[Language, Any] = {
    Language.PYTHON: tree_sitter_python.language,
    Language.TYPESCRIPT: tree_sitter_typescript.language_typescript,
    Language.TSX: tree_sitter_typescript.language_tsx,
    Language.GO: tree_sitter_go.language,
    Language.RUST: tree_sitter_rust.language,
}


def git_blob_sha(content: bytes) -> str:
    import hashlib

    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


class TreeSitterParser:
    def __init__(
        self,
        generated_globs: Sequence[str] = (),
        exclude_globs: Sequence[str] = (),
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        self._generated_globs = tuple(generated_globs)
        self._exclude_globs = tuple(exclude_globs)
        self._max_bytes = max_file_bytes
        self._parsers: dict[Language, ts.Parser] = {}
        self._extractors: dict[Language, BaseExtractor] = {
            Language.PYTHON: PythonExtractor(),
            Language.TYPESCRIPT: TypeScriptExtractor(),
            Language.TSX: TsxExtractor(),
            Language.GO: GoExtractor(),
            Language.RUST: RustExtractor(),
        }

    def _parser(self, lang: Language) -> ts.Parser:
        parser = self._parsers.get(lang)
        if parser is None:
            parser = ts.Parser(ts.Language(_GRAMMARS[lang]()))
            self._parsers[lang] = parser
        return parser

    def supports(self, path: str) -> bool:
        if is_excluded(path, self._exclude_globs):
            return False
        return detect_language(path) is not None or is_proto(path)

    def parse(self, repo: str, path: str, commit_sha: str, content: bytes) -> ParsedFile | None:
        if not self.supports(path) or not content or len(content) > self._max_bytes or looks_binary(content):
            return None
        blob = git_blob_sha(content)
        if is_proto(path):
            return parse_proto(repo, path, commit_sha, content, blob)
        lang = detect_language(path)
        if lang is None:
            return None
        text = content.decode("utf-8", errors="replace")
        try:
            tree = self._parser(lang).parse(content)
        except (ValueError, RuntimeError) as exc:  # tree-sitter raises these on internal failures
            raise ParseError(f"{repo}/{path}: {exc}") from exc
        ctx = FileContext(
            repo=repo,
            path=path,
            commit_sha=commit_sha,
            source=content,
            text=text,
            lines=text.split("\n"),
            is_generated=is_generated(path, text[:2000], self._generated_globs),
            is_test=is_test(path),
        )
        result = self._extractors[lang].extract(tree, ctx)
        return ParsedFile(
            repo=repo,
            path=path,
            commit_sha=commit_sha,
            language=lang,
            blob_sha=blob,
            module_name=result.module_name,
            symbols=tuple(result.symbols),
            refs=tuple(result.refs),
            imports=tuple(b.render() for b in result.imports),
            rpc_links=tuple(result.rpc_links),
        )


# ---------------------------------------------------------------------------
# Process-pool parsing: tree-sitter objects are not picklable, so each worker
# process builds its own parser once (initializer) and returns pydantic models.
# ---------------------------------------------------------------------------
_WORKER_PARSER: TreeSitterParser | None = None


def _init_worker(generated_globs: tuple[str, ...], exclude_globs: tuple[str, ...]) -> None:
    global _WORKER_PARSER  # noqa: PLW0603 - one parser per pool process, built by the initializer
    _WORKER_PARSER = TreeSitterParser(generated_globs, exclude_globs)


def _parse_in_worker(args: tuple[str, str, str, bytes]) -> tuple[str, ParsedFile | None, str | None]:
    repo, path, sha, content = args
    assert _WORKER_PARSER is not None
    try:
        return path, _WORKER_PARSER.parse(repo, path, sha, content), None
    except Exception as exc:  # one bad file must never fail a whole ingest; reported to caller
        return path, None, f"{type(exc).__name__}: {exc}"


class ParallelParser:
    """Async facade: small batches parse inline, large ones fan out to a process pool."""

    INLINE_THRESHOLD = 24

    def __init__(
        self,
        parser: TreeSitterParser,
        workers: int | None = None,
        generated_globs: Sequence[str] = (),
        exclude_globs: Sequence[str] = (),
    ) -> None:
        self._parser = parser
        self._workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self._init_args = (tuple(generated_globs), tuple(exclude_globs))
        self._pool: ProcessPoolExecutor | None = None

    @property
    def inner(self) -> TreeSitterParser:
        return self._parser

    def supports(self, path: str) -> bool:
        return self._parser.supports(path)

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers, initializer=_init_worker, initargs=self._init_args
            )
        return self._pool

    async def parse_many(
        self, repo: str, commit_sha: str, blobs: dict[str, bytes]
    ) -> tuple[list[ParsedFile], dict[str, str]]:
        items = [(repo, path, commit_sha, content) for path, content in blobs.items() if self._parser.supports(path)]
        errors: dict[str, str] = {}
        parsed: list[ParsedFile] = []
        if len(items) <= self.INLINE_THRESHOLD:
            for _, path, sha, content in items:
                try:
                    pf = self._parser.parse(repo, path, sha, content)
                except Exception as exc:  # isolate per-file failures
                    errors[path] = f"{type(exc).__name__}: {exc}"
                    continue
                if pf is not None:
                    parsed.append(pf)
            return parsed, errors
        loop = asyncio.get_running_loop()
        pool = self._ensure_pool()
        chunksize = max(1, len(items) // (self._workers * 4))
        results = await loop.run_in_executor(None, lambda: list(pool.map(_parse_in_worker, items, chunksize=chunksize)))
        for path, pf, err in results:
            if err:
                errors[path] = err
            elif pf is not None:
                parsed.append(pf)
        return parsed, errors

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
