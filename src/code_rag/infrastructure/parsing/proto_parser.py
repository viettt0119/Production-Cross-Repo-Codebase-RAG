"""Lightweight .proto parser: services and rpc methods become symbols + DECLARES_RPC links.

Protobuf IDL is regular enough that a comment-stripping scanner is reliable for
`service` / `rpc` / `message` declarations, and it avoids shipping another grammar.
"""

from __future__ import annotations

import re

from code_rag.domain.entities import ParsedFile, RpcLink, SymbolNode
from code_rag.domain.value_objects import (
    EdgeKind,
    Language,
    LineSpan,
    RepoCoordinate,
    SymbolKind,
    make_symbol_id,
    sha256_hex,
)
from code_rag.infrastructure.parsing.extractors.base import endpoint

_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
_BLOCK_START = re.compile(r"^\s*(service|message|enum)\s+(\w+)\s*\{", re.MULTILINE)
_RPC = re.compile(r"^\s*rpc\s+(\w+)\s*\(\s*(stream\s+)?([\w.]+)\s*\)\s*returns\s*\(\s*(stream\s+)?([\w.]+)\s*\)")
_WS = re.compile(r"\s+")


def _strip_comments(text: str) -> str:
    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", blank, text)


def _block_end(lines: list[str], start_idx: int) -> int:
    depth = 0
    for i in range(start_idx, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0 and i >= start_idx and "{" in "".join(lines[start_idx : i + 1]):
            return i
    return len(lines) - 1


def parse_proto(repo: str, path: str, commit_sha: str, content: bytes, blob_sha: str) -> ParsedFile:
    raw = content.decode("utf-8", errors="replace")
    lines = raw.split("\n")
    clean_lines = _strip_comments(raw).split("\n")
    clean = "\n".join(clean_lines)
    pkg_match = _PACKAGE.search(clean)
    package = pkg_match.group(1) if pkg_match else path.rsplit("/", 1)[-1].removesuffix(".proto")
    coord = RepoCoordinate(repo=repo, path=path, commit_sha=commit_sha)
    symbols: list[SymbolNode] = []
    links: list[RpcLink] = []

    for m in _BLOCK_START.finditer(clean):
        kind_word, name = m.group(1), m.group(2)
        start_idx = clean.count("\n", 0, m.start())
        while start_idx < len(clean_lines) and not clean_lines[start_idx].strip():
            start_idx += 1
        end_idx = _block_end(clean_lines, start_idx)
        span = LineSpan(start=start_idx + 1, end=end_idx + 1)
        kind = {"service": SymbolKind.INTERFACE, "message": SymbolKind.STRUCT, "enum": SymbolKind.ENUM}[kind_word]
        qn = f"{package}.{name}"
        sym_id = make_symbol_id(repo, path, qn, kind)
        source = "\n".join(lines[start_idx : end_idx + 1])
        symbols.append(
            SymbolNode(
                symbol_id=sym_id,
                coord=coord,
                language=Language.PROTO,
                kind=kind,
                name=name,
                qualified_name=qn,
                span=span,
                signature=f"{kind_word} {name}",
                source=source,
                content_hash=sha256_hex(f"proto|{qn}|{_WS.sub(' ', ''.join(clean_lines[start_idx : end_idx + 1]))}"),
            )
        )
        if kind_word != "service":
            continue
        for i in range(start_idx + 1, end_idx + 1):
            rm = _RPC.match(clean_lines[i])
            if not rm:
                continue
            method = rm.group(1)
            rpc_qn = f"{qn}.{method}"
            rpc_id = make_symbol_id(repo, path, rpc_qn, SymbolKind.METHOD)
            end_i = i
            while end_i < end_idx and ";" not in clean_lines[end_i] and "}" not in clean_lines[end_i]:
                end_i += 1
            rpc_src = "\n".join(lines[i : end_i + 1])
            symbols.append(
                SymbolNode(
                    symbol_id=rpc_id,
                    coord=coord,
                    language=Language.PROTO,
                    kind=SymbolKind.METHOD,
                    name=method,
                    qualified_name=rpc_qn,
                    parent_id=sym_id,
                    span=LineSpan(start=i + 1, end=end_i + 1),
                    signature=_WS.sub(" ", rm.group(0)).strip(),
                    source=rpc_src,
                    content_hash=sha256_hex(f"proto-rpc|{rpc_qn}|{_WS.sub(' ', rm.group(0))}"),
                )
            )
            links.append(
                RpcLink(
                    symbol_id=rpc_id,
                    endpoint=endpoint(name, method),
                    kind=EdgeKind.DECLARES_RPC,
                    at_line=i + 1,
                )
            )

    return ParsedFile(
        repo=repo,
        path=path,
        commit_sha=commit_sha,
        language=Language.PROTO,
        blob_sha=blob_sha,
        module_name=package,
        symbols=tuple(symbols),
        refs=(),
        imports=(),
        rpc_links=tuple(links),
    )
