"""Language-agnostic tree-sitter walking machinery.

Each language extractor answers a handful of questions about node types
(what is a definition, what is a call, what are the imports, the bases, RPC hints).
`BaseExtractor.extract` turns those answers into SymbolNodes, SymbolRefs and RpcLinks.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from tree_sitter import Node, Tree

from code_rag.domain.entities import RpcEndpoint, RpcLink, SymbolNode, SymbolRef
from code_rag.domain.value_objects import (
    EdgeKind,
    Language,
    LineSpan,
    RepoCoordinate,
    SymbolId,
    SymbolKind,
    make_symbol_id,
    sha256_hex,
)

CONTAINER_SKELETON_MIN_LINES = 40
MIN_MODULE_RESIDUAL_LINES = 3
MAX_SIGNATURE_CHARS = 400

_WS = re.compile(r"\s+")
_GENERIC_ARGS = re.compile(r"<[^<>]*>")
_RPC_SUFFIXES = ("ServiceStub", "ServiceClient", "ServiceServicer", "ServiceServer")


@dataclass(slots=True)
class DefMatch:
    """What a language extractor reports for a definition node."""

    kind: SymbolKind | None  # None => scope-only container (e.g. Rust `mod`)
    name: str
    def_node: Node
    span_node: Node
    body: Node | None = None  # node whose children are member definitions (containers only)
    bases: list[tuple[str, EdgeKind]] = field(default_factory=list)
    qualifier: str | None = None  # explicit scope override (Go receiver, Rust impl type)


@dataclass(slots=True)
class _Def:
    match: DefMatch
    kind: SymbolKind
    qualified_name: str
    parent: _Def | None
    children: list[_Def] = field(default_factory=list)
    symbol_id: SymbolId | None = None
    lead_row: int | None = None  # first row of attached doc comments / attributes
    lead_text: str = ""


@dataclass(slots=True)
class ImportBinding:
    alias: str
    full: str

    def render(self) -> str:
        return f"{self.alias}={self.full}"


@dataclass(slots=True)
class ExtractionResult:
    module_name: str
    symbols: list[SymbolNode]
    refs: list[SymbolRef]
    imports: list[ImportBinding]
    rpc_links: list[RpcLink]


@dataclass(slots=True)
class FileContext:
    repo: str
    path: str
    commit_sha: str
    source: bytes
    text: str
    lines: list[str]
    is_generated: bool
    is_test: bool


def node_text(node: Node | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def line_span(node: Node) -> LineSpan:
    start = node.start_point.row + 1
    end_row = node.end_point.row
    if node.end_point.column == 0 and end_row > node.start_point.row:
        end_row -= 1
    return LineSpan(start=start, end=max(start, end_row + 1))


def service_from_type_name(type_name: str) -> str | None:
    """`BillingServiceStub` / `NewBillingServiceClient` / `UnimplementedBillingServiceServer` -> BillingService."""
    base = type_name.rsplit(".", 1)[-1]
    if base.startswith("New"):
        base = base[3:]
    if base.startswith("Unimplemented"):
        base = base[len("Unimplemented") :]
    for suffix in ("Stub", "Client", "Servicer", "Server"):
        if base.endswith(suffix) and len(base) > len(suffix):
            candidate = base[: -len(suffix)]
            return candidate or None
    return None


def looks_like_rpc_type(type_name: str) -> bool:
    base = type_name.rsplit(".", 1)[-1]
    return any(base.endswith(s) for s in _RPC_SUFFIXES)


def endpoint(service: str, method: str) -> RpcEndpoint:
    return RpcEndpoint(service=service.lower(), method=method.lower())


class BaseExtractor(ABC):
    language: Language
    comment_types: frozenset[str] = frozenset({"comment"})
    call_types: frozenset[str] = frozenset()
    descend_types: frozenset[str] = frozenset()
    stop_names: frozenset[str] = frozenset()
    strip_prefixes: tuple[str, ...] = ()
    leading_types: frozenset[str] = frozenset({"attribute_item", "decorator"})

    # ---- language hooks -------------------------------------------------
    @abstractmethod
    def module_name(self, repo: str, path: str) -> str:
        """Dotted module/package name used as the qualified-name prefix for a file."""

    @abstractmethod
    def match_definition(self, node: Node, parent: _Def | None) -> DefMatch | None:
        """Classify `node` as a definition (or scope-only container); None if it is neither."""

    @abstractmethod
    def callee(self, call: Node) -> str | None:
        """Dotted callee text of a call node (`a.b.c`), or None if not statically nameable."""

    @abstractmethod
    def imports(self, root: Node, module_name: str) -> list[ImportBinding]:
        """Import bindings (local alias -> fully qualified dotted name) declared by the file."""

    def body_of(self, def_node: Node) -> Node | None:
        return def_node.child_by_field_name("body")

    def rpc_links(self, root: Node, defs: list[_Def], ctx: FileContext) -> list[RpcLink]:
        return []

    # ---- generic machinery ----------------------------------------------
    def extract(self, tree: Tree, ctx: FileContext) -> ExtractionResult:
        module = self.module_name(ctx.repo, ctx.path)
        root = tree.root_node
        top: list[_Def] = []
        self._walk(root, [module] if module else [], None, top)
        flat = list(self._flatten(top))
        self._link_parents_by_qualifier(flat)
        for d in flat:
            self._attach_leading(d)

        coord = RepoCoordinate(repo=ctx.repo, path=ctx.path, commit_sha=ctx.commit_sha)
        ordinals: dict[tuple[str, SymbolKind], int] = {}
        for d in flat:
            key = (d.qualified_name, d.kind)
            ordinal = ordinals.get(key, 0)
            ordinals[key] = ordinal + 1
            d.symbol_id = make_symbol_id(ctx.repo, ctx.path, d.qualified_name, d.kind, ordinal)

        symbols: list[SymbolNode] = [self._to_symbol(d, ctx, coord) for d in flat]
        refs: list[SymbolRef] = []
        imports = self.imports(root, module)
        for d in flat:
            refs.extend(self._refs_for(d, ctx))

        module_symbol = self._module_symbol(root, top, ctx, coord, module)
        if module_symbol is not None:
            symbols.append(module_symbol)
            refs.extend(self._module_refs(root, top, module_symbol.symbol_id, ctx))

        rpc = self.rpc_links(root, flat, ctx)
        return ExtractionResult(module_name=module, symbols=symbols, refs=refs, imports=imports, rpc_links=rpc)

    def _walk(self, node: Node, scope: list[str], parent: _Def | None, out: list[_Def]) -> None:
        for child in node.named_children:
            match = self.match_definition(child, parent)
            if match is not None:
                if match.kind is None:
                    if match.body is not None:
                        self._walk(match.body, [*scope, match.name], parent, out)
                    continue
                qualifier = match.qualifier
                prefix = [*scope, qualifier] if qualifier else scope
                qn = ".".join([*prefix, match.name]) if prefix else match.name
                d = _Def(match=match, kind=match.kind, qualified_name=qn, parent=parent)
                out.append(d)
                if match.body is not None:
                    self._walk(match.body, [*prefix, match.name], d, d.children)
            elif child.type in self.descend_types:
                self._walk(child, scope, parent, out)

    def _flatten(self, defs: list[_Def]) -> Iterator[_Def]:
        for d in defs:
            yield d
            yield from self._flatten(d.children)

    def _link_parents_by_qualifier(self, flat: list[_Def]) -> None:
        """Go methods / Rust impl members declare their parent type by name, not by nesting."""
        by_qn = {d.qualified_name: d for d in flat if d.parent is None}
        for d in flat:
            if d.parent is None and d.match.qualifier:
                parent_qn = d.qualified_name.rsplit(".", 1)[0]
                parent = by_qn.get(parent_qn)
                if parent is not None and parent is not d:
                    d.parent = parent

    def _attach_leading(self, d: _Def) -> None:
        """Doc comments / attributes directly above a definition belong to it (Go, Rust, TS, JSDoc)."""
        node = d.match.span_node
        row = node.start_point.row
        texts: list[str] = []
        prev = node.prev_sibling
        while prev is not None and (prev.type in self.comment_types or prev.type in self.leading_types):
            if prev.end_point.row < row - 1:
                break  # blank line separates it from the definition
            row = prev.start_point.row
            texts.insert(0, node_text(prev))
            prev = prev.prev_sibling
        if row < node.start_point.row:
            d.lead_row = row
            d.lead_text = "\n".join(texts)

    def span_of(self, d: _Def) -> LineSpan:
        base = line_span(d.match.span_node)
        if d.lead_row is not None:
            return LineSpan(start=d.lead_row + 1, end=base.end)
        return base

    def _signature(self, m: DefMatch) -> str:
        start = m.span_node.start_byte
        body = self.body_of(m.def_node)
        end = body.start_byte if body is not None else m.def_node.end_byte
        raw = (m.span_node.text or b"")[: max(0, end - start)].decode("utf-8", errors="replace")
        if body is None:
            raw = raw.split("\n", 1)[0]
        sig = _WS.sub(" ", raw).strip().rstrip("{:").strip()
        return sig[:MAX_SIGNATURE_CHARS]

    def _iter_tokens(self, node: Node, skip: set[tuple[int, int]]) -> Iterator[str]:
        stack: list[tuple[Node, bool]] = [(node, True)]
        while stack:
            n, is_root = stack.pop()
            if n.type in self.comment_types:
                continue
            if not is_root and (n.start_byte, n.end_byte) in skip:
                continue
            if n.child_count == 0:
                if n.text:
                    yield n.text.decode("utf-8", errors="replace")
                continue
            stack.extend((c, False) for c in reversed(n.children))

    def _content_hash(self, d: _Def) -> str:
        skip = {(c.match.span_node.start_byte, c.match.span_node.end_byte) for c in d.children}
        tokens = " ".join(self._iter_tokens(d.match.span_node, skip))
        child_names = ",".join(c.qualified_name for c in d.children)
        doc = _WS.sub(" ", d.lead_text)
        return sha256_hex(f"{d.kind.value}|{d.qualified_name}|{doc}|{tokens}|{child_names}")

    def _source(self, d: _Def, ctx: FileContext, span: LineSpan) -> str:
        if d.children and span.length >= CONTAINER_SKELETON_MIN_LINES:
            return self._skeleton(d, ctx, span)
        return "\n".join(ctx.lines[span.start - 1 : span.end])

    def _skeleton(self, d: _Def, ctx: FileContext, span: LineSpan) -> str:
        """Container source with member bodies collapsed to their header line."""
        collapsed: dict[int, tuple[int, str]] = {}
        for c in d.children:
            cs = self.span_of(c)
            body = self.body_of(c.match.def_node)
            header_end = body.start_point.row + 1 if body is not None else cs.start
            header_end = min(max(header_end, cs.start), cs.end)
            if cs.end > header_end:
                collapsed[header_end + 1] = (cs.end, c.match.name)
        out: list[str] = []
        line_no = span.start
        while line_no <= span.end:
            if line_no in collapsed:
                end, name = collapsed[line_no]
                prev = ctx.lines[line_no - 2] if line_no >= 2 else ""
                indent = prev[: len(prev) - len(prev.lstrip())] + "    "
                out.append(f"{indent}⋯ body of {name}: lines {line_no}-{end}")
                line_no = end + 1
                continue
            out.append(ctx.lines[line_no - 1])
            line_no += 1
        return "\n".join(out)

    def _to_symbol(self, d: _Def, ctx: FileContext, coord: RepoCoordinate) -> SymbolNode:
        assert d.symbol_id is not None
        span = self.span_of(d)
        return SymbolNode(
            symbol_id=d.symbol_id,
            coord=coord,
            language=self.language,
            kind=d.kind,
            name=d.match.name,
            qualified_name=d.qualified_name,
            parent_id=d.parent.symbol_id if d.parent is not None else None,
            span=span,
            signature=self._signature(d.match),
            source=self._source(d, ctx, span),
            content_hash=self._content_hash(d),
            is_generated=ctx.is_generated,
            is_test=ctx.is_test,
        )

    def normalize_callee(self, raw: str) -> str | None:
        text = raw.replace("::", ".").replace("?.", ".").replace("!.", ".")
        text = _GENERIC_ARGS.sub("", _GENERIC_ARGS.sub("", text))
        for p in self.strip_prefixes:
            if text.startswith(p):
                text = text[len(p) :]
        text = text.strip(". ")
        if not text or any(ch in text for ch in "()[]{}\"' \n"):
            return None
        if text in self.stop_names or text.rsplit(".", 1)[-1] in self.stop_names:
            return None
        return text

    def _calls_in(self, node: Node, skip: set[tuple[int, int]]) -> Iterator[tuple[str, int]]:
        stack = [node]
        while stack:
            n = stack.pop()
            if n is not node and (n.start_byte, n.end_byte) in skip:
                continue
            if n.type in self.call_types:
                raw = self.callee(n)
                if raw:
                    name = self.normalize_callee(raw)
                    if name:
                        yield name, n.start_point.row + 1
            stack.extend(reversed(n.named_children))

    def _refs_for(self, d: _Def, ctx: FileContext) -> list[SymbolRef]:
        assert d.symbol_id is not None
        skip = {(c.match.span_node.start_byte, c.match.span_node.end_byte) for c in d.children}
        refs: list[SymbolRef] = []
        seen: set[tuple[str, EdgeKind]] = set()
        line = d.match.span_node.start_point.row + 1
        for base, kind in d.match.bases:
            norm = self.normalize_callee(base)
            if norm and (norm, kind) not in seen:
                seen.add((norm, kind))
                refs.append(
                    SymbolRef(
                        src_id=d.symbol_id,
                        src_repo=ctx.repo,
                        src_path=ctx.path,
                        kind=kind,
                        target_name=norm,
                        at_line=line,
                    )
                )
        for name, at in self._calls_in(d.match.span_node, skip):
            if (name, EdgeKind.CALLS) in seen:
                continue
            seen.add((name, EdgeKind.CALLS))
            refs.append(
                SymbolRef(
                    src_id=d.symbol_id,
                    src_repo=ctx.repo,
                    src_path=ctx.path,
                    kind=EdgeKind.CALLS,
                    target_name=name,
                    at_line=at,
                )
            )
        return refs

    def _residual_lines(self, root: Node, top: list[_Def], ctx: FileContext) -> list[int]:
        covered: set[int] = set()
        for d in top:
            s = self.span_of(d)
            covered.update(range(s.start, s.end + 1))
        residual: list[int] = []
        for i, text in enumerate(ctx.lines, start=1):
            if i in covered or not text.strip():
                continue
            residual.append(i)
        return residual

    def _module_symbol(
        self, root: Node, top: list[_Def], ctx: FileContext, coord: RepoCoordinate, module: str
    ) -> SymbolNode | None:
        residual = self._residual_lines(root, top, ctx)
        if len(residual) < MIN_MODULE_RESIDUAL_LINES:
            return None
        span = LineSpan(start=residual[0], end=residual[-1])
        source = "\n".join(ctx.lines[i - 1] for i in residual)
        qn = module or ctx.path
        return SymbolNode(
            symbol_id=make_symbol_id(ctx.repo, ctx.path, qn, SymbolKind.MODULE),
            coord=coord,
            language=self.language,
            kind=SymbolKind.MODULE,
            name=qn.rsplit(".", 1)[-1],
            qualified_name=qn,
            span=span,
            signature=f"module {qn}",
            source=source,
            content_hash=sha256_hex("module|" + _WS.sub(" ", source)),
            is_generated=ctx.is_generated,
            is_test=ctx.is_test,
        )

    def _module_refs(self, root: Node, top: list[_Def], module_id: SymbolId, ctx: FileContext) -> list[SymbolRef]:
        skip = {(d.match.span_node.start_byte, d.match.span_node.end_byte) for d in top}
        out: list[SymbolRef] = []
        seen: set[str] = set()
        for name, at in self._calls_in(root, skip):
            if name in seen:
                continue
            seen.add(name)
            out.append(
                SymbolRef(
                    src_id=module_id,
                    src_repo=ctx.repo,
                    src_path=ctx.path,
                    kind=EdgeKind.CALLS,
                    target_name=name,
                    at_line=at,
                )
            )
        return out


def strip_dir_prefixes(path: str, prefixes: tuple[str, ...] = ("src/", "lib/", "app/")) -> str:
    for p in prefixes:
        if path.startswith(p):
            return path[len(p) :]
    return path
