from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Node

from code_rag.domain.entities import RpcLink
from code_rag.domain.value_objects import EdgeKind, Language, SymbolKind

from .base import (
    BaseExtractor,
    DefMatch,
    FileContext,
    ImportBinding,
    _Def,
    endpoint,
    looks_like_rpc_type,
    node_text,
    service_from_type_name,
    strip_dir_prefixes,
)

_IDENT = frozenset({"identifier"})


def _dotted(node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type in _IDENT:
        return node_text(node)
    if node.type == "attribute":
        attr = node_text(node.child_by_field_name("attribute"))
        obj = _dotted(node.child_by_field_name("object"))
        return f"{obj}.{attr}" if obj else attr
    return None


class PythonExtractor(BaseExtractor):
    language = Language.PYTHON
    comment_types = frozenset({"comment"})
    call_types = frozenset({"call"})
    descend_types = frozenset(
        {
            "if_statement",
            "else_clause",
            "elif_clause",
            "try_statement",
            "except_clause",
            "finally_clause",
            "block",
        }
    )
    strip_prefixes = ("self.", "cls.", "super().")
    stop_names = frozenset(
        {
            "print",
            "len",
            "range",
            "isinstance",
            "issubclass",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "super",
            "getattr",
            "setattr",
            "hasattr",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "min",
            "max",
            "sum",
            "any",
            "all",
            "open",
            "repr",
            "type",
            "iter",
            "next",
            "abs",
            "round",
            "format",
            "vars",
            "id",
            "hash",
            "object",
            "property",
            "staticmethod",
            "classmethod",
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "RuntimeError",
            "NotImplementedError",
            "append",
            "extend",
            "get",
            "items",
            "keys",
            "values",
            "pop",
            "update",
            "join",
            "split",
            "strip",
            "startswith",
            "endswith",
            "replace",
            "lower",
            "upper",
            "encode",
            "decode",
            "remove",
            "copy",
            "dataclass",
            "field",
            "scaleb",
        }
    )

    def module_name(self, repo: str, path: str) -> str:
        p = PurePosixPath(strip_dir_prefixes(path))
        parts = list(p.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def match_definition(self, node: Node, parent: _Def | None) -> DefMatch | None:
        if node.type == "decorated_definition":
            inner = node.child_by_field_name("definition")
            if inner is None:
                return None
            m = self.match_definition(inner, parent)
            if m is not None:
                m.span_node = node
            return m
        if node.type == "function_definition":
            in_class = parent is not None and parent.kind is SymbolKind.CLASS
            return DefMatch(
                kind=SymbolKind.METHOD if in_class else SymbolKind.FUNCTION,
                name=node_text(node.child_by_field_name("name")),
                def_node=node,
                span_node=node,
            )
        if node.type == "class_definition":
            bases: list[tuple[str, EdgeKind]] = []
            sup = node.child_by_field_name("superclasses")
            if sup is not None:
                for arg in sup.named_children:
                    name = _dotted(arg)
                    if name and name not in ("object", "ABC", "Protocol", "Generic"):
                        bases.append((name, EdgeKind.INHERITS))
            return DefMatch(
                kind=SymbolKind.CLASS,
                name=node_text(node.child_by_field_name("name")),
                def_node=node,
                span_node=node,
                body=node.child_by_field_name("body"),
                bases=bases,
            )
        return None

    def callee(self, call: Node) -> str | None:
        fn = call.child_by_field_name("function")
        if fn is None:
            return None
        dotted = _dotted(fn)
        if dotted:
            return dotted
        if fn.type == "attribute":  # chained call: a().b() -> "b"
            return node_text(fn.child_by_field_name("attribute"))
        return None

    def imports(self, root: Node, module_name: str) -> list[ImportBinding]:
        out: list[ImportBinding] = []
        for node in root.named_children:
            if node.type == "import_statement":
                for name_node in node.children_by_field_name("name"):
                    if name_node.type == "aliased_import":
                        full = node_text(name_node.child_by_field_name("name"))
                        out.append(ImportBinding(node_text(name_node.child_by_field_name("alias")), full))
                    else:
                        full = node_text(name_node)
                        out.append(ImportBinding(full.split(".")[0], full.split(".")[0]))
            elif node.type == "import_from_statement":
                mod_node = node.child_by_field_name("module_name")
                base = self._resolve_from(mod_node, module_name)
                for name_node in node.children_by_field_name("name"):
                    if name_node.type == "aliased_import":
                        name = node_text(name_node.child_by_field_name("name"))
                        alias = node_text(name_node.child_by_field_name("alias"))
                    else:
                        name = alias = node_text(name_node)
                    out.append(ImportBinding(alias, f"{base}.{name}" if base else name))
        return out

    @staticmethod
    def _resolve_from(mod_node: Node | None, module_name: str) -> str:
        if mod_node is None:
            return ""
        if mod_node.type != "relative_import":
            return node_text(mod_node)
        text = node_text(mod_node)
        level = len(text) - len(text.lstrip("."))
        rest = text.lstrip(".")
        pkg_parts = module_name.split(".")[:-1] if module_name else []
        if level > 1:
            pkg_parts = pkg_parts[: max(0, len(pkg_parts) - (level - 1))]
        return ".".join([*pkg_parts, rest] if rest else pkg_parts)

    def rpc_links(self, root: Node, defs: list[_Def], ctx: FileContext) -> list[RpcLink]:
        links: list[RpcLink] = []
        stub_vars: dict[str, str] = {}
        # 1) variables assigned from `<...>XxxServiceStub(channel)`
        stack = [root]
        while stack:
            n = stack.pop()
            if n.type == "assignment":
                right = n.child_by_field_name("right")
                left = _dotted(n.child_by_field_name("left"))
                if right is not None and right.type == "call" and left:
                    ctor = self.callee(right) or ""
                    svc = service_from_type_name(ctor) if looks_like_rpc_type(ctor) else None
                    if svc:
                        stub_vars[left.removeprefix("self.").removeprefix("cls.")] = svc
            stack.extend(n.named_children)
        # 2) client calls + 3) servicer implementations
        for d in defs:
            if d.symbol_id is None:
                continue
            for base, _ in d.match.bases:
                if base.endswith("Servicer"):
                    svc = service_from_type_name(base)
                    if svc:
                        for c in d.children:
                            if c.symbol_id and not c.match.name.startswith("_"):
                                links.append(
                                    RpcLink(
                                        symbol_id=c.symbol_id,
                                        endpoint=endpoint(svc, c.match.name),
                                        kind=EdgeKind.SERVES_RPC,
                                        at_line=c.match.span_node.start_point.row + 1,
                                    )
                                )
            if d.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                continue
            skip = {(c.match.span_node.start_byte, c.match.span_node.end_byte) for c in d.children}
            for name, at in self._calls_in(d.match.span_node, skip):
                head, _, method = name.rpartition(".")
                svc = stub_vars.get(head)
                if svc and method:
                    links.append(
                        RpcLink(
                            symbol_id=d.symbol_id,
                            endpoint=endpoint(svc, method),
                            kind=EdgeKind.CALLS_RPC,
                            at_line=at,
                        )
                    )
        return links
