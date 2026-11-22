from __future__ import annotations

import posixpath
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

_IDENT = frozenset({"identifier", "property_identifier", "type_identifier", "this", "private_property_identifier"})
_FUNC_VALUES = frozenset({"arrow_function", "function_expression", "function", "generator_function"})


def _dotted(node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type in _IDENT:
        return node_text(node)
    if node.type == "member_expression":
        prop = node_text(node.child_by_field_name("property"))
        obj = _dotted(node.child_by_field_name("object"))
        return f"{obj}.{prop}" if obj else prop
    if node.type == "non_null_expression" and node.named_child_count:
        return _dotted(node.named_children[0])
    return None


def module_from_path(path: str) -> str:
    p = PurePosixPath(strip_dir_prefixes(path))
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "index":
        parts = parts[:-1]
    return ".".join(parts)


class TypeScriptExtractor(BaseExtractor):
    language = Language.TYPESCRIPT
    comment_types = frozenset({"comment"})
    call_types = frozenset({"call_expression", "new_expression"})
    descend_types = frozenset({"export_statement", "ambient_declaration", "statement_block"})
    strip_prefixes = ("this.", "super.")
    stop_names = frozenset(
        {
            "console.log",
            "console.error",
            "console.warn",
            "console.info",
            "console.debug",
            "JSON.stringify",
            "JSON.parse",
            "Object.keys",
            "Object.values",
            "Object.entries",
            "Object.assign",
            "Array.isArray",
            "Promise.all",
            "Promise.resolve",
            "Promise.reject",
            "String",
            "Number",
            "Boolean",
            "Error",
            "Date",
            "Map",
            "Set",
            "Array",
            "Promise",
            "parseInt",
            "parseFloat",
            "setTimeout",
            "clearTimeout",
            "require",
            "push",
            "map",
            "filter",
            "reduce",
            "forEach",
            "then",
            "catch",
            "finally",
            "join",
            "split",
            "trim",
            "toString",
            "slice",
            "includes",
            "indexOf",
            "find",
            "some",
            "every",
            "concat",
            "get",
            "set",
            "has",
            "delete",
            "keys",
            "values",
            "entries",
            "resolve",
            "reject",
        }
    )

    def module_name(self, repo: str, path: str) -> str:
        return module_from_path(path)

    def match_definition(self, node: Node, parent: _Def | None) -> DefMatch | None:
        t = node.type
        if t == "export_statement":
            decl = node.child_by_field_name("declaration")
            if decl is not None:
                m = self.match_definition(decl, parent)
                if m is not None:
                    m.span_node = node
                return m
            return None
        if t in ("function_declaration", "generator_function_declaration", "function_signature"):
            if t == "function_signature":
                return None
            return DefMatch(SymbolKind.FUNCTION, node_text(node.child_by_field_name("name")), node, node)
        if t in ("class_declaration", "abstract_class_declaration"):
            bases: list[tuple[str, EdgeKind]] = []
            for child in node.named_children:
                if child.type == "class_heritage":
                    for clause in child.named_children:
                        kind = EdgeKind.INHERITS if clause.type == "extends_clause" else EdgeKind.IMPLEMENTS
                        for b in clause.named_children:
                            name = _dotted(b) or (
                                _dotted(b.named_children[0])
                                if b.type == "generic_type" and b.named_child_count
                                else None
                            )
                            if name:
                                bases.append((name, kind))
            return DefMatch(
                SymbolKind.CLASS,
                node_text(node.child_by_field_name("name")),
                node,
                node,
                body=node.child_by_field_name("body"),
                bases=bases,
            )
        if t == "interface_declaration":
            bases = []
            for child in node.named_children:
                if child.type == "extends_type_clause":
                    for b in child.named_children:
                        name = _dotted(b)
                        if name:
                            bases.append((name, EdgeKind.INHERITS))
            return DefMatch(SymbolKind.INTERFACE, node_text(node.child_by_field_name("name")), node, node, bases=bases)
        if t == "enum_declaration":
            return DefMatch(SymbolKind.ENUM, node_text(node.child_by_field_name("name")), node, node)
        if t == "type_alias_declaration":
            return DefMatch(SymbolKind.TYPE_ALIAS, node_text(node.child_by_field_name("name")), node, node)
        if t == "method_definition" and parent is not None:
            return DefMatch(SymbolKind.METHOD, node_text(node.child_by_field_name("name")), node, node)
        if t == "public_field_definition" and parent is not None:
            value = node.child_by_field_name("value")
            if value is not None and value.type in _FUNC_VALUES:
                return DefMatch(SymbolKind.METHOD, node_text(node.child_by_field_name("name")), value, node)
            return None
        if t in ("lexical_declaration", "variable_declaration"):
            declarators = [c for c in node.named_children if c.type == "variable_declarator"]
            if len(declarators) == 1:
                value = declarators[0].child_by_field_name("value")
                if value is not None and value.type in _FUNC_VALUES:
                    name = node_text(declarators[0].child_by_field_name("name"))
                    return DefMatch(SymbolKind.FUNCTION, name, value, node)
        return None

    def callee(self, call: Node) -> str | None:
        fn = call.child_by_field_name("function") or call.child_by_field_name("constructor")
        if fn is None:
            return None
        dotted = _dotted(fn)
        if dotted:
            return dotted
        if fn.type == "member_expression":
            return node_text(fn.child_by_field_name("property"))
        return None

    def imports(self, root: Node, module_name: str) -> list[ImportBinding]:
        out: list[ImportBinding] = []
        file_dir = module_name.rsplit(".", 1)[0].replace(".", "/") if "." in module_name else ""
        for node in root.named_children:
            if node.type != "import_statement":
                continue
            src = node.child_by_field_name("source")
            spec = node_text(src).strip("'\"`")
            if not spec:
                continue
            if spec.startswith("."):
                resolved = posixpath.normpath(posixpath.join(file_dir, spec))
                mod = module_from_path(resolved + ".ts")
            else:
                mod = spec.lstrip("@").replace("/", ".")
            for clause in node.named_children:
                if clause.type != "import_clause":
                    continue
                for part in clause.named_children:
                    if part.type == "identifier":
                        out.append(ImportBinding(node_text(part), mod))
                    elif part.type == "namespace_import" and part.named_child_count:
                        out.append(ImportBinding(node_text(part.named_children[0]), mod))
                    elif part.type == "named_imports":
                        for s in part.named_children:
                            if s.type != "import_specifier":
                                continue
                            name = node_text(s.child_by_field_name("name"))
                            alias = node_text(s.child_by_field_name("alias")) or name
                            out.append(ImportBinding(alias, f"{mod}.{name}"))
        return out

    def rpc_links(self, root: Node, defs: list[_Def], ctx: FileContext) -> list[RpcLink]:
        client_vars: dict[str, str] = {}
        stack = [root]
        while stack:
            n = stack.pop()
            target: str | None = None
            value: Node | None = None
            if n.type == "variable_declarator":
                target, value = _dotted(n.child_by_field_name("name")), n.child_by_field_name("value")
            elif n.type == "assignment_expression":
                target, value = _dotted(n.child_by_field_name("left")), n.child_by_field_name("right")
            elif n.type == "public_field_definition":
                target, value = _dotted(n.child_by_field_name("name")), n.child_by_field_name("value")
            if target and value is not None and value.type == "new_expression":
                ctor = self.callee(value) or ""
                svc = service_from_type_name(ctor) if looks_like_rpc_type(ctor) else None
                if svc:
                    client_vars[target.removeprefix("this.")] = svc
            stack.extend(n.named_children)
        links: list[RpcLink] = []
        for d in defs:
            if d.symbol_id is None or d.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                continue
            for name, at in self._calls_in(d.match.span_node, set()):
                head, _, method = name.rpartition(".")
                svc = client_vars.get(head)
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


class TsxExtractor(TypeScriptExtractor):
    language = Language.TSX
