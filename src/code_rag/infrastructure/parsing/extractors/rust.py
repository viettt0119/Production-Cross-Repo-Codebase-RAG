from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Node

from code_rag.domain.value_objects import EdgeKind, Language, SymbolKind

from .base import BaseExtractor, DefMatch, ImportBinding, _Def, node_text

_IDENT = frozenset({"identifier", "type_identifier", "field_identifier", "self", "crate", "super"})


def _dotted(node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type in _IDENT:
        return node_text(node)
    if node.type in ("scoped_identifier", "scoped_type_identifier"):
        name = node_text(node.child_by_field_name("name"))
        path = _dotted(node.child_by_field_name("path"))
        return f"{path}.{name}" if path else name
    if node.type == "field_expression":
        field = node_text(node.child_by_field_name("field"))
        value = _dotted(node.child_by_field_name("value"))
        return f"{value}.{field}" if value else field
    if node.type == "generic_type":
        return _dotted(node.child_by_field_name("type"))
    if node.type == "generic_function":
        return _dotted(node.child_by_field_name("function"))
    return None


class RustExtractor(BaseExtractor):
    language = Language.RUST
    comment_types = frozenset({"line_comment", "block_comment"})
    call_types = frozenset({"call_expression"})
    descend_types = frozenset()
    strip_prefixes = ("self.", "Self.", "crate.", "super.")
    stop_names = frozenset(
        {
            "Some",
            "None",
            "Ok",
            "Err",
            "Box.new",
            "Vec.new",
            "String.new",
            "String.from",
            "HashMap.new",
            "clone",
            "unwrap",
            "expect",
            "to_string",
            "into",
            "iter",
            "map",
            "collect",
            "len",
            "push",
            "is_empty",
            "as_ref",
            "as_str",
            "get",
            "insert",
            "contains",
            "format",
            "Default.default",
        }
    )

    def __init__(self, crate_name: str | None = None) -> None:
        self._crate_name = crate_name

    def module_name(self, repo: str, path: str) -> str:
        p = PurePosixPath(path)
        parts = list(p.with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        crate = self._crate_name or repo.rsplit("/", maxsplit=1)[-1].replace("-", "_")
        if (parts and parts[-1] in ("lib", "main")) or (parts and parts[-1] == "mod"):
            parts = parts[:-1]
        return ".".join([crate, *parts])

    def match_definition(self, node: Node, parent: _Def | None) -> DefMatch | None:
        t = node.type
        name = node_text(node.child_by_field_name("name"))
        if t == "function_item":
            in_type = parent is not None and parent.kind in (SymbolKind.IMPL, SymbolKind.TRAIT)
            return DefMatch(SymbolKind.METHOD if in_type else SymbolKind.FUNCTION, name, node, node)
        if t == "struct_item":
            return DefMatch(SymbolKind.STRUCT, name, node, node)
        if t == "enum_item":
            return DefMatch(SymbolKind.ENUM, name, node, node)
        if t == "union_item":
            return DefMatch(SymbolKind.STRUCT, name, node, node)
        if t == "type_item":
            return DefMatch(SymbolKind.TYPE_ALIAS, name, node, node)
        if t == "trait_item":
            bases: list[tuple[str, EdgeKind]] = []
            bounds = node.child_by_field_name("bounds")
            if bounds is not None:
                for b in bounds.named_children:
                    bn = _dotted(b)
                    if bn:
                        bases.append((bn, EdgeKind.INHERITS))
            return DefMatch(SymbolKind.TRAIT, name, node, node, body=node.child_by_field_name("body"), bases=bases)
        if t == "impl_item":
            type_name = _dotted(node.child_by_field_name("type")) or "Unknown"
            trait = _dotted(node.child_by_field_name("trait"))
            short_type = type_name.rsplit(".", 1)[-1]
            impl_bases = (
                [(trait, EdgeKind.IMPLEMENTS), (type_name, EdgeKind.INHERITS)]
                if trait
                else [(type_name, EdgeKind.INHERITS)]
            )
            label = f"impl_{trait.rsplit('.', 1)[-1]}" if trait else "impl"
            return DefMatch(
                SymbolKind.IMPL,
                label,
                node,
                node,
                body=node.child_by_field_name("body"),
                bases=impl_bases,
                qualifier=short_type,
            )
        if t == "mod_item":
            body = node.child_by_field_name("body")
            if body is not None:
                return DefMatch(kind=None, name=name, def_node=node, span_node=node, body=body)
        return None

    def _walk(self, node: Node, scope: list[str], parent: _Def | None, out: list[_Def]) -> None:
        for child in node.named_children:
            match = self.match_definition(child, parent)
            if match is None:
                continue
            if match.kind is None:
                if match.body is not None:
                    self._walk(match.body, [*scope, match.name], parent, out)
                continue
            if match.kind is SymbolKind.IMPL:
                type_scope = [*scope, match.qualifier or ""]
                d = _Def(
                    match=match,
                    kind=SymbolKind.IMPL,
                    qualified_name=".".join([*type_scope, match.name]),
                    parent=parent,
                )
                out.append(d)
                if match.body is not None:
                    # members are qualified as Type.method, not Type.impl.method
                    self._walk(match.body, type_scope, d, d.children)
                continue
            qn = ".".join([*scope, match.name])
            d = _Def(match=match, kind=match.kind, qualified_name=qn, parent=parent)
            out.append(d)
            if match.body is not None:
                self._walk(match.body, [*scope, match.name], d, d.children)

    def callee(self, call: Node) -> str | None:
        fn = call.child_by_field_name("function")
        if fn is None:
            return None
        dotted = _dotted(fn)
        if dotted:
            return dotted
        if fn.type == "field_expression":
            return node_text(fn.child_by_field_name("field"))
        return None

    def imports(self, root: Node, module_name: str) -> list[ImportBinding]:
        out: list[ImportBinding] = []
        crate = module_name.split(".", maxsplit=1)[0]
        parent_mod = module_name.rsplit(".", 1)[0] if "." in module_name else crate

        def norm(path: str) -> str:
            path = path.replace("::", ".")
            if path.startswith("crate."):
                return f"{crate}.{path[6:]}"
            if path.startswith("super."):
                return f"{parent_mod}.{path[6:]}"
            if path.startswith("self."):
                return f"{module_name}.{path[5:]}"
            return path

        def expand(node: Node, prefix: str) -> None:
            t = node.type
            if t in ("identifier", "scoped_identifier", "crate", "self", "super"):
                full = norm(f"{prefix}{node_text(node)}" if prefix else node_text(node))
                alias = full.rsplit(".", 1)[-1]
                if alias in ("self",):
                    full = full.rsplit(".", 1)[0]
                    alias = full.rsplit(".", 1)[-1]
                out.append(ImportBinding(alias, full))
            elif t == "use_as_clause":
                full = norm(prefix + node_text(node.child_by_field_name("path")))
                out.append(ImportBinding(node_text(node.child_by_field_name("alias")), full))
            elif t == "scoped_use_list":
                path = node.child_by_field_name("path")
                new_prefix = f"{prefix}{node_text(path)}::" if path is not None else prefix
                lst = node.child_by_field_name("list")
                if lst is not None:
                    for item in lst.named_children:
                        expand(item, new_prefix)
            elif t == "use_list":
                for item in node.named_children:
                    expand(item, prefix)

        stack = [root]
        while stack:
            n = stack.pop()
            if n.type == "use_declaration":
                arg = n.child_by_field_name("argument")
                if arg is not None:
                    expand(arg, "")
            elif n.type in ("source_file", "mod_item", "declaration_list"):
                stack.extend(n.named_children)
        return out
