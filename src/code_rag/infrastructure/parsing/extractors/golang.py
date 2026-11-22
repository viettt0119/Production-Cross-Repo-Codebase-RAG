from __future__ import annotations

import re
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
    node_text,
    service_from_type_name,
)

_VERSION_SEGMENT = re.compile(r"^v\d+$")
_IDENT = frozenset({"identifier", "field_identifier", "package_identifier", "type_identifier"})


def _dotted(node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type in _IDENT:
        return node_text(node)
    if node.type == "selector_expression":
        field = node_text(node.child_by_field_name("field"))
        operand = _dotted(node.child_by_field_name("operand"))
        return f"{operand}.{field}" if operand else field
    if node.type == "qualified_type":
        return f"{node_text(node.child_by_field_name('package'))}.{node_text(node.child_by_field_name('name'))}"
    if node.type == "pointer_type" and node.named_child_count:
        return _dotted(node.named_children[0])
    if node.type == "generic_type":
        return _dotted(node.child_by_field_name("type"))
    return None


def _receiver_type(method: Node) -> str | None:
    recv = method.child_by_field_name("receiver")
    if recv is None:
        return None
    stack = [recv]
    while stack:
        n = stack.pop()
        if n.type == "type_identifier":
            return node_text(n)
        stack.extend(reversed(n.named_children))
    return None


class GoExtractor(BaseExtractor):
    language = Language.GO
    comment_types = frozenset({"comment"})
    call_types = frozenset({"call_expression"})
    descend_types = frozenset()
    stop_names = frozenset(
        {
            "len",
            "cap",
            "make",
            "new",
            "append",
            "copy",
            "delete",
            "panic",
            "recover",
            "print",
            "println",
            "close",
            "string",
            "int",
            "int64",
            "float64",
            "byte",
            "error",
            "errors.New",
            "errors.Is",
            "errors.As",
            "fmt.Println",
            "fmt.Printf",
            "fmt.Sprintf",
            "fmt.Errorf",
            "fmt.Sprint",
            "log.Printf",
            "log.Println",
            "log.Fatal",
            "log.Fatalf",
        }
    )

    def module_name(self, repo: str, path: str) -> str:
        parent = PurePosixPath(path).parent
        if str(parent) in ("", "."):
            return repo.rsplit("/", maxsplit=1)[-1].replace("-", "_")
        return ".".join(parent.parts)

    def match_definition(self, node: Node, parent: _Def | None) -> DefMatch | None:
        t = node.type
        if t == "function_declaration":
            return DefMatch(SymbolKind.FUNCTION, node_text(node.child_by_field_name("name")), node, node)
        if t == "method_declaration":
            recv = _receiver_type(node)
            return DefMatch(SymbolKind.METHOD, node_text(node.child_by_field_name("name")), node, node, qualifier=recv)
        if t == "type_declaration":
            specs = [c for c in node.named_children if c.type in ("type_spec", "type_alias")]
            if len(specs) == 1:
                return self._type_spec(specs[0], span_node=node)
            return DefMatch(kind=None, name="", def_node=node, span_node=node, body=None)
        if t in ("type_spec", "type_alias"):
            return self._type_spec(node, span_node=node)
        return None

    def _walk(self, node: Node, scope: list[str], parent: _Def | None, out: list[_Def]) -> None:
        # Grouped `type ( A ...; B ... )` declarations: emit one symbol per spec.
        for child in node.named_children:
            if child.type == "type_declaration":
                specs = [c for c in child.named_children if c.type in ("type_spec", "type_alias")]
                if len(specs) > 1:
                    for spec in specs:
                        m = self._type_spec(spec, span_node=spec)
                        out.append(
                            _Def(
                                match=m,
                                kind=m.kind or SymbolKind.TYPE_ALIAS,
                                qualified_name=".".join([*scope, m.name]),
                                parent=parent,
                            )
                        )
                    continue
            match = self.match_definition(child, parent)
            if match is None or match.kind is None:
                continue
            prefix = [*scope, match.qualifier] if match.qualifier else scope
            out.append(
                _Def(
                    match=match,
                    kind=match.kind,
                    qualified_name=".".join([*prefix, match.name]),
                    parent=parent,
                )
            )

    def _type_spec(self, spec: Node, span_node: Node) -> DefMatch:
        name = node_text(spec.child_by_field_name("name"))
        typ = spec.child_by_field_name("type")
        bases: list[tuple[str, EdgeKind]] = []
        kind = SymbolKind.TYPE_ALIAS
        if typ is not None and typ.type == "struct_type":
            kind = SymbolKind.STRUCT
            for fl in typ.named_children:
                if fl.type != "field_declaration_list":
                    continue
                for field in fl.named_children:
                    if field.type == "field_declaration" and field.child_by_field_name("name") is None:
                        embedded = _dotted(field.child_by_field_name("type"))
                        if embedded:
                            bases.append((embedded, EdgeKind.INHERITS))
        elif typ is not None and typ.type == "interface_type":
            kind = SymbolKind.INTERFACE
            for elem in typ.named_children:
                if elem.type == "type_elem":
                    for t in elem.named_children:
                        embedded = _dotted(t)
                        if embedded:
                            bases.append((embedded, EdgeKind.INHERITS))
        return DefMatch(kind, name, spec, span_node, bases=bases)

    def body_of(self, def_node: Node) -> Node | None:
        if def_node.type in ("type_spec", "type_alias"):
            return None
        return def_node.child_by_field_name("body")

    def callee(self, call: Node) -> str | None:
        fn = call.child_by_field_name("function")
        if fn is None:
            return None
        dotted = _dotted(fn)
        if dotted:
            return dotted
        if fn.type == "selector_expression":
            return node_text(fn.child_by_field_name("field"))
        return None

    def imports(self, root: Node, module_name: str) -> list[ImportBinding]:
        out: list[ImportBinding] = []
        stack = [c for c in root.named_children if c.type == "import_declaration"]
        while stack:
            n = stack.pop()
            if n.type == "import_spec":
                path = node_text(n.child_by_field_name("path")).strip('"`')
                segs = [s for s in path.split("/") if s]
                if not segs:
                    continue
                last = segs[-1]
                if _VERSION_SEGMENT.match(last) and len(segs) > 1:
                    last = segs[-2]
                name_node = n.child_by_field_name("name")
                alias = node_text(name_node) if name_node is not None else last.replace("-", "_")
                if alias in ("_", "."):
                    continue
                out.append(ImportBinding(alias, ".".join(segs)))
            else:
                stack.extend(n.named_children)
        return out

    def rpc_links(self, root: Node, defs: list[_Def], ctx: FileContext) -> list[RpcLink]:
        links: list[RpcLink] = []
        struct_service: dict[str, str] = {}
        client_vars: dict[str, str] = {}
        for d in defs:
            if d.kind is SymbolKind.STRUCT:
                for base, _ in d.match.bases:
                    if "Unimplemented" in base and base.endswith("Server"):
                        svc = service_from_type_name(base)
                        if svc:
                            struct_service[d.match.name] = svc
                # struct fields typed as generated clients: `billing pb.BillingServiceClient`
                typ = d.match.def_node.child_by_field_name("type")
                if typ is not None:
                    for fl in typ.named_children:
                        for field in fl.named_children:
                            fname = field.child_by_field_name("name")
                            ftype = _dotted(field.child_by_field_name("type"))
                            if fname is not None and ftype and ftype.endswith("Client"):
                                svc = service_from_type_name(ftype)
                                if svc:
                                    client_vars[node_text(fname)] = svc
        stack = [root]
        while stack:
            n = stack.pop()
            if n.type in ("short_var_declaration", "assignment_statement", "var_spec"):
                left = n.child_by_field_name("left") or n.child_by_field_name("name")
                right = n.child_by_field_name("right") or n.child_by_field_name("value")
                if left is not None and right is not None:
                    lefts = [_dotted(x) for x in (left.named_children if left.type == "expression_list" else [left])]
                    rights = right.named_children if right.type == "expression_list" else [right]
                    for lname, rnode in zip(lefts, rights, strict=False):
                        if lname and rnode.type == "call_expression":
                            ctor = self.callee(rnode) or ""
                            base = ctor.rsplit(".", 1)[-1]
                            if base.startswith("New") and base.endswith("Client"):
                                svc = service_from_type_name(base)
                                if svc:
                                    client_vars[lname.rsplit(".", 1)[-1]] = svc
            elif n.type == "call_expression":
                ctor = (self.callee(n) or "").rsplit(".", 1)[-1]
                if ctor.startswith("Register") and ctor.endswith("Server"):
                    svc = ctor[len("Register") : -len("Server")]
                    args = n.child_by_field_name("arguments")
                    if args is not None and args.named_child_count >= 2:
                        impl = args.named_children[1]
                        tname = None
                        for sub in [impl, *impl.named_children]:
                            if sub.type == "composite_literal":
                                tname = _dotted(sub.child_by_field_name("type"))
                        if tname and svc:
                            struct_service.setdefault(tname.rsplit(".", 1)[-1], svc)
            stack.extend(n.named_children)
        for d in defs:
            if d.symbol_id is None or d.kind is not SymbolKind.METHOD:
                continue
            svc = struct_service.get(d.match.qualifier or "")
            if svc and d.match.name[:1].isupper():
                links.append(
                    RpcLink(
                        symbol_id=d.symbol_id,
                        endpoint=endpoint(svc, d.match.name),
                        kind=EdgeKind.SERVES_RPC,
                        at_line=d.match.span_node.start_point.row + 1,
                    )
                )
        for d in defs:
            if d.symbol_id is None or d.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                continue
            for name, at in self._calls_in(d.match.span_node, set()):
                parts = name.split(".")
                if len(parts) >= 2 and parts[-2] in client_vars:
                    links.append(
                        RpcLink(
                            symbol_id=d.symbol_id,
                            endpoint=endpoint(client_vars[parts[-2]], parts[-1]),
                            kind=EdgeKind.CALLS_RPC,
                            at_line=at,
                        )
                    )
        return links
