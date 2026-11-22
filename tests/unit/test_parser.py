from __future__ import annotations

from code_rag.domain.entities import ParsedFile
from code_rag.domain.value_objects import EdgeKind, Language, SymbolKind
from code_rag.infrastructure.parsing.classify import detect_language, is_excluded, is_generated, is_test
from code_rag.infrastructure.parsing.treesitter_parser import TreeSitterParser

SHA = "a" * 40


def _parse(parser: TreeSitterParser, path: str, src: str, repo: str = "acme/demo") -> ParsedFile:
    pf = parser.parse(repo, path, SHA, src.encode())
    assert pf is not None
    return pf


def _by_qn(pf: ParsedFile) -> dict[str, object]:
    return {s.qualified_name: s for s in pf.symbols}


PY = '''"""Module doc."""
import os
from .client import Charge as C

TIMEOUT = 3
RETRIES = 2
NAME = "x"


@decorator
class Service(base.Base):
    """A service."""

    def run(self, x: int) -> int:
        return self.helper.compute(x) + C(x)

    async def stop(self) -> None:
        os.remove("f")


def top_level(a):
    return a
'''


def test_python_symbols_spans_and_parents(parser: TreeSitterParser) -> None:
    pf = _parse(parser, "pkg/svc.py", PY)
    assert pf.language is Language.PYTHON
    assert pf.module_name == "pkg.svc"
    syms = _by_qn(pf)
    cls = syms["pkg.svc.Service"]
    run = syms["pkg.svc.Service.run"]
    assert cls.kind is SymbolKind.CLASS  # type: ignore[attr-defined]
    assert cls.span.start == 10, "decorator must be part of the class span"  # type: ignore[attr-defined]
    assert run.kind is SymbolKind.METHOD and run.parent_id == cls.symbol_id  # type: ignore[attr-defined]
    assert syms["pkg.svc.top_level"].kind is SymbolKind.FUNCTION  # type: ignore[attr-defined]
    assert "pkg.svc" in syms, "module-level constants become a MODULE chunk"
    assert "C=pkg.client.Charge" in pf.imports
    targets = {(r.kind, r.target_name) for r in pf.refs}
    assert (EdgeKind.CALLS, "helper.compute") in targets
    assert (EdgeKind.INHERITS, "base.Base") in targets


def test_content_hash_ignores_formatting_and_line_shifts(parser: TreeSitterParser) -> None:
    base = _by_qn(_parse(parser, "pkg/svc.py", PY))
    shifted_src = "\n\n# a new comment\n" + PY.replace("return a", "return  a   # same")
    shifted = _by_qn(_parse(parser, "pkg/svc.py", shifted_src))
    a, b = base["pkg.svc.top_level"], shifted["pkg.svc.top_level"]
    assert a.symbol_id == b.symbol_id  # type: ignore[attr-defined]
    assert a.content_hash == b.content_hash  # type: ignore[attr-defined]
    assert a.span != b.span  # type: ignore[attr-defined]
    changed = _by_qn(_parse(parser, "pkg/svc.py", PY.replace("return a", "return a + 1")))
    assert changed["pkg.svc.top_level"].content_hash != a.content_hash  # type: ignore[attr-defined]


def test_container_hash_ignores_member_bodies(parser: TreeSitterParser) -> None:
    base = _by_qn(_parse(parser, "pkg/svc.py", PY))
    edited = _by_qn(_parse(parser, "pkg/svc.py", PY.replace('os.remove("f")', 'os.remove("g")')))
    assert base["pkg.svc.Service"].content_hash == edited["pkg.svc.Service"].content_hash  # type: ignore[attr-defined]
    assert base["pkg.svc.Service.stop"].content_hash != edited["pkg.svc.Service.stop"].content_hash  # type: ignore[attr-defined]


GO = """package server

import (
	pb "github.com/acme/proto/billing/v1"
)

// Server serves billing.
type Server struct {
	pb.UnimplementedBillingServiceServer
}

// Charge charges.
func (s *Server) Charge(ctx context.Context, r *pb.Req) (*pb.Resp, error) {
	return s.ledger.Record(r), nil
}

func helper() {}
"""


def test_go_methods_doc_comments_and_rpc(parser: TreeSitterParser) -> None:
    pf = _parse(parser, "internal/server/server.go", GO)
    syms = _by_qn(pf)
    struct = syms["internal.server.Server"]
    method = syms["internal.server.Server.Charge"]
    assert struct.kind is SymbolKind.STRUCT  # type: ignore[attr-defined]
    assert struct.span.start == 7, "doc comment belongs to the struct"  # type: ignore[attr-defined]
    assert method.parent_id == struct.symbol_id  # type: ignore[attr-defined]
    assert method.span.start == 12  # type: ignore[attr-defined]
    assert "pb=github.com.acme.proto.billing.v1" in pf.imports
    implicit = _parse(parser, "x/y.go", 'package y\nimport "github.com/acme/billing/v2"\n')
    assert implicit.imports == ("billing=github.com.acme.billing.v2",), "implicit alias skips the /vN segment"
    rpc = {(lnk.kind, lnk.endpoint.endpoint_id) for lnk in pf.rpc_links}
    assert (EdgeKind.SERVES_RPC, "billingservice/charge") in rpc


RUST = """use crate::billing::{charge, Ledger as L};

/// Storage.
pub trait Store {
    fn put(&mut self) -> i32;
}

#[derive(Default)]
pub struct Mem {}

impl Store for Mem {
    fn put(&mut self) -> i32 { charge::run(1) }
}
"""


def test_rust_impl_blocks_and_imports(parser: TreeSitterParser) -> None:
    pf = _parse(parser, "src/store.rs", RUST, repo="acme/ledger-rs")
    syms = _by_qn(pf)
    assert pf.module_name == "ledger_rs.store"
    assert syms["ledger_rs.store.Mem"].span.start == 8  # type: ignore[attr-defined]
    impl = syms["ledger_rs.store.Mem.impl_Store"]
    put = syms["ledger_rs.store.Mem.put"]
    assert impl.kind is SymbolKind.IMPL and put.parent_id == impl.symbol_id  # type: ignore[attr-defined]
    assert {"charge=ledger_rs.billing.charge", "L=ledger_rs.billing.Ledger"} <= set(pf.imports)
    assert (EdgeKind.IMPLEMENTS, "Store") in {(r.kind, r.target_name) for r in pf.refs}


TS = """import { applyDiscount } from "./pricing";
import { BillingServiceClient } from "../gen/billing_grpc_pb";

export const total = (x: number): number => applyDiscount(x);

export class Refunds extends Base implements Svc {
  private billing = new BillingServiceClient("b:1");
  async refund(id: string) { return this.billing.refund({ id }); }
}
"""


def test_typescript_arrow_functions_classes_rpc(parser: TreeSitterParser) -> None:
    pf = _parse(parser, "src/cart/refunds.ts", TS)
    syms = _by_qn(pf)
    assert syms["cart.refunds.total"].kind is SymbolKind.FUNCTION  # type: ignore[attr-defined]
    assert syms["cart.refunds.Refunds.refund"].kind is SymbolKind.METHOD  # type: ignore[attr-defined]
    assert "applyDiscount=cart.pricing.applyDiscount" in pf.imports
    refs = {(r.kind, r.target_name) for r in pf.refs}
    assert (EdgeKind.INHERITS, "Base") in refs and (EdgeKind.IMPLEMENTS, "Svc") in refs
    assert any(lnk.endpoint.endpoint_id == "billingservice/refund" for lnk in pf.rpc_links)


def test_proto_services_declare_rpc_endpoints(parser: TreeSitterParser) -> None:
    src = 'syntax = "proto3";\npackage b.v1;\nservice Pay {\n  rpc Charge(A) returns (B);\n}\nmessage A {}\n'
    pf = _parse(parser, "b.proto", src)
    syms = _by_qn(pf)
    assert syms["b.v1.Pay"].kind is SymbolKind.INTERFACE  # type: ignore[attr-defined]
    assert syms["b.v1.Pay.Charge"].span.start == 4  # type: ignore[attr-defined]
    assert pf.rpc_links[0].endpoint.endpoint_id == "pay/charge"


def test_classification() -> None:
    assert detect_language("a/b.py") is Language.PYTHON
    assert detect_language("x.d.ts") is None
    assert is_generated("gen/billing_pb2.py", "")
    assert is_generated("x.go", "// Code generated by protoc-gen-go. DO NOT EDIT.\n")
    assert not is_generated("x.go", "package x\n")
    assert is_test("pkg/foo_test.go") and is_test("tests/test_a.py") and not is_test("pkg/a.py")
    assert is_excluded("web/node_modules/react/index.js") and is_excluded("vendor/x.go")


def test_unsupported_or_binary_files_are_skipped(parser: TreeSitterParser) -> None:
    assert parser.parse("acme/x", "README.md", SHA, b"# hi") is None
    assert parser.parse("acme/x", "a.py", SHA, b"\x00\x01binary") is None
    assert parser.parse("acme/x", "a.py", SHA, b"") is None
