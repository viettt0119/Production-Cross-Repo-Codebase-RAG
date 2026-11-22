from __future__ import annotations

from pathlib import Path

from code_rag.application.ingest.symbol_differ import diff_symbols
from code_rag.application.ingest.symbol_resolver import SymbolResolver
from code_rag.domain.changeset import LedgerEntry
from code_rag.domain.value_objects import EdgeKind
from code_rag.infrastructure.parsing.treesitter_parser import TreeSitterParser
from code_rag.infrastructure.stores.sqlite_hash_ledger import SqliteHashLedger

SHA = "b" * 40

V1 = """def a():
    return 1


def b():
    return 2


def c():
    return 3
"""

# a: shifted only, b: body changed, c: deleted, d: added
V2 = """

def a():
    return 1


def b():
    return 20


def d():
    return 4
"""


def _entries(parser: TreeSitterParser, src: str) -> list[LedgerEntry]:
    pf = parser.parse("acme/x", "m.py", SHA, src.encode())
    assert pf is not None
    return [
        LedgerEntry(
            symbol_id=s.symbol_id,
            repo=s.repo,
            path=s.path,
            name=s.name,
            qualified_name=s.qualified_name,
            kind=s.kind.value,
            content_hash=s.content_hash,
            start_line=s.span.start,
            end_line=s.span.end,
            commit_sha=SHA,
        )
        for s in pf.symbols
    ]


def test_diff_classifies_embed_relocate_delete(parser: TreeSitterParser) -> None:
    old = _entries(parser, V1)
    new = parser.parse("acme/x", "m.py", SHA, V2.encode())
    assert new is not None
    delta = diff_symbols(new.symbols, old)
    assert {s.name for s in delta.to_embed} == {"b", "d"}
    assert {s.name for s in delta.to_relocate} == {"a"}
    deleted = {e.name for e in old if e.symbol_id in set(delta.to_delete)}
    assert deleted == {"c"}
    assert diff_symbols(new.symbols, _entries(parser, V2)).touched == 0


def test_resolver_links_across_repos_via_imports(tmp_path: Path, parser: TreeSitterParser) -> None:
    ledger = SqliteHashLedger(tmp_path / "l.db")
    money = parser.parse(
        "acme/shared-py",
        "acme_common/money.py",
        SHA,
        b"class Money:\n    def to_minor_units(self):\n        return 1\n\ndef parse_money(raw):\n    return Money()\n",
    )
    svc = parser.parse(
        "acme/checkout-py",
        "checkout/service.py",
        SHA,
        b"from acme_common.money import parse_money\n\ndef total(x):\n    m = parse_money(x)\n"
        b"    return m.to_minor_units()\n\ndef helper():\n    return total(1)\n",
    )
    assert money and svc
    ledger.replace_file(money, 10)
    ledger.replace_file(svc, 10)
    resolver = SymbolResolver(ledger)
    edges = resolver.resolve(ledger.refs_from([s.symbol_id for s in svc.symbols]))
    by_target = {
        (
            next(s.qualified_name for s in svc.symbols if s.symbol_id == e.src_id),
            next(s.qualified_name for s in (*money.symbols, *svc.symbols) if s.symbol_id == e.dst_id),
        ): e
        for e in edges
    }
    imported = by_target[("checkout.service.total", "acme_common.money.parse_money")]
    assert imported.cross_repo and imported.confidence >= 0.9 and imported.kind is EdgeKind.CALLS
    local = by_target[("checkout.service.helper", "checkout.service.total")]
    assert not local.cross_repo
    # `m.to_minor_units()` has no type info: a unique global name still links, with low confidence
    method = by_target.get(("checkout.service.total", "acme_common.money.Money.to_minor_units"))
    assert method is not None and method.confidence < 0.5


def test_resolver_skips_ambiguous_names(tmp_path: Path, parser: TreeSitterParser) -> None:
    ledger = SqliteHashLedger(tmp_path / "l.db")
    for i in range(5):
        pf = parser.parse(f"acme/r{i}", "x.py", SHA, b"def process():\n    return 1\n")
        assert pf
        ledger.replace_file(pf, 2)
    caller = parser.parse("acme/caller", "c.py", SHA, b"def go():\n    return process()\n")
    assert caller
    ledger.replace_file(caller, 2)
    edges = SymbolResolver(ledger).resolve(ledger.refs_from([s.symbol_id for s in caller.symbols]))
    assert edges == [], "5 equally plausible cross-repo targets must not fan out"
