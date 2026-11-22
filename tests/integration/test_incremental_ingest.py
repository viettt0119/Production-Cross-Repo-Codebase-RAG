from __future__ import annotations

import pytest

from code_rag.domain.query import SearchFilter
from code_rag.infrastructure.container import Container
from tests.conftest import CorpusRepos

pytestmark = pytest.mark.integration

CHECKOUT = "acme/checkout-py"
BILLING = "acme/billing-go"


async def test_full_index_builds_all_stores(indexed: Container, corpus: CorpusRepos) -> None:
    c = indexed
    ledger_syms = c.ledger.all_symbols()
    assert len(ledger_syms) > 60
    assert await c.vectors.count() == len(ledger_syms)
    assert c.lexical.num_docs() == len(ledger_syms)
    stats = c.graph.stats()
    assert stats["Symbol"] == len(ledger_syms)
    assert stats["SERVES_RPC"] == 2 and stats["CALLS_RPC"] == 3 and stats["DECLARES_RPC"] == 2
    for repo in corpus.paths:
        assert c.ledger.indexed_sha(repo) is not None


async def test_rerun_without_new_commits_is_a_noop(indexed: Container) -> None:
    stats = await indexed.ingest_workflow.run(CHECKOUT)
    assert stats.symbols_embedded == 0 and stats.files_changed == 0


async def test_line_shift_relocates_without_reembedding(indexed: Container, corpus: CorpusRepos) -> None:
    path = "checkout/inventory.py"
    corpus.write(CHECKOUT, path, "# header comment\n\n" + corpus.read(CHECKOUT, path))
    corpus.commit(CHECKOUT)
    stats = await indexed.ingest_workflow.run(CHECKOUT)
    assert stats.symbols_embedded == 0
    assert stats.symbols_relocated >= 4
    entries = {e.qualified_name: e for e in indexed.ledger.file_entries(CHECKOUT, path)}
    reserve = entries["checkout.inventory.Inventory.reserve"]
    assert reserve.start_line == 14  # moved down by 2 lines
    chunk = (await indexed.vectors.fetch([reserve.symbol_id]))[reserve.symbol_id]
    assert chunk.symbol.span.start == 14, "payload must carry the new location for citations"


async def test_body_change_reembeds_only_that_symbol(indexed: Container, corpus: CorpusRepos) -> None:
    path = "checkout/inventory.py"
    src = corpus.read(CHECKOUT, path).replace(
        "self._stock[sku] = available - qty", "self._stock[sku] = max(0, available - qty)"
    )
    corpus.write(CHECKOUT, path, src)
    corpus.commit(CHECKOUT)
    stats = await indexed.ingest_workflow.run(CHECKOUT)
    assert stats.symbols_embedded == 1
    assert stats.symbols_deleted == 0


async def test_delete_and_rename_files(indexed: Container, corpus: CorpusRepos) -> None:
    repo_path = corpus.paths[CHECKOUT]
    old_ids = {e.symbol_id for e in indexed.ledger.file_entries(CHECKOUT, "checkout/api.py")}
    (repo_path / "checkout" / "api.py").unlink()
    (repo_path / "checkout" / "inventory.py").rename(repo_path / "checkout" / "stock.py")
    corpus.commit(CHECKOUT, "delete + rename")
    stats = await indexed.ingest_workflow.run(CHECKOUT)
    assert stats.symbols_deleted >= len(old_ids) + 4
    assert indexed.ledger.file_entries(CHECKOUT, "checkout/api.py") == []
    assert indexed.ledger.file_entries(CHECKOUT, "checkout/inventory.py") == []
    assert {e.qualified_name for e in indexed.ledger.file_entries(CHECKOUT, "checkout/stock.py")} >= {
        "checkout.stock.Inventory.reserve"
    }
    assert not await indexed.vectors.fetch(list(old_ids))
    flt = SearchFilter(repos=(CHECKOUT,))
    hits = indexed.lexical.search("create order handler", flt, 10)
    assert all(h.symbol_id not in old_ids for h in hits)


async def test_cross_repo_edges_heal_when_target_appears(container: Container, corpus: CorpusRepos) -> None:
    """Index the caller before its dependency: edges must appear once the dependency is indexed."""
    c = container
    await c.ingest_workflow.run(CHECKOUT)
    svc = {e.qualified_name: e.symbol_id for e in c.ledger.file_entries(CHECKOUT, "checkout/service.py")}
    order_total = svc["checkout.service.CheckoutService.order_total"]
    before = c.graph.neighbors([order_total], limit=20, min_confidence=0.5)
    assert not any(n.cross_repo for n in before)
    await c.ingest_workflow.run("acme/shared-py")
    after = c.graph.neighbors([order_total], limit=20, min_confidence=0.5)
    money = {e.symbol_id: e.qualified_name for e in c.ledger.all_symbols("acme/shared-py")}
    linked = {money[n.neighbor_id] for n in after if n.neighbor_id in money}
    assert "acme_common.money.parse_money" in linked


async def test_rpc_bridge_links_python_client_to_go_server(indexed: Container) -> None:
    client = {e.qualified_name: e.symbol_id for e in indexed.ledger.all_symbols(CHECKOUT)}
    charge = client["checkout.billing_client.BillingClient.charge"]
    neighbors = indexed.graph.neighbors([charge], limit=20)
    go = {e.symbol_id: e.qualified_name for e in indexed.ledger.all_symbols(BILLING)}
    rpc = [n for n in neighbors if n.relation.startswith("RPC:")]
    assert any(go.get(n.neighbor_id) == "internal.server.BillingServer.Charge" for n in rpc)
    proto = {e.symbol_id: e.qualified_name for e in indexed.ledger.all_symbols("acme/proto")}
    assert any(proto.get(n.neighbor_id) == "billing.v1.BillingService.Charge" for n in rpc)


async def test_go_server_change_keeps_rpc_links(indexed: Container, corpus: CorpusRepos) -> None:
    path = "internal/server/server.go"
    corpus.write(BILLING, path, corpus.read(BILLING, path).replace('Status: "captured"', 'Status: "settled"'))
    corpus.commit(BILLING)
    stats = await indexed.ingest_workflow.run(BILLING)
    assert stats.symbols_embedded == 1
    assert indexed.graph.stats()["SERVES_RPC"] == 2
