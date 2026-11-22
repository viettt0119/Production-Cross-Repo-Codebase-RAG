from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from code_rag.domain.entities import CodeChunk, SymbolNode
from code_rag.domain.query import SearchFilter
from code_rag.domain.value_objects import (
    Language,
    LineSpan,
    RepoCoordinate,
    SymbolKind,
    make_symbol_id,
    sha256_hex,
)
from code_rag.infrastructure.providers.offline import HashingEmbeddingProvider
from code_rag.infrastructure.stores.graph_store import EmbeddedGraphStore
from code_rag.infrastructure.stores.qdrant_index import QdrantVectorIndex
from code_rag.infrastructure.stores.tantivy_index import TantivyLexicalIndex

pytestmark = pytest.mark.integration


def _sym(name: str, source: str, generated: bool = False, repo: str = "acme/x") -> SymbolNode:
    return SymbolNode(
        symbol_id=make_symbol_id(repo, f"{name}.py", name, SymbolKind.FUNCTION),
        coord=RepoCoordinate(repo=repo, path=f"{name}.py", commit_sha="d" * 40),
        language=Language.PYTHON,
        kind=SymbolKind.FUNCTION,
        name=name,
        qualified_name=f"m.{name}",
        span=LineSpan(start=1, end=3),
        signature=f"def {name}()",
        source=source,
        content_hash=sha256_hex(source),
        is_generated=generated,
    )


async def test_qdrant_upsert_is_idempotent_and_penalizes_generated() -> None:
    emb = HashingEmbeddingProvider(128)
    index = QdrantVectorIndex(
        AsyncQdrantClient(location=":memory:"),
        emb.model_name,
        emb.dimension,
        on_disk=False,
        quantize=False,
        payload_indexes=False,
    )
    await index.ensure_ready()
    real = _sym("charge_card", "def charge_card(): stripe charge card amount")
    gen = _sym("charge_card_pb", "def charge_card_pb(): stripe charge card amount", generated=True)
    chunks = [CodeChunk(symbol=s, summary=None, embedding_model=emb.model_name) for s in (real, gen)]
    vecs = await emb.embed([s.source for s in (real, gen)], "document")
    await index.upsert(chunks, vecs, [None, None])
    await index.upsert(chunks, vecs, [None, None])
    assert await index.count() == 2
    qv = (await emb.embed(["stripe charge card amount"], "query"))[0]
    ranked = await index.search(qv, "code", SearchFilter(), 5)
    assert ranked[0].symbol_id == real.symbol_id
    with_gen = await index.search(qv, "code", SearchFilter(include_generated=True), 5)
    assert {h.symbol_id for h in with_gen} == {real.symbol_id, gen.symbol_id}
    moved = real.model_copy(update={"span": LineSpan(start=10, end=12)})
    await index.update_locations([moved])
    assert (await index.fetch([real.symbol_id]))[real.symbol_id].symbol.span.start == 10
    await index.delete([gen.symbol_id])
    assert await index.count() == 1


def test_tantivy_bm25f_field_boosts_and_filters(tmp_path: Path) -> None:
    lex = TantivyLexicalIndex(tmp_path / "t")
    named = _sym("refundCharge", "def refundCharge(): return 1")
    mentioned = _sym("helper", "def helper(): # calls refund charge refund charge")
    other_repo = _sym("refundCharge", "def refundCharge(): pass", repo="acme/y")
    lex.upsert([CodeChunk(symbol=s, embedding_model="m") for s in (named, mentioned, other_repo)])
    lex.commit()
    hits = lex.search("refund charge", SearchFilter(repos=("acme/x",)), 5)
    assert [h.symbol_id for h in hits][:2] == [named.symbol_id, mentioned.symbol_id]
    lex.upsert([CodeChunk(symbol=named, embedding_model="m")])
    lex.commit()
    assert lex.num_docs() == 3, "upsert by symbol_id must not duplicate"
    lex.delete([named.symbol_id])
    lex.commit()
    assert named.symbol_id not in {h.symbol_id for h in lex.search("refund", SearchFilter(), 5)}
    lex.close()


def test_graph_snapshot_publication_for_readonly_consumers(tmp_path: Path) -> None:
    writer = EmbeddedGraphStore(tmp_path / "g")
    a, b = _sym("a", "x"), _sym("b", "y")
    writer.upsert_symbols([a, b])
    from code_rag.domain.entities import SymbolEdge
    from code_rag.domain.value_objects import EdgeKind

    writer.replace_outgoing_edges(
        [a.symbol_id],
        [
            SymbolEdge(
                src_id=a.symbol_id,
                dst_id=b.symbol_id,
                kind=EdgeKind.CALLS,
                cross_repo=False,
                confidence=0.9,
                at_line=2,
            )
        ],
    )
    reader = EmbeddedGraphStore(tmp_path / "g", readonly=True)
    assert reader.neighbors([a.symbol_id], 5) == [], "nothing published yet"
    writer.publish()
    assert [n.neighbor_id for n in reader.neighbors([a.symbol_id], 5)] == [b.symbol_id]
    writer.replace_outgoing_edges([a.symbol_id], [])
    writer.publish()
    assert reader.neighbors([a.symbol_id], 5) == [], "reader follows the new snapshot"
    writer.close()
    reader.close()
