"""Embedded property-graph SymbolGraph adapter (LadybugDB — the maintained fork of Kùzu).

Kùzu was archived upstream in Oct 2025; LadybugDB keeps the same Cypher dialect and Python API.
The `kuzu` module is accepted as a drop-in fallback.

Concurrency model: embedded graph DBs allow one read-write process. The ingest worker owns the
live database and, after each ingest, `publish()` checkpoints it and atomically swaps a
`current -> snapshots/<ts>` symlink. API processes open the current snapshot read-only and
transparently reopen when the symlink moves.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from code_rag.domain.entities import RpcLink, SymbolEdge, SymbolNode
from code_rag.domain.errors import IndexStoreError
from code_rag.domain.ports.indexing import GraphNeighbor
from code_rag.domain.value_objects import EdgeKind, SymbolId

log = logging.getLogger(__name__)

DB_FILE = "graph.lbug"
_SYMBOL_EDGE_KINDS = (EdgeKind.CALLS, EdgeKind.INHERITS, EdgeKind.IMPLEMENTS)
_RPC_KINDS = (EdgeKind.CALLS_RPC, EdgeKind.SERVES_RPC, EdgeKind.DECLARES_RPC)

DDL = [
    "CREATE NODE TABLE IF NOT EXISTS Repo(name STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE IF NOT EXISTS File(id STRING, repo STRING, path STRING, language STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS Symbol(id STRING, repo STRING, path STRING, name STRING, "
    "qualified_name STRING, kind STRING, start_line INT64, end_line INT64, is_generated BOOLEAN, "
    "PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS RpcEndpoint(id STRING, service STRING, method STRING, PRIMARY KEY(id))",
    "CREATE REL TABLE IF NOT EXISTS IN_REPO(FROM File TO Repo)",
    "CREATE REL TABLE IF NOT EXISTS DEFINES(FROM File TO Symbol)",
    "CREATE REL TABLE IF NOT EXISTS MEMBER_OF(FROM Symbol TO Symbol)",
    *(
        f"CREATE REL TABLE IF NOT EXISTS {rel}(FROM Symbol TO Symbol, "
        "cross_repo BOOLEAN, confidence DOUBLE, at_line INT64)"
        for rel in ("CALLS", "INHERITS", "IMPLEMENTS")
    ),
    "CREATE REL TABLE IF NOT EXISTS CALLS_RPC(FROM Symbol TO RpcEndpoint, at_line INT64)",
    "CREATE REL TABLE IF NOT EXISTS SERVES_RPC(FROM Symbol TO RpcEndpoint, at_line INT64)",
    "CREATE REL TABLE IF NOT EXISTS DECLARES_RPC(FROM Symbol TO RpcEndpoint, at_line INT64)",
]


def _engine() -> ModuleType:
    for name in ("ladybug", "kuzu"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise IndexStoreError("no embedded graph engine installed (pip install ladybug)")


class EmbeddedGraphStore:
    def __init__(self, root: str | Path, readonly: bool = False, keep_snapshots: int = 3) -> None:
        self._root = Path(root)
        self._readonly = readonly
        self._keep = keep_snapshots
        self._engine = _engine()
        self._lock = threading.RLock()
        self._db: Any = None
        self._conn: Any = None
        self._opened_path: str | None = None
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "snapshots").mkdir(exist_ok=True)
        if not readonly:
            (self._root / "live").mkdir(exist_ok=True)
            conn = self._open(str(self._root / "live" / DB_FILE), read_only=False)
            for stmt in DDL:
                conn.execute(stmt)

    # ---- connection management ----------------------------------------------
    def _open(self, path: str, read_only: bool) -> Any:
        self._db = self._engine.Database(path, read_only=read_only)
        self._conn = self._engine.Connection(self._db)
        self._opened_path = path
        return self._conn

    def _current_snapshot(self) -> str | None:
        link = self._root / "current"
        if not link.exists():
            return None
        return os.path.realpath(link / DB_FILE)

    def _conn_for_read(self) -> Any:
        if not self._readonly:
            return self._conn
        target = self._current_snapshot()
        if target is None:
            return None
        if target != self._opened_path:
            self._conn = None
            self._db = None
            self._open(target, read_only=True)
        return self._conn

    def _rows(self, query: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        with self._lock:
            conn = self._conn_for_read()
            if conn is None:
                return []
            result = conn.execute(query, params or {})
            rows: list[list[Any]] = []
            while result.has_next():
                rows.append(list(result.get_next()))
            return rows

    def _exec(self, query: str, params: dict[str, Any] | None = None) -> None:
        if self._readonly:
            raise IndexStoreError("graph opened read-only")
        with self._lock:
            self._conn.execute(query, params or {})

    # ---- writes ------------------------------------------------------------------
    def upsert_symbols(self, symbols: Sequence[SymbolNode]) -> None:
        if not symbols:
            return
        files: dict[str, dict[str, str]] = {}
        rows: list[dict[str, Any]] = []
        for s in symbols:
            fid = f"{s.coord.repo}:{s.coord.path}"
            files[fid] = {"id": fid, "repo": s.coord.repo, "path": s.coord.path, "language": s.language.value}
            rows.append(
                {
                    "id": s.symbol_id,
                    "repo": s.coord.repo,
                    "path": s.coord.path,
                    "name": s.name,
                    "qn": s.qualified_name,
                    "kind": s.kind.value,
                    "sl": s.span.start,
                    "el": s.span.end,
                    "gen": s.is_generated,
                    "file": fid,
                    "parent": s.parent_id or "",
                }
            )
        self._exec(
            "UNWIND $rows AS r MERGE (repo:Repo {name: r.repo})",
            {"rows": [{"repo": f["repo"]} for f in files.values()]},
        )
        self._exec(
            "UNWIND $rows AS r MERGE (f:File {id: r.id}) SET f.repo = r.repo, f.path = r.path, f.language = r.language",
            {"rows": list(files.values())},
        )
        self._exec(
            "UNWIND $rows AS r MATCH (f:File {id: r.id}), (repo:Repo {name: r.repo}) MERGE (f)-[:IN_REPO]->(repo)",
            {"rows": list(files.values())},
        )
        self._exec(
            "UNWIND $rows AS r MERGE (s:Symbol {id: r.id}) SET s.repo = r.repo, s.path = r.path, s.name = r.name, "
            "s.qualified_name = r.qn, s.kind = r.kind, s.start_line = r.sl, s.end_line = r.el, "
            "s.is_generated = r.gen",
            {"rows": rows},
        )
        self._exec(
            "UNWIND $rows AS r MATCH (f:File {id: r.file}), (s:Symbol {id: r.id}) MERGE (f)-[:DEFINES]->(s)",
            {"rows": rows},
        )
        with_parent = [r for r in rows if r["parent"]]
        if with_parent:
            self._exec(
                "UNWIND $rows AS r MATCH (c:Symbol {id: r.id}), (p:Symbol {id: r.parent}) MERGE (c)-[:MEMBER_OF]->(p)",
                {"rows": with_parent},
            )

    def delete_symbols(self, ids: Sequence[SymbolId]) -> None:
        if ids:
            self._exec("MATCH (s:Symbol) WHERE s.id IN $ids DETACH DELETE s", {"ids": list(ids)})

    def replace_outgoing_edges(self, src_ids: Sequence[SymbolId], edges: Sequence[SymbolEdge]) -> None:
        if src_ids:
            for kind in _SYMBOL_EDGE_KINDS:
                self._exec(
                    f"MATCH (s:Symbol)-[r:{kind.value}]->(:Symbol) WHERE s.id IN $ids DELETE r",
                    {"ids": list(src_ids)},
                )
        for kind in _SYMBOL_EDGE_KINDS:
            rows = [
                {
                    "src": e.src_id,
                    "dst": e.dst_id,
                    "cross": e.cross_repo,
                    "conf": e.confidence,
                    "line": e.at_line,
                }
                for e in edges
                if e.kind is kind
            ]
            if rows:
                self._exec(
                    f"UNWIND $rows AS r MATCH (a:Symbol {{id: r.src}}), (b:Symbol {{id: r.dst}}) "
                    f"MERGE (a)-[e:{kind.value}]->(b) SET e.cross_repo = r.cross, e.confidence = r.conf, "
                    f"e.at_line = r.line",
                    {"rows": rows},
                )

    def replace_rpc_links(self, symbol_ids: Sequence[SymbolId], links: Sequence[RpcLink]) -> None:
        if symbol_ids:
            for kind in _RPC_KINDS:
                self._exec(
                    f"MATCH (s:Symbol)-[r:{kind.value}]->(:RpcEndpoint) WHERE s.id IN $ids DELETE r",
                    {"ids": list(symbol_ids)},
                )
        if not links:
            return
        endpoints = {
            link.endpoint.endpoint_id: {
                "id": link.endpoint.endpoint_id,
                "service": link.endpoint.service,
                "method": link.endpoint.method,
            }
            for link in links
        }
        self._exec(
            "UNWIND $rows AS r MERGE (e:RpcEndpoint {id: r.id}) SET e.service = r.service, e.method = r.method",
            {"rows": list(endpoints.values())},
        )
        for kind in _RPC_KINDS:
            rows = [
                {"sid": link.symbol_id, "eid": link.endpoint.endpoint_id, "line": link.at_line}
                for link in links
                if link.kind is kind
            ]
            if rows:
                self._exec(
                    f"UNWIND $rows AS r MATCH (s:Symbol {{id: r.sid}}), (e:RpcEndpoint {{id: r.eid}}) "
                    f"MERGE (s)-[x:{kind.value}]->(e) SET x.at_line = r.line",
                    {"rows": rows},
                )

    # ---- reads -----------------------------------------------------------------
    def neighbors(self, seed_ids: Sequence[SymbolId], limit: int, min_confidence: float = 0.5) -> list[GraphNeighbor]:
        if not seed_ids:
            return []
        ids = list(seed_ids)
        rpc = self._rows(
            "MATCH (s:Symbol)-[:CALLS_RPC|SERVES_RPC|DECLARES_RPC]->(e:RpcEndpoint)"
            "<-[:CALLS_RPC|SERVES_RPC|DECLARES_RPC]-(n:Symbol) "
            "WHERE s.id IN $ids AND n.id <> s.id AND n.is_generated = false "
            "RETURN s.id, n.id, e.id, s.repo <> n.repo LIMIT $lim",
            {"ids": ids, "lim": limit * 4},
        )
        direct = self._rows(
            "MATCH (s:Symbol)-[r:CALLS|INHERITS|IMPLEMENTS]-(n:Symbol) "
            "WHERE s.id IN $ids AND n.is_generated = false AND r.confidence >= $c "
            "RETURN s.id, n.id, label(r), r.cross_repo, r.confidence ORDER BY r.cross_repo DESC, r.confidence DESC "
            "LIMIT $lim",
            {"ids": ids, "c": min_confidence, "lim": limit * 4},
        )
        parents = self._rows(
            "MATCH (s:Symbol)-[:MEMBER_OF]->(p:Symbol) WHERE s.id IN $ids RETURN s.id, p.id",
            {"ids": ids},
        )
        seeds = set(ids)
        seen: set[str] = set()
        out: list[GraphNeighbor] = []

        def add(seed: str, nid: str, relation: str, cross: bool) -> None:
            if nid in seeds or nid in seen or len(out) >= limit:
                return
            seen.add(nid)
            out.append(GraphNeighbor(SymbolId(nid), SymbolId(seed), relation, bool(cross)))

        for seed, nid, eid, cross in rpc:
            add(seed, nid, f"RPC:{eid}", cross)
        for seed, nid, rel, cross, _conf in direct:
            if cross:
                add(seed, nid, str(rel), True)
        for seed, nid, rel, cross, _conf in direct:
            add(seed, nid, str(rel), bool(cross))
        for seed, pid in parents:
            add(seed, pid, "MEMBER_OF", False)
        return out

    def cross_repo_pairs(self, limit: int) -> list[tuple[SymbolId, SymbolId, str]]:
        rows = self._rows(
            "MATCH (a:Symbol)-[r:CALLS|INHERITS|IMPLEMENTS]->(b:Symbol) WHERE r.cross_repo = true "
            "AND a.is_generated = false AND b.is_generated = false RETURN a.id, b.id, label(r) LIMIT $n",
            {"n": limit},
        )
        rows += self._rows(
            "MATCH (a:Symbol)-[:CALLS_RPC]->(e:RpcEndpoint)<-[:SERVES_RPC]-(b:Symbol) "
            "RETURN a.id, b.id, 'RPC:' + e.id LIMIT $n",
            {"n": limit},
        )
        return [(SymbolId(a), SymbolId(b), str(r)) for a, b, r in rows]

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for label in ("Symbol", "File", "RpcEndpoint"):
            rows = self._rows(f"MATCH (n:{label}) RETURN count(*)")
            out[label] = int(rows[0][0]) if rows else 0
        for kind in (*_SYMBOL_EDGE_KINDS, *_RPC_KINDS):
            rows = self._rows(f"MATCH ()-[r:{kind.value}]->() RETURN count(*)")
            out[kind.value] = int(rows[0][0]) if rows else 0
        return out

    # ---- snapshot publication ----------------------------------------------------
    def publish(self) -> None:
        if self._readonly:
            return
        with self._lock:
            self._conn.execute("CHECKPOINT")
            ts = time.strftime("%Y%m%dT%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
            snap_dir = self._root / "snapshots" / ts
            snap_dir.mkdir(parents=True)
            live = self._root / "live"
            for item in live.iterdir():
                if item.name.startswith(DB_FILE):
                    shutil.copy2(item, snap_dir / item.name)
            tmp_link = self._root / f".current.{ts}"
            tmp_link.symlink_to(snap_dir.relative_to(self._root), target_is_directory=True)
            os.replace(tmp_link, self._root / "current")
        self._prune()

    def _prune(self) -> None:
        snaps = sorted((self._root / "snapshots").iterdir())
        current = os.path.realpath(self._root / "current")
        for old in snaps[: -self._keep]:
            if os.path.realpath(old) != current:
                shutil.rmtree(old, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            self._conn = None
            self._db = None
