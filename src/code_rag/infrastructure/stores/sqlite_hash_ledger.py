"""SQLite-backed HashLedger: per-symbol content hashes, file records, unresolved refs, ingest runs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from code_rag.domain.changeset import LedgerEntry
from code_rag.domain.entities import ParsedFile, SymbolRef
from code_rag.domain.ports.indexing import FileRecord
from code_rag.domain.value_objects import EdgeKind, SymbolId

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS files (
    repo TEXT NOT NULL, path TEXT NOT NULL, language TEXT NOT NULL, module_name TEXT NOT NULL,
    blob_sha TEXT NOT NULL, line_count INTEGER NOT NULL, imports_json TEXT NOT NULL,
    commit_sha TEXT NOT NULL, PRIMARY KEY (repo, path)
);
CREATE TABLE IF NOT EXISTS symbols (
    symbol_id TEXT PRIMARY KEY, repo TEXT NOT NULL, path TEXT NOT NULL, name TEXT NOT NULL,
    qualified_name TEXT NOT NULL, kind TEXT NOT NULL, content_hash TEXT NOT NULL,
    start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, commit_sha TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_symbols_file ON symbols(repo, path);
CREATE INDEX IF NOT EXISTS ix_symbols_name ON symbols(name);
CREATE TABLE IF NOT EXISTS refs (
    src_id TEXT NOT NULL, repo TEXT NOT NULL, path TEXT NOT NULL, kind TEXT NOT NULL,
    target_name TEXT NOT NULL, target_last TEXT NOT NULL, at_line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_refs_src ON refs(src_id);
CREATE INDEX IF NOT EXISTS ix_refs_file ON refs(repo, path);
CREATE INDEX IF NOT EXISTS ix_refs_target ON refs(target_last);
CREATE TABLE IF NOT EXISTS repos (
    repo TEXT PRIMARY KEY, indexed_sha TEXT, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id TEXT PRIMARY KEY, repo TEXT NOT NULL, from_sha TEXT, to_sha TEXT NOT NULL,
    status TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL, stats_json TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS pending_batches (
    batch_id TEXT PRIMARY KEY, repo TEXT NOT NULL, created_at REAL NOT NULL, status TEXT NOT NULL
);
"""


def _entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        symbol_id=SymbolId(row["symbol_id"]),
        repo=row["repo"],
        path=row["path"],
        name=row["name"],
        qualified_name=row["qualified_name"],
        kind=row["kind"],
        content_hash=row["content_hash"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        commit_sha=row["commit_sha"],
    )


def _ref(row: sqlite3.Row) -> SymbolRef:
    return SymbolRef(
        src_id=SymbolId(row["src_id"]),
        src_repo=row["repo"],
        src_path=row["path"],
        kind=EdgeKind(row["kind"]),
        target_name=row["target_name"],
        at_line=row["at_line"],
    )


class SqliteHashLedger:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    # ---- files & symbols ---------------------------------------------------
    def file_entries(self, repo: str, path: str) -> list[LedgerEntry]:
        rows = self._query("SELECT * FROM symbols WHERE repo=? AND path=?", (repo, path))
        return [_entry(r) for r in rows]

    def get_file(self, repo: str, path: str) -> FileRecord | None:
        rows = self._query("SELECT * FROM files WHERE repo=? AND path=?", (repo, path))
        if not rows:
            return None
        r = rows[0]
        return FileRecord(
            repo=r["repo"],
            path=r["path"],
            language=r["language"],
            module_name=r["module_name"],
            blob_sha=r["blob_sha"],
            line_count=r["line_count"],
            imports=tuple(json.loads(r["imports_json"])),
            commit_sha=r["commit_sha"],
        )

    def line_count(self, repo: str, path: str) -> int | None:
        rows = self._query("SELECT line_count FROM files WHERE repo=? AND path=?", (repo, path))
        return int(rows[0]["line_count"]) if rows else None

    def files_in_repo(self, repo: str) -> list[str]:
        return [r["path"] for r in self._query("SELECT path FROM files WHERE repo=?", (repo,))]

    def replace_file(self, parsed: ParsedFile, line_count: int) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO files(repo,path,language,module_name,blob_sha,line_count,imports_json,commit_sha) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(repo,path) DO UPDATE SET language=excluded.language, "
                "module_name=excluded.module_name, blob_sha=excluded.blob_sha, line_count=excluded.line_count, "
                "imports_json=excluded.imports_json, commit_sha=excluded.commit_sha",
                (
                    parsed.repo,
                    parsed.path,
                    parsed.language.value,
                    parsed.module_name,
                    parsed.blob_sha,
                    line_count,
                    json.dumps(list(parsed.imports)),
                    parsed.commit_sha,
                ),
            )
            c.execute("DELETE FROM symbols WHERE repo=? AND path=?", (parsed.repo, parsed.path))
            c.executemany(
                "INSERT OR REPLACE INTO symbols(symbol_id,repo,path,name,qualified_name,kind,content_hash,"
                "start_line,end_line,commit_sha) VALUES(?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        s.symbol_id,
                        s.repo,
                        s.path,
                        s.name,
                        s.qualified_name,
                        s.kind.value,
                        s.content_hash,
                        s.span.start,
                        s.span.end,
                        parsed.commit_sha,
                    )
                    for s in parsed.symbols
                ],
            )
            c.execute("DELETE FROM refs WHERE repo=? AND path=?", (parsed.repo, parsed.path))
            c.executemany(
                "INSERT INTO refs(src_id,repo,path,kind,target_name,target_last,at_line) VALUES(?,?,?,?,?,?,?)",
                [
                    (
                        r.src_id,
                        r.src_repo,
                        r.src_path,
                        r.kind.value,
                        r.target_name,
                        r.target_name.rsplit(".", 1)[-1],
                        r.at_line,
                    )
                    for r in parsed.refs
                ],
            )

    def delete_file(self, repo: str, path: str) -> list[SymbolId]:
        with self._tx() as c:
            ids = [
                SymbolId(r["symbol_id"])
                for r in c.execute("SELECT symbol_id FROM symbols WHERE repo=? AND path=?", (repo, path)).fetchall()
            ]
            c.execute("DELETE FROM symbols WHERE repo=? AND path=?", (repo, path))
            c.execute("DELETE FROM refs WHERE repo=? AND path=?", (repo, path))
            c.execute("DELETE FROM files WHERE repo=? AND path=?", (repo, path))
        return ids

    def lookup_by_name(self, name: str) -> list[LedgerEntry]:
        return [_entry(r) for r in self._query("SELECT * FROM symbols WHERE name=?", (name,))]

    def symbol_entries(self, ids: Sequence[SymbolId]) -> dict[SymbolId, LedgerEntry]:
        out: dict[SymbolId, LedgerEntry] = {}
        for i in range(0, len(ids), 500):
            chunk = list(ids[i : i + 500])
            marks = ",".join("?" * len(chunk))
            for r in self._query(f"SELECT * FROM symbols WHERE symbol_id IN ({marks})", chunk):
                out[SymbolId(r["symbol_id"])] = _entry(r)
        return out

    def all_symbols(self, repo: str | None = None) -> list[LedgerEntry]:
        if repo:
            return [_entry(r) for r in self._query("SELECT * FROM symbols WHERE repo=?", (repo,))]
        return [_entry(r) for r in self._query("SELECT * FROM symbols")]

    # ---- refs ----------------------------------------------------------------
    def refs_from(self, src_ids: Sequence[SymbolId]) -> list[SymbolRef]:
        out: list[SymbolRef] = []
        for i in range(0, len(src_ids), 500):
            chunk = list(src_ids[i : i + 500])
            marks = ",".join("?" * len(chunk))
            out.extend(_ref(r) for r in self._query(f"SELECT * FROM refs WHERE src_id IN ({marks})", chunk))
        return out

    def refs_targeting(self, names: Sequence[str]) -> list[SymbolRef]:
        out: list[SymbolRef] = []
        for i in range(0, len(names), 500):
            chunk = list(names[i : i + 500])
            marks = ",".join("?" * len(chunk))
            out.extend(_ref(r) for r in self._query(f"SELECT * FROM refs WHERE target_last IN ({marks})", chunk))
        return out

    def all_src_ids_with_refs(self, repo: str | None = None) -> list[SymbolId]:
        if repo:
            rows = self._query("SELECT DISTINCT src_id FROM refs WHERE repo=?", (repo,))
        else:
            rows = self._query("SELECT DISTINCT src_id FROM refs")
        return [SymbolId(r["src_id"]) for r in rows]

    # ---- repos & runs ----------------------------------------------------------
    def indexed_sha(self, repo: str) -> str | None:
        rows = self._query("SELECT indexed_sha FROM repos WHERE repo=?", (repo,))
        return rows[0]["indexed_sha"] if rows else None

    def set_indexed_sha(self, repo: str, sha: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO repos(repo,indexed_sha,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(repo) DO UPDATE SET indexed_sha=excluded.indexed_sha, updated_at=excluded.updated_at",
                (repo, sha, time.time()),
            )

    def repos(self) -> list[str]:
        return [r["repo"] for r in self._query("SELECT repo FROM repos ORDER BY repo")]

    def start_run(self, repo: str, from_sha: str | None, to_sha: str) -> str:
        run_id = uuid.uuid4().hex
        with self._tx() as c:
            c.execute(
                "INSERT INTO ingest_runs(run_id,repo,from_sha,to_sha,status,started_at) VALUES(?,?,?,?,?,?)",
                (run_id, repo, from_sha, to_sha, "running", time.time()),
            )
        return run_id

    def finish_run(self, run_id: str, status: str, stats_json: str, error: str | None = None) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE ingest_runs SET status=?, finished_at=?, stats_json=?, error=? WHERE run_id=?",
                (status, time.time(), stats_json, error, run_id),
            )

    def recent_runs(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self._query("SELECT * FROM ingest_runs ORDER BY started_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ---- async summary batches ----------------------------------------------------
    def add_pending_batch(self, batch_id: str, repo: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO pending_batches(batch_id,repo,created_at,status) VALUES(?,?,?,?)",
                (batch_id, repo, time.time(), "pending"),
            )

    def pending_batches(self) -> list[tuple[str, str]]:
        rows = self._query("SELECT batch_id, repo FROM pending_batches WHERE status='pending' ORDER BY created_at")
        return [(r["batch_id"], r["repo"]) for r in rows]

    def complete_batch(self, batch_id: str, status: str = "done") -> None:
        with self._tx() as c:
            c.execute("UPDATE pending_batches SET status=? WHERE batch_id=?", (status, batch_id))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
