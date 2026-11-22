"""Tantivy LexicalIndex adapter: field-weighted BM25F over symbol names, summaries and bodies.

Identifiers are expanded in Python (`getUserById` -> `getuserbyid get user by id`) before
indexing and querying, so the Rust-side analyzer can stay a simple lowercase tokenizer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path

import tantivy

from code_rag.domain.entities import CodeChunk
from code_rag.domain.errors import IndexStoreError
from code_rag.domain.ports.indexing import ScoredId
from code_rag.domain.query import SearchFilter
from code_rag.domain.value_objects import SymbolId
from code_rag.infrastructure.text import code_tokens, expand_identifiers

DEFAULT_BOOSTS: dict[str, float] = {"symbol_name": 4.0, "summary": 2.0, "body": 1.0}
MAX_BODY_CHARS = 20_000


def _schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("symbol_id", stored=True, tokenizer_name="raw")
    sb.add_text_field("symbol_name", stored=False, tokenizer_name="code")
    sb.add_text_field("summary", stored=False, tokenizer_name="en_stem")
    sb.add_text_field("body", stored=False, tokenizer_name="code")
    sb.add_text_field("repo", stored=True, tokenizer_name="raw")
    sb.add_text_field("language", stored=True, tokenizer_name="raw")
    sb.add_boolean_field("is_generated", stored=True, indexed=True)
    sb.add_boolean_field("is_test", stored=True, indexed=True)
    return sb.build()


def _code_analyzer() -> tantivy.TextAnalyzer:
    return (
        tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
        .filter(tantivy.Filter.remove_long(64))
        .filter(tantivy.Filter.lowercase())
        .build()
    )


class TantivyLexicalIndex:
    def __init__(
        self,
        path: str | Path,
        readonly: bool = False,
        boosts: dict[str, float] | None = None,
        writer_heap_bytes: int = 256_000_000,
        reload_interval_s: float = 2.0,
    ) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._schema = _schema()
        exists = tantivy.Index.exists(str(self._path))
        self._index = tantivy.Index(self._schema, path=str(self._path), reuse=exists)
        self._index.register_tokenizer("code", _code_analyzer())
        self._readonly = readonly
        self._boosts = boosts or DEFAULT_BOOSTS
        self._heap = writer_heap_bytes
        self._writer: tantivy.IndexWriter | None = None
        self._lock = threading.Lock()
        self._reload_interval = reload_interval_s
        self._last_reload = 0.0
        self._dirty = False

    def _w(self) -> tantivy.IndexWriter:
        if self._readonly:
            raise IndexStoreError("lexical index opened read-only")
        if self._writer is None:
            self._writer = self._index.writer(heap_size=self._heap, num_threads=0)
        return self._writer

    def upsert(self, chunks: Sequence[CodeChunk]) -> None:
        with self._lock:
            w = self._w()
            for c in chunks:
                s = c.symbol
                w.delete_documents_by_term("symbol_id", s.symbol_id)
                w.add_document(
                    tantivy.Document(
                        symbol_id=s.symbol_id,
                        symbol_name=expand_identifiers(f"{s.name} {s.qualified_name}"),
                        summary=c.summary or "",
                        body=expand_identifiers(f"{s.coord.path} {s.signature}\n{s.source[:MAX_BODY_CHARS]}"),
                        repo=s.coord.repo,
                        language=s.language.value,
                        is_generated=s.is_generated,
                        is_test=s.is_test,
                    )
                )
            self._dirty = True

    def delete(self, ids: Sequence[SymbolId]) -> None:
        if not ids:
            return
        with self._lock:
            w = self._w()
            for sid in ids:
                w.delete_documents_by_term("symbol_id", sid)
            self._dirty = True

    def commit(self) -> None:
        with self._lock:
            if self._writer is not None and self._dirty:
                self._writer.commit()
                self._dirty = False
            self._index.reload()
            self._last_reload = time.monotonic()

    def _maybe_reload(self) -> None:
        now = time.monotonic()
        if now - self._last_reload >= self._reload_interval:
            self._index.reload()
            self._last_reload = now

    def search(self, text: str, flt: SearchFilter, limit: int) -> list[ScoredId]:
        terms = list(dict.fromkeys(code_tokens(text)))
        if not terms:
            return []
        self._maybe_reload()
        query_text = " ".join(terms)
        text_query, _errors = self._index.parse_query_lenient(query_text, list(self._boosts), field_boosts=self._boosts)
        clauses: list[tuple[tantivy.Occur, tantivy.Query]] = [(tantivy.Occur.Must, text_query)]
        if flt.repos:
            clauses.append((tantivy.Occur.Must, tantivy.Query.term_set_query(self._schema, "repo", list(flt.repos))))
        if flt.languages:
            clauses.append(
                (
                    tantivy.Occur.Must,
                    tantivy.Query.term_set_query(self._schema, "language", list(flt.languages)),
                )
            )
        if not flt.include_tests:
            clauses.append((tantivy.Occur.Must, tantivy.Query.term_query(self._schema, "is_test", False)))
        query = tantivy.Query.boolean_query(clauses)
        penalty = 0.0 if flt.include_generated else flt.generated_penalty
        searcher = self._index.searcher()
        result = searcher.search(query, limit * 2 if penalty else limit)
        scored: list[ScoredId] = []
        for score, addr in result.hits:
            doc = searcher.doc(addr)
            sid = doc["symbol_id"][0]
            gen = bool(doc["is_generated"][0]) if doc["is_generated"] else False
            scored.append(ScoredId(SymbolId(sid), float(score) * ((1 - penalty) if gen else 1.0)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    def num_docs(self) -> int:
        self._index.reload()
        return int(self._index.searcher().num_docs)

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                if self._dirty:
                    self._writer.commit()
                self._writer.wait_merging_threads()
                self._writer = None
