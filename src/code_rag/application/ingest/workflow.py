"""Incremental (and full) ingestion pipeline.

    git diff -> read blobs -> AST parse (process pool) -> symbol hash diff
      -> [summaries (per file, cached prompt)  ||  code embeddings]
      -> summary embeddings
      -> write: Qdrant, Tantivy, graph nodes/RPC links   (stores first)
      -> write: hash ledger                              (source of truth last)
      -> resolve refs (outgoing + incoming) -> graph edges -> publish snapshot -> mark indexed sha

Ordering makes every step idempotent: a crash before `set_indexed_sha` simply re-runs the same
diff; stores are upserted by deterministic ids and the ledger is only advanced after them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from code_rag.application.ingest.symbol_differ import diff_symbols
from code_rag.application.ingest.symbol_resolver import SymbolResolver
from code_rag.domain.changeset import ChangeSet, ChangeType, IngestStats, LedgerEntry, SymbolDelta
from code_rag.domain.entities import CodeChunk, ParsedFile, SymbolNode
from code_rag.domain.errors import ProviderError
from code_rag.domain.ports.indexing import HashLedger, LexicalIndex, SymbolGraph, VectorIndex
from code_rag.domain.ports.models import BatchSummarizer, EmbeddingProvider, FileToSummarize, SymbolSummarizer
from code_rag.domain.ports.observability import Tracer
from code_rag.domain.ports.parsing import BulkCodeParser
from code_rag.domain.ports.source import SourceRepository
from code_rag.domain.value_objects import SymbolId

log = logging.getLogger(__name__)

T = TypeVar("T")
SummaryMode = Literal["realtime", "batch", "none", "auto"]
MAX_EMBED_SOURCE_CHARS = 16_000
BLOB_READ_BATCH = 500


def code_embedding_text(sym: SymbolNode) -> str:
    return (
        f"// {sym.coord.repo}/{sym.coord.path}\n// {sym.kind.value} {sym.qualified_name}\n"
        f"{sym.source[:MAX_EMBED_SOURCE_CHARS]}"
    )


def summary_embedding_text(sym: SymbolNode, summary: str) -> str:
    return f"{sym.qualified_name} ({sym.kind.value} in {sym.coord.repo}): {summary}"


@dataclass
class IngestDependencies:
    source: SourceRepository
    parser: BulkCodeParser
    ledger: HashLedger
    vectors: VectorIndex
    lexical: LexicalIndex
    graph: SymbolGraph
    code_embedder: EmbeddingProvider
    summary_embedder: EmbeddingProvider
    summarizer: SymbolSummarizer | None
    batch_summarizer: BatchSummarizer | None
    tracer: Tracer
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class IngestOptions:
    summary_mode: SummaryMode = "auto"
    full_batch_threshold: int = 2_000  # symbols; above this, a full index uses the Batches API
    files_per_slice: int = 400
    exclude_tests_from_summary: bool = False


@dataclass
class _Plan:
    to_sha: str
    from_sha: str | None
    full: bool
    parse_paths: list[str]
    delete_paths: list[str]
    files_changed: int


@dataclass
class _SliceResult:
    embedded: int = 0
    relocated: int = 0
    deleted: int = 0
    unchanged: int = 0
    summaries_failed: int = 0
    files_parsed: int = 0
    touched_src_ids: set[SymbolId] = field(default_factory=set)
    changed_names: set[str] = field(default_factory=set)
    pending_summary: list[FileToSummarize] = field(default_factory=list)


class IncrementalIngestWorkflow:
    def __init__(self, deps: IngestDependencies, options: IngestOptions | None = None) -> None:
        self._d = deps
        self._o = options or IngestOptions()
        self._resolver = SymbolResolver(deps.ledger)

    # ------------------------------------------------------------------ public
    async def run(
        self, repo: str, to_sha: str | None = None, from_sha: str | None = None, force_full: bool = False
    ) -> IngestStats:
        d = self._d
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        with d.tracer.span("ingest", input={"repo": repo, "to": to_sha, "from": from_sha}, as_type="chain") as root:
            await self._timed(timings, "sync", d.source.sync(repo))
            target = to_sha or await d.source.head(repo)
            plan = await self._timed(timings, "plan", self._plan(repo, target, from_sha, force_full))
            if plan is None:
                return self._stats(repo, from_sha, target, timings, t0, _SliceResult(), 0, 0)
            run_id = d.ledger.start_run(repo, plan.from_sha, plan.to_sha)
            try:
                await d.vectors.ensure_ready()
                result = await self._execute(repo, plan, timings)
                resolved = await self._timed(timings, "resolve", self._resolve(result))
                with d.tracer.span("publish"):
                    t = time.perf_counter()
                    d.graph.publish()
                    timings["publish"] = (time.perf_counter() - t) * 1000
                d.ledger.set_indexed_sha(repo, plan.to_sha)
                if result.pending_summary:
                    await self._submit_batch(repo, result.pending_summary)
                stats = self._stats(repo, plan.from_sha, plan.to_sha, timings, t0, result, plan.files_changed, resolved)
                d.ledger.finish_run(run_id, "succeeded", stats.model_dump_json())
                root.set_output(stats.model_dump())
                return stats
            except Exception as exc:
                d.ledger.finish_run(
                    run_id, "failed", json.dumps({"timings_ms": timings}), f"{type(exc).__name__}: {exc}"
                )
                raise
            finally:
                d.tracer.flush()

    # ------------------------------------------------------------------ planning
    async def _plan(self, repo: str, to_sha: str, from_sha: str | None, force_full: bool) -> _Plan | None:
        d = self._d
        base = from_sha if from_sha is not None else d.ledger.indexed_sha(repo)
        if base == to_sha and not force_full:
            log.info("%s already indexed at %s", repo, to_sha[:10])
            return None
        if base is not None and not force_full and not await d.source.is_ancestor(repo, base, to_sha):
            if await d.source.is_ancestor(repo, to_sha, base):
                log.info("%s: %s is older than indexed %s; skipping stale event", repo, to_sha[:10], base[:10])
                return None
            log.warning("%s: history rewritten (%s !<= %s); falling back to full reconcile", repo, base, to_sha)
            base = None
        if base is None or force_full:
            files = [p for p in await d.source.list_files(repo, to_sha) if d.parser.supports(p)]
            known = set(self._ledger_files(repo))
            present = set(files)
            return _Plan(
                to_sha=to_sha,
                from_sha=base,
                full=True,
                parse_paths=files,
                delete_paths=sorted(known - present),
                files_changed=len(files),
            )
        cs: ChangeSet = await d.source.diff(repo, base, to_sha)
        parse_paths: list[str] = []
        delete_paths: list[str] = []
        for fc in cs.files:
            if fc.change is ChangeType.DELETED:
                delete_paths.append(fc.path)
                continue
            if fc.change is ChangeType.RENAMED and fc.old_path:
                delete_paths.append(fc.old_path)
            if d.parser.supports(fc.path):
                parse_paths.append(fc.path)
            elif d.ledger.get_file(repo, fc.path) is not None:
                delete_paths.append(fc.path)
        return _Plan(to_sha, base, False, parse_paths, delete_paths, len(cs.files))

    def _ledger_files(self, repo: str) -> list[str]:
        return self._d.ledger.files_in_repo(repo)

    # ------------------------------------------------------------------ execution
    async def _execute(self, repo: str, plan: _Plan, timings: dict[str, float]) -> _SliceResult:
        total = _SliceResult()
        mode = self._summary_mode(plan)
        slices = [
            plan.parse_paths[i : i + self._o.files_per_slice]
            for i in range(0, len(plan.parse_paths), self._o.files_per_slice)
        ] or [[]]
        for idx, paths in enumerate(slices):
            deletes = plan.delete_paths if idx == 0 else []
            part = await self._process_slice(repo, plan.to_sha, paths, deletes, mode, timings)
            total.embedded += part.embedded
            total.relocated += part.relocated
            total.deleted += part.deleted
            total.unchanged += part.unchanged
            total.summaries_failed += part.summaries_failed
            total.files_parsed += part.files_parsed
            total.touched_src_ids |= part.touched_src_ids
            total.changed_names |= part.changed_names
            total.pending_summary.extend(part.pending_summary)
        return total

    def _summary_mode(self, plan: _Plan) -> Literal["realtime", "batch", "none"]:
        d, mode = self._d, self._o.summary_mode
        if mode == "auto":
            if d.summarizer is None and d.batch_summarizer is None:
                return "none"
            if (
                plan.full
                and d.batch_summarizer is not None
                and len(plan.parse_paths) * 8 > self._o.full_batch_threshold
            ):
                return "batch"
            return "realtime" if d.summarizer is not None else "batch"
        if mode == "realtime" and d.summarizer is None:
            return "none"
        if mode == "batch" and d.batch_summarizer is None:
            return "realtime" if d.summarizer is not None else "none"
        return mode

    async def _process_slice(
        self,
        repo: str,
        sha: str,
        paths: Sequence[str],
        delete_paths: Sequence[str],
        mode: Literal["realtime", "batch", "none"],
        timings: dict[str, float],
    ) -> _SliceResult:
        d = self._d
        res = _SliceResult()

        blobs: dict[str, bytes] = {}
        t = time.perf_counter()
        for i in range(0, len(paths), BLOB_READ_BATCH):
            blobs.update(await d.source.read_blobs(repo, sha, paths[i : i + BLOB_READ_BATCH]))
        timings["read_blobs"] = timings.get("read_blobs", 0) + (time.perf_counter() - t) * 1000

        with d.tracer.span("parse", input={"files": len(blobs)}):
            t = time.perf_counter()
            parsed, errors = await d.parser.parse_many(repo, sha, blobs)
            timings["parse"] = timings.get("parse", 0) + (time.perf_counter() - t) * 1000
        for path, err in errors.items():
            log.warning("parse failed for %s/%s: %s", repo, path, err)
        res.files_parsed = len(parsed)
        parsed_paths = {pf.path for pf in parsed}
        # supported files that produced nothing (emptied, binary, too large) are removed
        vanished = [p for p in paths if p not in parsed_paths and p not in errors]
        all_deletes = list(dict.fromkeys([*delete_paths, *vanished]))

        deltas: dict[str, SymbolDelta] = {}
        for pf in parsed:
            deltas[pf.path] = diff_symbols(pf.symbols, d.ledger.file_entries(repo, pf.path))
        deleted_entries: list[LedgerEntry] = []
        for p in all_deletes:
            deleted_entries.extend(d.ledger.file_entries(repo, p))
        deleted_ids = [sid for dl in deltas.values() for sid in dl.to_delete] + [e.symbol_id for e in deleted_entries]
        deleted_names = {e.name for e in deleted_entries}
        if any(dl.to_delete for dl in deltas.values()):
            old = d.ledger.symbol_entries([sid for dl in deltas.values() for sid in dl.to_delete])
            deleted_names |= {e.name for e in old.values()}

        to_embed = [s for dl in deltas.values() for s in dl.to_embed]
        relocated = [s for dl in deltas.values() for s in dl.to_relocate]

        summaries: dict[SymbolId, str] = {}
        if to_embed:
            summaries, failed, pending = await self._summaries_and_vectors_prepare(
                repo, parsed, blobs, deltas, mode, timings
            )
            res.summaries_failed = failed
            res.pending_summary = pending
        chunks, code_vecs, summary_vecs = await self._embed(to_embed, summaries, timings)

        async with d.write_lock:
            with d.tracer.span(
                "write_stores",
                input={"embed": len(chunks), "relocate": len(relocated), "delete": len(deleted_ids)},
            ):
                t = time.perf_counter()
                await self._write_stores(parsed, chunks, code_vecs, summary_vecs, relocated, deleted_ids, timings)
                timings["write_stores"] = timings.get("write_stores", 0) + (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            for pf in parsed:
                d.ledger.replace_file(pf, line_count=blobs[pf.path].count(b"\n") + 1)
            for p in all_deletes:
                d.ledger.delete_file(repo, p)
            timings["write_ledger"] = timings.get("write_ledger", 0) + (time.perf_counter() - t) * 1000

        res.embedded = len(to_embed)
        res.relocated = len(relocated)
        res.deleted = len(deleted_ids)
        res.unchanged = sum(len(dl.unchanged) for dl in deltas.values())
        for pf in parsed:
            res.touched_src_ids.update(s.symbol_id for s in pf.symbols)
        res.changed_names = {s.name for s in to_embed} | deleted_names
        return res

    async def _write_stores(
        self,
        parsed: Sequence[ParsedFile],
        chunks: list[CodeChunk],
        code_vecs: list[list[float]],
        summary_vecs: list[list[float] | None],
        relocated: list[SymbolNode],
        deleted_ids: list[SymbolId],
        timings: dict[str, float],
    ) -> None:
        d = self._d

        def tick(name: str, start: float) -> float:
            now = time.perf_counter()
            timings[name] = timings.get(name, 0) + (now - start) * 1000
            return now

        t = time.perf_counter()
        if chunks:
            await d.vectors.upsert(chunks, code_vecs, summary_vecs)
        existing: dict[SymbolId, CodeChunk] = {}
        if relocated:
            await d.vectors.update_locations(relocated)
            existing = await d.vectors.fetch([s.symbol_id for s in relocated])
        if deleted_ids:
            await d.vectors.delete(deleted_ids)
        t = tick("write_vectors", t)

        if chunks:
            d.lexical.upsert(chunks)
        if relocated:
            d.lexical.upsert(
                [
                    CodeChunk(
                        symbol=s,
                        summary=existing[s.symbol_id].summary if s.symbol_id in existing else None,
                        summary_model=existing[s.symbol_id].summary_model if s.symbol_id in existing else None,
                        embedding_model=d.code_embedder.model_name,
                    )
                    for s in relocated
                ]
            )
        if deleted_ids:
            d.lexical.delete(deleted_ids)
        d.lexical.commit()
        t = tick("write_lexical", t)

        if deleted_ids:
            d.graph.delete_symbols(deleted_ids)
        d.graph.upsert_symbols([c.symbol for c in chunks] + relocated)
        d.graph.replace_rpc_links(
            [s.symbol_id for pf in parsed for s in pf.symbols], [link for pf in parsed for link in pf.rpc_links]
        )
        tick("write_graph", t)

    async def _summaries_and_vectors_prepare(
        self,
        repo: str,
        parsed: Sequence[ParsedFile],
        blobs: dict[str, bytes],
        deltas: dict[str, SymbolDelta],
        mode: Literal["realtime", "batch", "none"],
        timings: dict[str, float],
    ) -> tuple[dict[SymbolId, str], int, list[FileToSummarize]]:
        items: list[FileToSummarize] = []
        for pf in parsed:
            syms = deltas[pf.path].to_embed
            if self._o.exclude_tests_from_summary:
                syms = tuple(s for s in syms if not s.is_test)
            if syms:
                items.append(
                    FileToSummarize(
                        repo=repo,
                        path=pf.path,
                        language=pf.language.value,
                        file_source=blobs[pf.path].decode("utf-8", errors="replace"),
                        symbols=tuple(syms),
                    )
                )
        if mode == "none" or not items:
            return {}, 0, []
        if mode == "batch":
            return {}, 0, items
        summarizer = self._d.summarizer
        assert summarizer is not None
        with self._d.tracer.span(
            "summarize", input={"files": len(items)}, as_type="generation", model=summarizer.model_name
        ):
            t = time.perf_counter()
            results = await asyncio.gather(*(summarizer.summarize_file(it) for it in items), return_exceptions=True)
            timings["summarize"] = timings.get("summarize", 0) + (time.perf_counter() - t) * 1000
        summaries: dict[SymbolId, str] = {}
        failed = 0
        for it, r in zip(items, results, strict=True):
            if isinstance(r, BaseException):
                if not isinstance(r, ProviderError | asyncio.TimeoutError):
                    log.exception("unexpected summarizer error for %s", it.path, exc_info=r)
                else:
                    log.warning("summary failed for %s/%s: %s (indexed without summary)", repo, it.path, r)
                failed += len(it.symbols)
                continue
            summaries.update(r)
            failed += sum(1 for s in it.symbols if s.symbol_id not in r)
        return summaries, failed, []

    async def _embed(
        self, symbols: Sequence[SymbolNode], summaries: dict[SymbolId, str], timings: dict[str, float]
    ) -> tuple[list[CodeChunk], list[list[float]], list[list[float] | None]]:
        if not symbols:
            return [], [], []
        d = self._d
        with d.tracer.span(
            "embed", input={"symbols": len(symbols)}, as_type="embedding", model=d.code_embedder.model_name
        ):
            t = time.perf_counter()
            with_summary = [s for s in symbols if s.symbol_id in summaries]
            code_task = asyncio.create_task(
                d.code_embedder.embed([code_embedding_text(s) for s in symbols], "document")
            )
            summary_vecs_list = (
                await d.summary_embedder.embed(
                    [summary_embedding_text(s, summaries[s.symbol_id]) for s in with_summary], "document"
                )
                if with_summary
                else []
            )
            code_vecs = await code_task
            timings["embed"] = timings.get("embed", 0) + (time.perf_counter() - t) * 1000
        by_id = {s.symbol_id: v for s, v in zip(with_summary, summary_vecs_list, strict=True)}
        summary_model = d.summarizer.model_name if d.summarizer else None
        chunks = [
            CodeChunk(
                symbol=s,
                summary=summaries.get(s.symbol_id),
                summary_model=summary_model if s.symbol_id in summaries else None,
                embedding_model=d.code_embedder.model_name,
            )
            for s in symbols
        ]
        return chunks, code_vecs, [by_id.get(s.symbol_id) for s in symbols]

    # ------------------------------------------------------------------ resolution
    async def _resolve(self, result: _SliceResult) -> int:
        d = self._d
        self._resolver.invalidate()
        src_ids: set[SymbolId] = set(result.touched_src_ids)
        if result.changed_names:
            incoming = d.ledger.refs_targeting(sorted(result.changed_names))
            src_ids.update(r.src_id for r in incoming)
        if not src_ids:
            return 0
        existing = set(d.ledger.symbol_entries(sorted(src_ids)))
        refs = d.ledger.refs_from(sorted(existing))
        edges = await asyncio.to_thread(self._resolver.resolve, refs)
        async with d.write_lock:
            d.graph.replace_outgoing_edges(sorted(existing), edges)
        return len(edges)

    async def _submit_batch(self, repo: str, items: list[FileToSummarize]) -> None:
        bs = self._d.batch_summarizer
        if bs is None:
            return
        batch_id = await bs.submit(items)
        self._d.ledger.add_pending_batch(batch_id, repo)
        log.info("submitted summary batch %s for %d files in %s", batch_id, len(items), repo)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    async def _timed(timings: dict[str, float], name: str, coro: Awaitable[T]) -> T:
        t = time.perf_counter()
        try:
            return await coro
        finally:
            timings[name] = timings.get(name, 0) + (time.perf_counter() - t) * 1000

    @staticmethod
    def _stats(
        repo: str,
        from_sha: str | None,
        to_sha: str,
        timings: dict[str, float],
        t0: float,
        r: _SliceResult,
        files_changed: int,
        edges: int,
    ) -> IngestStats:
        timings = {k: round(v, 1) for k, v in timings.items()}
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return IngestStats(
            repo=repo,
            from_sha=from_sha,
            to_sha=to_sha,
            files_changed=files_changed,
            files_parsed=r.files_parsed,
            symbols_embedded=r.embedded,
            symbols_relocated=r.relocated,
            symbols_deleted=r.deleted,
            symbols_unchanged=r.unchanged,
            edges_resolved=edges,
            summaries_failed=r.summaries_failed,
            timings_ms=timings,
        )


class SummaryEnrichment:
    """Applies results of asynchronous summary batches (bulk index) to vectors + BM25."""

    def __init__(self, deps: IngestDependencies) -> None:
        self._d = deps

    async def sync(self) -> dict[str, int]:
        d = self._d
        bs = d.batch_summarizer
        if bs is None:
            return {}
        applied: dict[str, int] = defaultdict(int)
        for batch_id, _repo in d.ledger.pending_batches():
            if await bs.poll(batch_id) != "ended":
                continue
            summaries = await bs.collect(batch_id)
            chunks = await d.vectors.fetch(list(summaries))
            live = [(sid, summaries[sid], chunks[sid]) for sid in summaries if sid in chunks]
            if live:
                vecs = await d.summary_embedder.embed(
                    [summary_embedding_text(c.symbol, s) for _, s, c in live], "document"
                )
                async with d.write_lock:
                    await d.vectors.set_summaries(
                        [(sid, s, bs.model_name, v) for (sid, s, _), v in zip(live, vecs, strict=True)]
                    )
                    d.lexical.upsert(
                        [
                            CodeChunk(
                                symbol=c.symbol,
                                summary=s,
                                summary_model=bs.model_name,
                                embedding_model=c.embedding_model,
                            )
                            for _, s, c in live
                        ]
                    )
                    d.lexical.commit()
            d.ledger.complete_batch(batch_id)
            applied[batch_id] = len(live)
        return dict(applied)
