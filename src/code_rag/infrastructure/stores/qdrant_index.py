"""Qdrant VectorIndex adapter: named vectors `code` + `summary`, deterministic point ids,
one collection per embedding model (A/B safe) and an alias for the active one."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from code_rag.domain.entities import CodeChunk, SymbolNode
from code_rag.domain.errors import IndexStoreError
from code_rag.domain.ports.indexing import ScoredId, VectorName
from code_rag.domain.query import SearchFilter
from code_rag.domain.value_objects import (
    Language,
    LineSpan,
    RepoCoordinate,
    SymbolId,
    SymbolKind,
    symbol_point_id,
)

log = logging.getLogger(__name__)
_UPSERT_BATCH = 256
_KEYWORD_FIELDS = ("repo", "path", "language", "kind", "symbol_id")
_BOOL_FIELDS = ("is_generated", "is_test")


def collection_name(prefix: str, model: str, dim: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"{prefix}__{slug}__{dim}"


def _payload(chunk: CodeChunk) -> dict[str, Any]:
    s = chunk.symbol
    return {
        "symbol_id": s.symbol_id,
        "repo": s.coord.repo,
        "path": s.coord.path,
        "commit_sha": s.coord.commit_sha,
        "language": s.language.value,
        "kind": s.kind.value,
        "name": s.name,
        "qualified_name": s.qualified_name,
        "parent_id": s.parent_id,
        "start_line": s.span.start,
        "end_line": s.span.end,
        "signature": s.signature,
        "source": s.source,
        "content_hash": s.content_hash,
        "is_generated": s.is_generated,
        "is_test": s.is_test,
        "summary": chunk.summary,
        "summary_model": chunk.summary_model,
        "has_summary": chunk.summary is not None,
        "embedding_model": chunk.embedding_model,
    }


def chunk_from_payload(p: dict[str, Any]) -> CodeChunk:
    sym = SymbolNode(
        symbol_id=SymbolId(p["symbol_id"]),
        coord=RepoCoordinate(repo=p["repo"], path=p["path"], commit_sha=p["commit_sha"]),
        language=Language(p["language"]),
        kind=SymbolKind(p["kind"]),
        name=p["name"],
        qualified_name=p["qualified_name"],
        parent_id=SymbolId(p["parent_id"]) if p.get("parent_id") else None,
        span=LineSpan(start=int(p["start_line"]), end=int(p["end_line"])),
        signature=p.get("signature", ""),
        source=p.get("source", ""),
        content_hash=p["content_hash"],
        is_generated=bool(p.get("is_generated", False)),
        is_test=bool(p.get("is_test", False)),
    )
    return CodeChunk(
        symbol=sym,
        summary=p.get("summary"),
        summary_model=p.get("summary_model"),
        embedding_model=p.get("embedding_model", "unknown"),
    )


class QdrantVectorIndex:
    def __init__(
        self,
        client: AsyncQdrantClient,
        embedding_model: str,
        code_dimension: int,
        summary_dimension: int | None = None,
        prefix: str = "chunks",
        server_side_boost: bool = True,
        on_disk: bool = True,
        quantize: bool = True,
        payload_indexes: bool = True,
    ) -> None:
        self._client = client
        self._embedding_model = embedding_model
        self._code_dim = code_dimension
        self._summary_dim = summary_dimension or code_dimension
        self._prefix = prefix
        self.collection = collection_name(prefix, embedding_model, code_dimension)
        self.alias = f"{prefix}_active"
        self._server_side_boost = server_side_boost
        self._on_disk = on_disk
        self._quantize = quantize
        self._payload_indexes = payload_indexes
        self._ready = False

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            if not await self._client.collection_exists(self.collection):
                quant = (
                    qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True))
                    if self._quantize
                    else None
                )
                await self._client.create_collection(
                    self.collection,
                    vectors_config={
                        "code": qm.VectorParams(
                            size=self._code_dim,
                            distance=qm.Distance.COSINE,
                            on_disk=self._on_disk,
                            quantization_config=quant,
                        ),
                        "summary": qm.VectorParams(size=self._summary_dim, distance=qm.Distance.COSINE),
                    },
                    on_disk_payload=self._on_disk,
                )
                if self._payload_indexes:
                    for field in _KEYWORD_FIELDS:
                        await self._client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)
                    for field in _BOOL_FIELDS:
                        await self._client.create_payload_index(self.collection, field, qm.PayloadSchemaType.BOOL)
            aliases = await self._client.get_aliases()
            if not any(a.alias_name == self.alias for a in aliases.aliases):
                await self.activate()
        except UnexpectedResponse as exc:
            raise IndexStoreError(f"qdrant ensure_ready failed: {exc}") from exc
        self._ready = True

    async def activate(self) -> None:
        """Point the `<prefix>_active` alias at this model's collection (A/B cut-over)."""
        await self._client.update_collection_aliases(
            change_aliases_operations=[
                qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=self.alias)),
                qm.CreateAliasOperation(
                    create_alias=qm.CreateAlias(collection_name=self.collection, alias_name=self.alias)
                ),
            ]
            if await self._alias_exists()
            else [
                qm.CreateAliasOperation(
                    create_alias=qm.CreateAlias(collection_name=self.collection, alias_name=self.alias)
                )
            ]
        )

    async def _alias_exists(self) -> bool:
        aliases = await self._client.get_aliases()
        return any(a.alias_name == self.alias for a in aliases.aliases)

    async def upsert(
        self,
        chunks: Sequence[CodeChunk],
        code_vectors: Sequence[Sequence[float]],
        summary_vectors: Sequence[Sequence[float] | None],
    ) -> None:
        if not (len(chunks) == len(code_vectors) == len(summary_vectors)):
            raise IndexStoreError("chunks/vectors length mismatch")
        points: list[qm.PointStruct] = []
        for chunk, cv, sv in zip(chunks, code_vectors, summary_vectors, strict=True):
            vector: dict[str, Any] = {"code": list(cv)}
            if sv is not None:
                vector["summary"] = list(sv)
            points.append(
                qm.PointStruct(id=symbol_point_id(chunk.symbol.symbol_id), vector=vector, payload=_payload(chunk))
            )
        for i in range(0, len(points), _UPSERT_BATCH):
            await self._client.upsert(self.collection, points=points[i : i + _UPSERT_BATCH], wait=True)

    async def set_summaries(self, items: Sequence[tuple[SymbolId, str, str, Sequence[float]]]) -> None:
        if not items:
            return
        await self._client.update_vectors(
            self.collection,
            points=[
                qm.PointVectors(id=symbol_point_id(sid), vector={"summary": list(vec)}) for sid, _, _, vec in items
            ],
            wait=True,
        )
        ops = [
            qm.SetPayloadOperation(
                set_payload=qm.SetPayload(
                    payload={"summary": summary, "summary_model": model, "has_summary": True},
                    points=[symbol_point_id(sid)],
                )
            )
            for sid, summary, model, _ in items
        ]
        for i in range(0, len(ops), _UPSERT_BATCH):
            await self._client.batch_update_points(self.collection, update_operations=ops[i : i + _UPSERT_BATCH])

    async def update_locations(self, symbols: Sequence[SymbolNode]) -> None:
        ops = [
            qm.SetPayloadOperation(
                set_payload=qm.SetPayload(
                    payload={
                        "start_line": s.span.start,
                        "end_line": s.span.end,
                        "commit_sha": s.coord.commit_sha,
                        "source": s.source,
                        "signature": s.signature,
                    },
                    points=[symbol_point_id(s.symbol_id)],
                )
            )
            for s in symbols
        ]
        for i in range(0, len(ops), _UPSERT_BATCH):
            await self._client.batch_update_points(self.collection, update_operations=ops[i : i + _UPSERT_BATCH])

    async def delete(self, ids: Sequence[SymbolId]) -> None:
        if not ids:
            return
        point_ids: list[qm.ExtendedPointId] = [symbol_point_id(i) for i in ids]
        await self._client.delete(self.collection, points_selector=qm.PointIdsList(points=point_ids), wait=True)

    @staticmethod
    def _filter(flt: SearchFilter) -> qm.Filter | None:
        must: list[qm.Condition] = []
        if flt.repos:
            must.append(qm.FieldCondition(key="repo", match=qm.MatchAny(any=list(flt.repos))))
        if flt.languages:
            must.append(qm.FieldCondition(key="language", match=qm.MatchAny(any=list(flt.languages))))
        if not flt.include_tests:
            must.append(qm.FieldCondition(key="is_test", match=qm.MatchValue(value=False)))
        return qm.Filter(must=must) if must else None

    async def search(self, vector: Sequence[float], using: VectorName, flt: SearchFilter, limit: int) -> list[ScoredId]:
        qfilter = self._filter(flt)
        penalty = 0.0 if flt.include_generated else flt.generated_penalty
        if penalty > 0 and self._server_side_boost:
            try:
                formula = qm.FormulaQuery(
                    formula=qm.MultExpression(
                        mult=[
                            "$score",
                            qm.SumExpression(
                                sum=[
                                    1.0,
                                    qm.MultExpression(
                                        mult=[
                                            -penalty,
                                            qm.FieldCondition(key="is_generated", match=qm.MatchValue(value=True)),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    )
                )
                res = await self._client.query_points(
                    self.collection,
                    prefetch=qm.Prefetch(query=list(vector), using=using, filter=qfilter, limit=limit * 3),
                    query=formula,
                    limit=limit,
                    with_payload=["symbol_id"],
                )
                return [ScoredId(SymbolId(str((p.payload or {})["symbol_id"])), float(p.score)) for p in res.points]
            except (UnexpectedResponse, ValueError, NotImplementedError) as exc:
                log.warning("server-side formula boosting unavailable (%s); using client-side penalty", exc)
                self._server_side_boost = False
        res = await self._client.query_points(
            self.collection,
            query=list(vector),
            using=using,
            query_filter=qfilter,
            limit=limit * 2 if penalty else limit,
            with_payload=["is_generated", "symbol_id"],
        )
        scored = [
            ScoredId(
                SymbolId(str((p.payload or {})["symbol_id"])),
                float(p.score) * ((1 - penalty) if (p.payload or {}).get("is_generated") else 1.0),
            )
            for p in res.points
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    async def fetch(self, ids: Sequence[SymbolId]) -> dict[SymbolId, CodeChunk]:
        if not ids:
            return {}
        point_ids: list[qm.ExtendedPointId] = [symbol_point_id(i) for i in ids]
        records = await self._client.retrieve(self.collection, ids=point_ids, with_payload=True, with_vectors=False)
        out: dict[SymbolId, CodeChunk] = {}
        for r in records:
            if r.payload:
                chunk = chunk_from_payload(dict(r.payload))
                out[chunk.symbol.symbol_id] = chunk
        return out

    async def count(self) -> int:
        res = await self._client.count(self.collection, exact=True)
        return int(res.count)

    async def all_symbol_ids(self) -> set[SymbolId]:
        ids: set[SymbolId] = set()
        offset: qm.ExtendedPointId | None = None
        while True:
            points, offset = await self._client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=["symbol_id"], with_vectors=False
            )
            ids.update(SymbolId(str((p.payload or {})["symbol_id"])) for p in points)
            if offset is None:
                return ids
