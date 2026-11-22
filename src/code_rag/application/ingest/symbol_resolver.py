"""Late-binding resolution of SymbolRefs into SymbolEdges across repository boundaries.

References are captured by *name* at parse time because the target may live in a repo
that has not been indexed yet. Resolution runs after every ingest and re-runs for refs
whose target names were added/removed, so edges self-heal in both directions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from code_rag.domain.changeset import LedgerEntry
from code_rag.domain.entities import SymbolEdge, SymbolRef
from code_rag.domain.ports.indexing import HashLedger
from code_rag.domain.value_objects import EdgeKind, SymbolId, SymbolKind

_TYPE_KINDS = frozenset(
    {
        SymbolKind.CLASS.value,
        SymbolKind.STRUCT.value,
        SymbolKind.INTERFACE.value,
        SymbolKind.TRAIT.value,
        SymbolKind.ENUM.value,
        SymbolKind.TYPE_ALIAS.value,
    }
)
_CALLABLE_EXCLUDED = frozenset({SymbolKind.MODULE.value, SymbolKind.IMPL.value})


@dataclass(frozen=True, slots=True)
class _Candidate:
    entry: LedgerEntry
    score: float
    confidence: float


def parse_imports(imports: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in imports:
        alias, _, full = item.partition("=")
        if alias and full:
            out[alias] = full.replace("/", ".")
    return out


def _common_suffix(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(reversed(a), reversed(b), strict=False):
        if x != y:
            break
        n += 1
    return n


class SymbolResolver:
    def __init__(self, ledger: HashLedger, min_confidence: float = 0.35, max_ties: int = 3) -> None:
        self._ledger = ledger
        self._min_confidence = min_confidence
        self._max_ties = max_ties
        self._import_cache: dict[tuple[str, str], dict[str, str]] = {}

    def _imports_for(self, repo: str, path: str) -> dict[str, str]:
        key = (repo, path)
        cached = self._import_cache.get(key)
        if cached is None:
            rec = self._ledger.get_file(repo, path)
            cached = parse_imports(rec.imports) if rec is not None else {}
            self._import_cache[key] = cached
        return cached

    def invalidate(self) -> None:
        self._import_cache.clear()

    def resolve(self, refs: Sequence[SymbolRef]) -> list[SymbolEdge]:
        edges: dict[tuple[SymbolId, SymbolId, EdgeKind], SymbolEdge] = {}
        name_cache: dict[str, list[LedgerEntry]] = {}
        for ref in refs:
            for cand in self._candidates(ref, name_cache):
                key = (ref.src_id, cand.entry.symbol_id, ref.kind)
                existing = edges.get(key)
                if existing is None or existing.confidence < cand.confidence:
                    edges[key] = SymbolEdge(
                        src_id=ref.src_id,
                        dst_id=cand.entry.symbol_id,
                        kind=ref.kind,
                        cross_repo=cand.entry.repo != ref.src_repo,
                        confidence=round(cand.confidence, 3),
                        at_line=ref.at_line,
                    )
        return list(edges.values())

    def _candidates(self, ref: SymbolRef, name_cache: dict[str, list[LedgerEntry]]) -> list[_Candidate]:
        segs = ref.target_name.split(".")
        imports = self._imports_for(ref.src_repo, ref.src_path)
        expanded = False
        if segs[0] in imports:
            segs = [*imports[segs[0]].split("."), *segs[1:]]
            expanded = True
        name = segs[-1]
        pool = name_cache.get(name)
        if pool is None:
            pool = self._ledger.lookup_by_name(name)
            name_cache[name] = pool
        scored: list[_Candidate] = []
        unique_global = len(pool) == 1
        for entry in pool:
            if entry.symbol_id == ref.src_id or not self._kind_ok(ref.kind, entry.kind):
                continue
            cq = entry.qualified_name.split(".")
            suffix = _common_suffix(segs, cq)
            if suffix == 0:
                continue
            same_repo = entry.repo == ref.src_repo
            same_file = same_repo and entry.path == ref.src_path
            confidence: float
            score = float(suffix)
            if suffix >= 2:
                confidence = 0.95 if expanded else 0.8
            elif len(segs) >= 2 and len(cq) >= 2 and self._receiver_matches(segs[-2], cq[-2]):
                confidence, score = 0.6, score + 0.5
            elif same_file:
                confidence = 0.7
            elif same_repo and len(segs) == 1:
                confidence = 0.55
            elif unique_global:
                confidence = 0.4
            else:
                continue
            score += 0.25 * same_repo + 0.25 * same_file
            scored.append(_Candidate(entry=entry, score=score, confidence=confidence))
        if not scored:
            return []
        scored.sort(key=lambda c: (c.score, c.confidence), reverse=True)
        best = scored[0].score
        top = [c for c in scored if abs(c.score - best) < 1e-6]
        if len(top) > self._max_ties:
            return []
        share = 1.0 / len(top)
        out = [
            _Candidate(
                entry=c.entry,
                score=c.score,
                confidence=c.confidence * (1.0 if share == 1.0 else max(share, 0.5)),
            )
            for c in top
        ]
        return [c for c in out if c.confidence >= self._min_confidence]

    @staticmethod
    def _receiver_matches(receiver: str, owner: str) -> bool:
        """`self.billing.charge` -> owner `BillingClient`: receiver name is a substring of the owner type."""
        recv = receiver.lower().lstrip("_")
        return len(recv) >= 3 and recv in owner.lower()

    @staticmethod
    def _kind_ok(edge: EdgeKind, target_kind: str) -> bool:
        if edge in (EdgeKind.INHERITS, EdgeKind.IMPLEMENTS):
            return target_kind in _TYPE_KINDS
        return target_kind not in _CALLABLE_EXCLUDED


def affected_names(names: Iterable[str]) -> list[str]:
    return sorted({n for n in names if n})


def edges_by_src(edges: Sequence[SymbolEdge]) -> Mapping[SymbolId, list[SymbolEdge]]:
    out: dict[SymbolId, list[SymbolEdge]] = {}
    for e in edges:
        out.setdefault(e.src_id, []).append(e)
    return out
