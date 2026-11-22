"""Symbol-level change detection: decides what must be re-embedded vs. merely relocated."""

from __future__ import annotations

from collections.abc import Sequence

from code_rag.domain.changeset import LedgerEntry, SymbolDelta
from code_rag.domain.entities import SymbolNode
from code_rag.domain.value_objects import SymbolId


def diff_symbols(new: Sequence[SymbolNode], old: Sequence[LedgerEntry]) -> SymbolDelta:
    """Compare freshly parsed symbols of one file with the ledger's view of that file.

    * new id or changed content_hash -> re-summarize + re-embed (`to_embed`)
    * same hash, moved lines           -> payload-only update   (`to_relocate`)
    * same hash, same lines            -> nothing               (`unchanged`)
    * id no longer present             -> delete everywhere     (`to_delete`)
    """
    old_by_id: dict[SymbolId, LedgerEntry] = {e.symbol_id: e for e in old}
    to_embed: list[SymbolNode] = []
    to_relocate: list[SymbolNode] = []
    unchanged: list[SymbolNode] = []
    for sym in new:
        prev = old_by_id.get(sym.symbol_id)
        if prev is None or prev.content_hash != sym.content_hash:
            to_embed.append(sym)
        elif prev.start_line != sym.span.start or prev.end_line != sym.span.end:
            to_relocate.append(sym)
        else:
            unchanged.append(sym)
    new_ids = {s.symbol_id for s in new}
    to_delete = tuple(sid for sid in old_by_id if sid not in new_ids)
    return SymbolDelta(
        to_embed=tuple(to_embed),
        to_relocate=tuple(to_relocate),
        to_delete=to_delete,
        unchanged=tuple(unchanged),
    )


def merge_deltas(deltas: Sequence[SymbolDelta]) -> SymbolDelta:
    return SymbolDelta(
        to_embed=tuple(s for d in deltas for s in d.to_embed),
        to_relocate=tuple(s for d in deltas for s in d.to_relocate),
        to_delete=tuple(s for d in deltas for s in d.to_delete),
        unchanged=tuple(s for d in deltas for s in d.unchanged),
    )
