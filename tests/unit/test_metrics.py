from __future__ import annotations

import math

from code_rag.application.evaluation.evaluator import GoldItem
from code_rag.application.evaluation.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from code_rag.domain.entities import SymbolNode
from code_rag.domain.value_objects import (
    Language,
    LineSpan,
    RepoCoordinate,
    SymbolKind,
    make_symbol_id,
    sha256_hex,
)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([0, 0, 1, 1]) == 1 / 3
    assert reciprocal_rank([0] * 10 + [1]) == 0.0
    assert reciprocal_rank([1]) == 1.0


def test_ndcg_graded() -> None:
    assert ndcg_at_k([3, 2], [3, 2]) == 1.0
    swapped = ndcg_at_k([2, 3], [3, 2])
    assert 0.8 < swapped < 1.0
    missing = ndcg_at_k([3, 0], [3, 2])
    expected = (2**3 - 1) / ((2**3 - 1) + (2**2 - 1) / math.log2(3))
    assert abs(missing - expected) < 1e-9
    assert ndcg_at_k([0, 0], []) == 0.0


def test_recall_and_percentiles() -> None:
    assert recall_at_k(1, 4) == 0.25
    vals = list(range(1, 101))
    assert percentile(vals, 50) == 50.5
    assert abs(percentile(vals, 95) - 95.05) < 1e-9
    assert percentile([], 99) == 0.0


def test_gold_matching_by_name_and_location() -> None:
    sym = SymbolNode(
        symbol_id=make_symbol_id("acme/x", "a/b.py", "a.b.C.run", SymbolKind.METHOD),
        coord=RepoCoordinate(repo="acme/x", path="a/b.py", commit_sha="e" * 40),
        language=Language.PYTHON,
        kind=SymbolKind.METHOD,
        name="run",
        qualified_name="a.b.C.run",
        span=LineSpan(start=10, end=20),
        signature="def run",
        source="",
        content_hash=sha256_hex("z"),
    )
    assert GoldItem(repo="acme/x", qualified_name="a.b.C.run").matches(sym)
    assert GoldItem(repo="acme/x", qualified_name="C.run").matches(sym)
    assert not GoldItem(repo="acme/y", qualified_name="a.b.C.run").matches(sym)
    assert GoldItem(repo="acme/x", path="a/b.py", start_line=18, end_line=30).matches(sym)
    assert not GoldItem(repo="acme/x", path="a/b.py", start_line=21).matches(sym)
