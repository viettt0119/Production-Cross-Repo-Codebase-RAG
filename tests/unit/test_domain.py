from __future__ import annotations

import pytest
from pydantic import ValidationError

from code_rag.domain.answer import Citation
from code_rag.domain.value_objects import (
    LineSpan,
    RepoCoordinate,
    SymbolKind,
    make_symbol_id,
    symbol_point_id,
)


def test_line_span_validates_order() -> None:
    with pytest.raises(ValidationError):
        LineSpan(start=10, end=3)
    span = LineSpan(start=3, end=10)
    assert span.contains(LineSpan(start=4, end=9))
    assert span.overlaps(LineSpan(start=10, end=12))
    assert not span.overlaps(LineSpan(start=11, end=12))
    assert span.length == 8


def test_repo_coordinate_rejects_escaping_paths() -> None:
    with pytest.raises(ValidationError):
        RepoCoordinate(repo="acme/x", path="../etc/passwd", commit_sha="a" * 40)
    with pytest.raises(ValidationError):
        RepoCoordinate(repo="no-slash", path="a.py", commit_sha="a" * 40)


def test_symbol_id_is_stable_and_discriminating() -> None:
    a = make_symbol_id("acme/x", "a.py", "a.f", SymbolKind.FUNCTION)
    assert a == make_symbol_id("acme/x", "a.py", "a.f", SymbolKind.FUNCTION)
    assert a != make_symbol_id("acme/x", "a.py", "a.f", SymbolKind.FUNCTION, ordinal=1)
    assert a != make_symbol_id("acme/y", "a.py", "a.f", SymbolKind.FUNCTION)
    assert symbol_point_id(a) == symbol_point_id(a)


def test_citation_parsing() -> None:
    text = (
        "Charge is idempotent (acme/billing-go/internal/server/server.go:28-38) and the client "
        "sets a deadline (acme/checkout-py/checkout/billing_client.py:27). Bad (x/y:0-1)."
    )
    cites = Citation.parse_all(text)
    assert [(c.repo, c.path, c.span.start, c.span.end) for c in cites] == [
        ("acme/billing-go", "internal/server/server.go", 28, 38),
        ("acme/checkout-py", "checkout/billing_client.py", 27, 27),
    ]
    assert cites[0].render() == "(acme/billing-go/internal/server/server.go:28-38)"
