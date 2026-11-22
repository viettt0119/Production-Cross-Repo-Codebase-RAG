from __future__ import annotations

from code_rag.application.citation.verifier import CitationVerifier
from code_rag.domain.answer import ClaimVerdict
from code_rag.domain.entities import CodeChunk, SymbolNode
from code_rag.domain.query import RankedContext
from code_rag.domain.value_objects import (
    Language,
    LineSpan,
    RepoCoordinate,
    SymbolKind,
    make_symbol_id,
    sha256_hex,
)

SOURCE = """def charge(self, order_id, total, key):
    request = billing_pb2.ChargeRequest(order_id=order_id)
    response = self._stub.Charge(request, timeout=self._timeout_s)
    return response.charge_id"""


class _Lines:
    def line_count(self, repo: str, path: str) -> int | None:
        return {
            ("acme/checkout-py", "checkout/billing_client.py"): 40,
            ("acme/billing-go", "server.go"): 50,
        }.get((repo, path))


def _ctx() -> RankedContext:
    sym = SymbolNode(
        symbol_id=make_symbol_id("acme/checkout-py", "checkout/billing_client.py", "c.charge", SymbolKind.METHOD),
        coord=RepoCoordinate(repo="acme/checkout-py", path="checkout/billing_client.py", commit_sha="c" * 40),
        language=Language.PYTHON,
        kind=SymbolKind.METHOD,
        name="charge",
        qualified_name="c.charge",
        span=LineSpan(start=19, end=28),
        signature="def charge",
        source=SOURCE,
        content_hash=sha256_hex("x"),
    )
    return RankedContext(chunk=CodeChunk(symbol=sym, embedding_model="m"), fused_score=1.0)


V = CitationVerifier(_Lines())
CITE = "(acme/checkout-py/checkout/billing_client.py:19-28)"


def test_fully_cited_answer_passes() -> None:
    answer = f"The client calls the Charge RPC with a deadline {CITE}.\n\n- It returns the charge id {CITE}."
    report = V.verify(answer, [_ctx()])
    assert report.passed and report.faithfulness == 1.0
    assert report.cited_symbol_ids == (_ctx().symbol_id,)


def test_missing_citation_fails() -> None:
    report = V.verify("The client calls the Charge RPC with a deadline.", [_ctx()])
    assert not report.passed
    assert report.checks[0].verdict is ClaimVerdict.MISSING_CITATION


def test_paragraph_end_citation_covers_previous_sentences() -> None:
    answer = f"The client builds a ChargeRequest from the order. Then it calls the stub with a timeout {CITE}."
    assert V.verify(answer, [_ctx()]).passed


def test_fabricated_and_out_of_context_citations() -> None:
    unknown = V.verify("The ledger records every single charge (acme/nope/x.py:1-2).", [_ctx()])
    assert unknown.checks[0].verdict is ClaimVerdict.UNKNOWN_FILE
    out_of_range = V.verify(
        "The client sets a long timeout here (acme/checkout-py/checkout/billing_client.py:30-99).", [_ctx()]
    )
    assert out_of_range.checks[0].verdict is ClaimVerdict.SPAN_OUT_OF_RANGE
    not_shown = V.verify("The server dedupes idempotency keys first (acme/billing-go/server.go:28-38).", [_ctx()])
    assert not_shown.checks[0].verdict is ClaimVerdict.NOT_IN_CONTEXT


def test_code_snippets_must_be_verbatim() -> None:
    good = (
        f"It calls the stub {CITE}.\n\n```python\nresponse = self._stub.Charge(request, timeout=self._timeout_s)\n```"
    )
    assert V.verify(good, [_ctx()]).passed
    bad = f"It calls the stub {CITE}.\n\n```python\nresponse = self._stub.Charge(request, retries=5)\n```"
    report = V.verify(bad, [_ctx()])
    assert not report.passed
    assert any(c.verdict is ClaimVerdict.NOT_IN_CONTEXT and "snippet" in c.detail for c in report.checks)


def test_abstention_and_pruning() -> None:
    abstain = V.verify("INSUFFICIENT_CONTEXT: the refund path is not in the sources.", [_ctx()])
    assert abstain.faithfulness == 1.0 and not abstain.checks
    mixed = f"The client calls the Charge RPC {CITE}.\n\nIt also retries three times on failure."
    report = V.verify(mixed, [_ctx()])
    pruned = CitationVerifier.strip_unverified(mixed, report)
    assert "retries three times" not in pruned and V.verify(pruned, [_ctx()]).passed
    assert "missing_citation" in CitationVerifier.feedback(report)


def test_headings_and_intro_lines_are_not_claims() -> None:
    answer = f"## Flow\n\nHere is how it works:\n\n- The client calls the Charge RPC {CITE}."
    assert V.verify(answer, [_ctx()]).passed
