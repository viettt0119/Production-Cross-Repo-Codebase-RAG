from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from code_rag.application.retrieval.agent import HybridRetrievalAgent
from code_rag.domain.errors import ProviderRefusalError
from code_rag.domain.query import RankedContext, SearchQuery
from code_rag.infrastructure.container import Container

pytestmark = pytest.mark.integration


async def test_search_ranks_relevant_cross_repo_symbols(indexed: Container) -> None:
    contexts = await indexed.retrieval_agent.search(SearchQuery(text="convert money to minor units"))
    names = [c.chunk.symbol.qualified_name for c in contexts]
    assert "acme_common.money.Money.to_minor_units" in names[:3]


async def test_graph_expansion_pulls_in_rpc_server(indexed: Container) -> None:
    contexts = await indexed.retrieval_agent.search(
        SearchQuery(text="BillingClient charge order idempotency key stub Charge")
    )
    names = {c.chunk.symbol.qualified_name for c in contexts}
    assert "checkout.billing_client.BillingClient.charge" in names
    assert "internal.server.BillingServer.Charge" in names, "Go server reached via RPC bridge"


async def test_ask_returns_verified_answer(indexed: Container) -> None:
    answer = await indexed.retrieval_agent.ask(SearchQuery(text="How is money converted to minor units?"))
    assert answer.report.passed and answer.report.faithfulness == 1.0
    assert answer.attempts == 1 and not answer.abstained
    assert "(acme/shared-py/acme_common/money.py:" in answer.answer_markdown
    assert {"retrieve_hybrid", "rerank", "synthesize", "verify_citations", "total"} <= set(answer.timings_ms)


class _ScriptedSynth:
    """First attempt omits citations, second attempt cites correctly."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    async def stream(
        self, question: str, contexts: Sequence[RankedContext], feedback: str | None
    ) -> AsyncIterator[str]:
        self.calls.append(feedback)
        top = contexts[0].chunk.symbol
        if feedback is None:
            yield "The conversion uses banker's rounding to integer minor units."
        else:
            yield f"The conversion uses banker's rounding to integer minor units {top.citation_anchor()}."


class _RefusingSynth:
    @property
    def model_name(self) -> str:
        return "refuser"

    async def stream(
        self, question: str, contexts: Sequence[RankedContext], feedback: str | None
    ) -> AsyncIterator[str]:
        raise ProviderRefusalError("test", "declined")
        yield ""  # pragma: no cover - makes this an async generator


def _agent_with(c: Container, synth: object) -> HybridRetrievalAgent:
    base = c.retrieval_agent
    deps = base._d
    deps2 = type(deps)(**{**deps.__dict__, "synthesizer": synth})
    return HybridRetrievalAgent(deps2, c.settings.retrieval)


async def test_verifier_feedback_triggers_reask(indexed: Container) -> None:
    synth = _ScriptedSynth()
    agent = _agent_with(indexed, synth)
    answer = await agent.ask(SearchQuery(text="How is money converted to minor units?"))
    assert answer.attempts == 2 and answer.report.passed
    assert synth.calls[0] is None and synth.calls[1] is not None and "missing_citation" in synth.calls[1]


async def test_refusal_becomes_abstention(indexed: Container) -> None:
    answer = await _agent_with(indexed, _RefusingSynth()).ask(SearchQuery(text="How is money converted?"))
    assert answer.abstained and answer.answer_markdown.startswith("INSUFFICIENT_CONTEXT")


async def test_stream_emits_stages_tokens_and_final(indexed: Container) -> None:
    events = [e async for e in indexed.retrieval_agent.stream(SearchQuery(text="How does the cart apply discounts?"))]
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "final" and "token" in kinds
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages[:5] == ["expand_query", "retrieve_hybrid", "fuse_rrf", "expand_graph", "rerank"]
    assert events[-1]["answer"].report.passed
