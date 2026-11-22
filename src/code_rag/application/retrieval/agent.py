"""LangGraph query agent.

    expand_query -> retrieve_hybrid -> fuse_rrf -> expand_graph -> rerank
        -> synthesize -> verify --(failed & attempts left)--> synthesize (with verifier feedback)
                              \\-> finalize (prune unverifiable claims or abstain)

Every node degrades gracefully: expansion/rerank failures fall back to the un-expanded query /
fused order; synthesis refusals become an explicit abstention. Token deltas and stage events are
emitted through LangGraph's custom stream so the API can forward them over SSE.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from code_rag.application.citation.verifier import CitationVerifier, is_abstention
from code_rag.application.retrieval.config import RetrievalConfig
from code_rag.application.retrieval.fusion import FusedHit, RankedList, rrf_fuse
from code_rag.application.timing import collect_timings
from code_rag.domain.answer import CitedAnswer, VerificationReport
from code_rag.domain.errors import ProviderError, ProviderRefusalError
from code_rag.domain.ports.indexing import LexicalIndex, SymbolGraph, VectorIndex
from code_rag.domain.ports.models import (
    AnswerSynthesizer,
    EmbeddingProvider,
    QueryExpander,
    QueryExpansion,
    Reranker,
)
from code_rag.domain.ports.observability import Tracer
from code_rag.domain.query import HitSource, RankedContext, SearchFilter, SearchQuery
from code_rag.domain.value_objects import SymbolKind

log = logging.getLogger(__name__)

GRAPH_SCORE_DECAY = 0.5
GRAPH_RESERVED_SLOTS = 2
CONTAINER_DEMOTION = 0.8
REWRITE_WEIGHT = 0.7
RERANK_DOC_SOURCE_CHARS = 1500
REFUSAL_MESSAGE = "INSUFFICIENT_CONTEXT: the model declined to answer this request."
UNVERIFIABLE_MESSAGE = "INSUFFICIENT_CONTEXT: no verifiable answer could be produced from the retrieved code."


class AgentState(TypedDict, total=False):
    question: str
    query: SearchQuery
    expansion: QueryExpansion
    ranked_lists: list[RankedList]
    fused: list[FusedHit]
    candidates: list[RankedContext]
    contexts: list[RankedContext]
    answer: str
    report: VerificationReport
    attempts: int
    feedback: str | None
    abstained: bool


@dataclass
class RetrievalDependencies:
    vectors: VectorIndex
    lexical: LexicalIndex
    graph: SymbolGraph
    code_embedder: EmbeddingProvider
    summary_embedder: EmbeddingProvider
    reranker: Reranker
    expander: QueryExpander | None
    synthesizer: AnswerSynthesizer
    verifier: CitationVerifier
    tracer: Tracer


def _emit(event: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:  # called outside a LangGraph run (e.g. search())
        return
    writer(event)


def rerank_document(ctx: RankedContext) -> str:
    s = ctx.chunk.symbol
    summary = f"\n{ctx.chunk.summary}" if ctx.chunk.summary else ""
    return (
        f"{s.coord.repo}/{s.coord.path} {s.kind.value} {s.qualified_name}\n{s.signature}{summary}\n"
        f"{s.source[:RERANK_DOC_SOURCE_CHARS]}"
    )


class HybridRetrievalAgent:
    def __init__(self, deps: RetrievalDependencies, settings: RetrievalConfig | None = None) -> None:
        self._d = deps
        self._s = settings or RetrievalConfig()
        self._graph = self._build()

    # ------------------------------------------------------------------ graph wiring
    def _build(self) -> Any:
        g: StateGraph[AgentState] = StateGraph(AgentState)
        g.add_node("expand_query", self.expand_query)
        g.add_node("retrieve_hybrid", self.retrieve_hybrid)
        g.add_node("fuse_rrf", self.fuse_rrf)
        g.add_node("expand_graph", self.expand_graph)
        g.add_node("rerank", self.rerank)
        g.add_node("synthesize", self.synthesize)
        g.add_node("verify", self.verify)
        g.add_node("finalize", self.finalize)
        g.add_edge(START, "expand_query")
        g.add_edge("expand_query", "retrieve_hybrid")
        g.add_edge("retrieve_hybrid", "fuse_rrf")
        g.add_edge("fuse_rrf", "expand_graph")
        g.add_edge("expand_graph", "rerank")
        g.add_edge("rerank", "synthesize")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", self._after_verify, {"synthesize": "synthesize", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    def _after_verify(self, state: AgentState) -> Literal["synthesize", "finalize"]:
        report = state.get("report")
        if state.get("abstained") or (report is not None and report.passed):
            return "finalize"
        if state.get("attempts", 0) >= self._s.max_synthesis_attempts:
            return "finalize"
        return "synthesize"

    # ------------------------------------------------------------------ nodes
    async def expand_query(self, state: AgentState) -> AgentState:
        _emit({"type": "stage", "stage": "expand_query"})
        question = state["question"]
        with self._d.tracer.span("expand_query", input=question) as span:
            expansion = QueryExpansion(rewrites=(), identifiers=())
            if self._d.expander is not None:
                try:
                    expansion = await self._d.expander.expand(question)
                except ProviderError as exc:
                    span.set_attribute("degraded", str(exc))
            span.set_output({"rewrites": expansion.rewrites, "identifiers": expansion.identifiers})
        return {"expansion": expansion}

    async def retrieve_hybrid(self, state: AgentState) -> AgentState:
        _emit({"type": "stage", "stage": "retrieve_hybrid"})
        d, s = self._d, self._s
        q = state["query"]
        exp = state.get("expansion") or QueryExpansion(rewrites=(), identifiers=())
        flt = SearchFilter.from_query(q, s.generated_penalty)
        texts = [q.text, *exp.rewrites]
        weights = [1.0] + [REWRITE_WEIGHT] * len(exp.rewrites)
        with d.tracer.span("retrieve_hybrid", input={"queries": texts}) as span:
            code_vecs_task = d.code_embedder.embed(texts, "query")
            if d.summary_embedder is d.code_embedder:
                code_vecs = await code_vecs_task
                summary_vecs = code_vecs
            else:
                code_vecs, summary_vecs = await asyncio.gather(code_vecs_task, d.summary_embedder.embed(texts, "query"))
            bm25_text = " ".join([q.text, *exp.identifiers, *exp.rewrites])
            jobs: list[tuple[HitSource, float, Any]] = []
            for vec, w in zip(code_vecs, weights, strict=True):
                jobs.append((HitSource.DENSE_CODE, w, d.vectors.search(vec, "code", flt, s.dense_k)))
            for vec, w in zip(summary_vecs, weights, strict=True):
                jobs.append((HitSource.DENSE_SUMMARY, w, d.vectors.search(vec, "summary", flt, s.dense_k)))
            jobs.append((HitSource.BM25, 1.0, asyncio.to_thread(d.lexical.search, bm25_text, flt, s.bm25_k)))
            results = await asyncio.gather(*(j[2] for j in jobs))
            lists = [
                RankedList(src, hits, w * s.source_weights.get(src.value, 1.0))
                for (src, w, _), hits in zip(jobs, results, strict=True)
            ]
            span.set_output({f"{rl.source.value}": len(rl.hits) for rl in lists})
        return {"ranked_lists": lists}

    async def fuse_rrf(self, state: AgentState) -> AgentState:
        _emit({"type": "stage", "stage": "fuse_rrf"})
        with self._d.tracer.span("fuse_rrf") as span:
            fused = rrf_fuse(state.get("ranked_lists", []), k=self._s.rrf_k, limit=self._s.fused_k)
            span.set_output({"fused": len(fused)})
        return {"fused": fused}

    async def expand_graph(self, state: AgentState) -> AgentState:
        _emit({"type": "stage", "stage": "expand_graph"})
        d, s = self._d, self._s
        fused = state.get("fused", [])
        with d.tracer.span("expand_graph") as span:
            chunks = await d.vectors.fetch([h.symbol_id for h in fused])
            base = [
                RankedContext(chunk=chunks[h.symbol_id], fused_score=h.score, sources=h.sources)
                for h in fused
                if h.symbol_id in chunks
            ]
            seeds = [c.symbol_id for c in base[: s.graph_seeds]]
            neighbors = await asyncio.to_thread(d.graph.neighbors, seeds, s.graph_limit, s.graph_min_confidence)
            known = {c.symbol_id for c in base}
            new_ids = [n.neighbor_id for n in neighbors if n.neighbor_id not in known]
            extra_chunks = await d.vectors.fetch(new_ids) if new_ids else {}
            seed_score = {c.symbol_id: c.fused_score for c in base}
            extras: list[RankedContext] = []
            for n in neighbors:
                if n.neighbor_id in extra_chunks and n.neighbor_id not in known:
                    known.add(n.neighbor_id)
                    extras.append(
                        RankedContext(
                            chunk=extra_chunks[n.neighbor_id],
                            fused_score=seed_score.get(n.seed_id, 0.0) * GRAPH_SCORE_DECAY,
                            via_graph_from=n.seed_id,
                            sources=(HitSource.GRAPH,),
                        )
                    )
            span.set_output(
                {
                    "base": len(base),
                    "graph_added": len(extras),
                    "relations": sorted({n.relation.split(":")[0] for n in neighbors}),
                }
            )
        return {"candidates": base + extras}

    async def rerank(self, state: AgentState) -> AgentState:
        _emit({"type": "stage", "stage": "rerank"})
        d, s = self._d, self._s
        cands = state.get("candidates", [])
        top_k = min(state["query"].top_k, s.rerank_top_k) if "query" in state else s.rerank_top_k
        with d.tracer.span("rerank", input={"candidates": len(cands)}, model=d.reranker.model_name) as span:
            if not cands:
                return {"contexts": []}
            try:
                results = await d.reranker.rerank(
                    state["question"], [rerank_document(c) for c in cands], top_k=len(cands)
                )
                q = state.get("query")
                penalty = 0.0 if (q is not None and q.include_generated) else s.generated_penalty
                scored = [
                    cands[r.index].model_copy(update={"rerank_score": self._penalize(r.score, cands[r.index], penalty)})
                    for r in results
                ]
                scored = self._demote_containers(scored)
            except ProviderError as exc:
                span.set_attribute("degraded", str(exc))
                scored = sorted(cands, key=lambda c: c.fused_score, reverse=True)
            final = self._graph_aware_select(scored, top_k)
            span.set_output([c.chunk.symbol.qualified_name for c in final])
        return {"contexts": final}

    @staticmethod
    def _demote_containers(scored: list[RankedContext]) -> list[RankedContext]:
        """Prefer the most specific chunk: a class/impl/module that is co-retrieved with one of its own
        members embeds that member's text and would otherwise outrank it."""
        parents = {c.chunk.symbol.parent_id for c in scored if c.chunk.symbol.parent_id}
        files_with_members = {
            (c.chunk.symbol.coord.repo, c.chunk.symbol.coord.path)
            for c in scored
            if c.chunk.symbol.kind is not SymbolKind.MODULE
        }
        out: list[RankedContext] = []
        for c in scored:
            sym, score = c.chunk.symbol, c.rerank_score or 0.0
            is_container = sym.symbol_id in parents or (
                sym.kind is SymbolKind.MODULE and (sym.coord.repo, sym.coord.path) in files_with_members
            )
            if is_container:
                score = score * CONTAINER_DEMOTION if score >= 0 else score / CONTAINER_DEMOTION
            out.append(c.model_copy(update={"rerank_score": score}))
        out.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return out

    @staticmethod
    def _penalize(score: float, ctx: RankedContext, penalty: float) -> float:
        if not ctx.chunk.symbol.is_generated or penalty <= 0:
            return score
        return score * (1 - penalty) if score >= 0 else score * (1 + penalty)

    @staticmethod
    def _graph_aware_select(scored: Sequence[RankedContext], top_k: int) -> list[RankedContext]:
        """Keep the reranker's order but reserve a few slots for graph neighbours (e.g. the RPC
        server behind a client stub) of the top results, which cross-encoders tend to under-rate."""
        head = list(scored[: max(1, top_k - GRAPH_RESERVED_SLOTS)])
        head_ids = {c.symbol_id for c in head}
        top_ids = {c.symbol_id for c in head[:3]}
        reserved = [c for c in scored[len(head) :] if c.via_graph_from is not None and c.via_graph_from in top_ids][
            :GRAPH_RESERVED_SLOTS
        ]
        chosen = head + reserved
        chosen_ids = head_ids | {c.symbol_id for c in reserved}
        for c in scored:
            if len(chosen) >= top_k:
                break
            if c.symbol_id not in chosen_ids:
                chosen.append(c)
                chosen_ids.add(c.symbol_id)
        return chosen[:top_k]

    async def synthesize(self, state: AgentState) -> AgentState:
        attempt = state.get("attempts", 0) + 1
        _emit({"type": "stage", "stage": "synthesize", "attempt": attempt})
        d = self._d
        contexts = state.get("contexts", [])
        if not contexts:
            return {
                "answer": "INSUFFICIENT_CONTEXT: no indexed code matched the question.",
                "attempts": attempt,
                "abstained": True,
            }
        parts: list[str] = []
        with d.tracer.span(
            "synthesize",
            input={"attempt": attempt, "feedback": state.get("feedback")},
            as_type="generation",
            model=d.synthesizer.model_name,
        ) as span:
            try:
                async for delta in d.synthesizer.stream(state["question"], contexts, state.get("feedback")):
                    parts.append(delta)
                    _emit({"type": "token", "attempt": attempt, "text": delta})
            except ProviderRefusalError as exc:
                span.set_attribute("refusal", str(exc))
                return {"answer": REFUSAL_MESSAGE, "attempts": attempt, "abstained": True}
            answer = "".join(parts).strip()
            span.set_output(answer)
        return {"answer": answer, "attempts": attempt, "abstained": is_abstention(answer)}

    async def verify(self, state: AgentState) -> AgentState:
        _emit({"type": "stage", "stage": "verify"})
        d = self._d
        with d.tracer.span("verify_citations") as span:
            report = d.verifier.verify(state.get("answer", ""), state.get("contexts", []))
            span.set_output(
                {
                    "passed": report.passed,
                    "faithfulness": report.faithfulness,
                    "failures": len(report.failures()),
                }
            )
        feedback = None if report.passed else CitationVerifier.feedback(report)
        if feedback and not state.get("abstained"):
            _emit({"type": "retry", "attempt": state.get("attempts", 1), "reason": feedback})
        return {"report": report, "feedback": feedback}

    async def finalize(self, state: AgentState) -> AgentState:
        _emit({"type": "stage", "stage": "finalize"})
        report = state.get("report")
        answer = state.get("answer", "")
        if state.get("abstained") or report is None or report.passed:
            return {}
        pruned = CitationVerifier.strip_unverified(answer, report)
        new_report = self._d.verifier.verify(pruned, state.get("contexts", []))
        if not new_report.passed:
            return {
                "answer": UNVERIFIABLE_MESSAGE,
                "abstained": True,
                "report": VerificationReport(checks=(), faithfulness=0.0),
            }
        return {
            "answer": pruned + "\n\n_Some statements were removed because their citations could not be verified._",
            "report": new_report,
        }

    # ------------------------------------------------------------------ public API
    def _initial(self, query: SearchQuery) -> AgentState:
        return {"question": query.text, "query": query, "attempts": 0, "feedback": None, "abstained": False}

    async def search(self, query: SearchQuery) -> list[RankedContext]:
        """Retrieval only (no LLM synthesis): used by `code-rag search` and retrieval metrics."""
        state: AgentState = self._initial(query)
        for node in (self.expand_query, self.retrieve_hybrid, self.fuse_rrf, self.expand_graph, self.rerank):
            update = await node(state)
            state.update(update)
        return state.get("contexts", [])

    async def ask(self, query: SearchQuery) -> CitedAnswer:
        answer, _ = await self.ask_detailed(query)
        return answer

    async def ask_detailed(self, query: SearchQuery) -> tuple[CitedAnswer, list[RankedContext]]:
        """Answer plus the final reranked contexts (used by the evaluator for ranking metrics)."""
        with (
            collect_timings() as timings,
            self._d.tracer.span("ask", input=query.text, as_type="agent") as span,
        ):
            t0 = time.perf_counter()
            final: AgentState = await self._graph.ainvoke(self._initial(query))
            timings["total"] = round((time.perf_counter() - t0) * 1000, 2)
            answer = self._to_answer(query, final, dict(timings))
            span.set_output({"passed": answer.report.passed, "attempts": answer.attempts})
            answer = answer.model_copy(update={"trace_id": self._d.tracer.current_trace_id})
        self._d.tracer.flush()
        return answer, list(final.get("contexts", []))

    async def stream(self, query: SearchQuery) -> AsyncIterator[dict[str, Any]]:
        """Yields stage/token/retry events, then a final {'type': 'final', 'answer': CitedAnswer}."""
        with collect_timings() as timings, self._d.tracer.span("ask", input=query.text, as_type="agent"):
            t0 = time.perf_counter()
            final: AgentState = self._initial(query)
            async for mode, chunk in self._graph.astream(self._initial(query), stream_mode=["custom", "values"]):
                if mode == "custom":
                    yield chunk
                else:
                    final = chunk
            timings["total"] = round((time.perf_counter() - t0) * 1000, 2)
            answer = self._to_answer(query, final, dict(timings))
            answer = answer.model_copy(update={"trace_id": self._d.tracer.current_trace_id})
        self._d.tracer.flush()
        yield {"type": "final", "answer": answer}

    @staticmethod
    def _to_answer(query: SearchQuery, state: AgentState, timings: dict[str, float]) -> CitedAnswer:
        report = state.get("report") or VerificationReport(checks=(), faithfulness=0.0)
        return CitedAnswer(
            question=query.text,
            answer_markdown=state.get("answer", ""),
            report=report,
            attempts=max(1, state.get("attempts", 1)),
            abstained=bool(state.get("abstained", False)),
            timings_ms=timings,
        )
