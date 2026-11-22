"""Benchmark evaluator: retrieval quality, citation faithfulness and latency over a question set."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from code_rag.application.evaluation.metrics import (
    latency_summary,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from code_rag.application.retrieval.agent import HybridRetrievalAgent
from code_rag.domain.answer import CitedAnswer, ClaimVerdict
from code_rag.domain.entities import SymbolNode
from code_rag.domain.errors import ProviderError
from code_rag.domain.ports.models import EntailmentJudge
from code_rag.domain.query import RankedContext, SearchQuery

log = logging.getLogger(__name__)
K = 10


class GoldItem(BaseModel):
    """A relevant symbol, identified by qualified name and/or file location."""

    repo: str
    qualified_name: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    grade: Annotated[int, Field(ge=1, le=3)] = 1

    def matches(self, sym: SymbolNode) -> bool:
        if sym.coord.repo != self.repo:
            return False
        if self.qualified_name is not None:
            return sym.qualified_name == self.qualified_name or sym.qualified_name.endswith("." + self.qualified_name)
        if self.path is not None and sym.coord.path == self.path:
            if self.start_line is None:
                return True
            end = self.end_line or self.start_line
            return sym.span.start <= end and self.start_line <= sym.span.end
        return False


class EvalQuestion(BaseModel):
    id: str
    question: str
    gold: list[GoldItem]
    category: str = "general"
    repos: list[str] = Field(default_factory=list)


class QuestionResult(BaseModel):
    id: str
    category: str
    mrr: float
    ndcg: float
    recall: float
    retrieved: list[str]
    faithfulness: float | None = None
    judge_faithfulness: float | None = None
    passed: bool | None = None
    abstained: bool | None = None
    attempts: int | None = None
    latency_ms: float
    stages_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class EvalReport(BaseModel):
    mode: Literal["retrieval", "full"]
    n_questions: int
    mrr_at_10: float
    ndcg_at_10: float
    recall_at_10: float
    citation_faithfulness: float | None
    judge_faithfulness: float | None
    verification_pass_rate: float | None
    abstain_rate: float | None
    latency_ms: dict[str, float]
    stage_latency_ms: dict[str, dict[str, float]]
    by_category: dict[str, dict[str, float]]
    errors: int
    config: dict[str, object] = Field(default_factory=dict)
    results: list[QuestionResult]

    def headline(self) -> str:
        parts = [
            f"MRR@10={self.mrr_at_10:.3f}",
            f"nDCG@10={self.ndcg_at_10:.3f}",
            f"R@10={self.recall_at_10:.3f}",
        ]
        if self.citation_faithfulness is not None:
            parts.append(f"CitationFaithfulness={self.citation_faithfulness:.3f}")
        if self.judge_faithfulness is not None:
            parts.append(f"JudgeFaithfulness={self.judge_faithfulness:.3f}")
        lat = self.latency_ms
        parts.append(f"latency p50/p95/p99={lat['p50']:.0f}/{lat['p95']:.0f}/{lat['p99']:.0f}ms")
        return "  ".join(parts)


def load_questions(path: str | Path) -> list[EvalQuestion]:
    out: list[EvalQuestion] = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            out.append(EvalQuestion.model_validate_json(line))
    return out


def score_ranking(contexts: Sequence[RankedContext], gold: Sequence[GoldItem]) -> tuple[float, float, float]:
    relevance: list[int] = []
    gains: list[float] = []
    matched_gold: set[int] = set()
    for ctx in contexts[:K]:
        hit = next((i for i, g in enumerate(gold) if i not in matched_gold and g.matches(ctx.chunk.symbol)), None)
        if hit is None:
            relevance.append(0)
            gains.append(0.0)
        else:
            matched_gold.add(hit)
            relevance.append(1)
            gains.append(float(gold[hit].grade))
    return (
        reciprocal_rank(relevance, K),
        ndcg_at_k(gains, [float(g.grade) for g in gold], K),
        recall_at_k(len(matched_gold), len(gold)),
    )


def _evidence(answer: CitedAnswer, contexts: Sequence[RankedContext]) -> list[tuple[str, dict[str, str]]]:
    """(claim text, {anchor: cited source lines}) for every verified claim."""
    out: list[tuple[str, dict[str, str]]] = []
    for check in answer.report.checks:
        if check.verdict is not ClaimVerdict.VERIFIED or not check.claim.citations:
            continue
        evidence: dict[str, str] = {}
        for c in check.claim.citations:
            for ctx in contexts:
                s = ctx.chunk.symbol
                if s.coord.repo == c.repo and s.coord.path == c.path and s.span.overlaps(c.span):
                    lines = s.source.split("\n")
                    lo = max(0, c.span.start - s.span.start)
                    hi = min(len(lines), c.span.end - s.span.start + 1)
                    evidence[c.render()] = "\n".join(lines[lo:hi]) or s.source
        if evidence:
            out.append((check.claim.text, evidence))
    return out


class BenchmarkEvaluator:
    def __init__(self, agent: HybridRetrievalAgent, judge: EntailmentJudge | None = None, concurrency: int = 4) -> None:
        self._agent = agent
        self._judge = judge
        self._sem = asyncio.Semaphore(concurrency)

    async def _one(self, q: EvalQuestion, mode: Literal["retrieval", "full"]) -> QuestionResult:
        query = SearchQuery(text=q.question, repos=tuple(q.repos))
        async with self._sem:
            t0 = time.perf_counter()
            try:
                if mode == "retrieval":
                    contexts = await self._agent.search(query)
                    answer = None
                else:
                    answer, contexts = await self._agent.ask_detailed(query)
            except ProviderError as exc:
                return QuestionResult(
                    id=q.id,
                    category=q.category,
                    mrr=0,
                    ndcg=0,
                    recall=0,
                    retrieved=[],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error=str(exc),
                )
            latency = (time.perf_counter() - t0) * 1000
        mrr, ndcg, recall = score_ranking(contexts, q.gold)
        result = QuestionResult(
            id=q.id,
            category=q.category,
            mrr=mrr,
            ndcg=ndcg,
            recall=recall,
            retrieved=[f"{c.chunk.symbol.coord.repo}:{c.chunk.symbol.qualified_name}" for c in contexts[:K]],
            latency_ms=round(latency, 1),
        )
        if answer is not None:
            result.faithfulness = answer.report.faithfulness if not answer.abstained else None
            result.passed = answer.report.passed
            result.abstained = answer.abstained
            result.attempts = answer.attempts
            result.stages_ms = answer.timings_ms
            if self._judge is not None and not answer.abstained:
                result.judge_faithfulness = await self._judge_answer(answer, contexts)
        return result

    async def _judge_answer(self, answer: CitedAnswer, contexts: Sequence[RankedContext]) -> float | None:
        assert self._judge is not None
        items = _evidence(answer, contexts)
        total_claims = len(answer.report.checks)
        if not total_claims:
            return None
        verdicts = await asyncio.gather(
            *(self._judge.supports(claim, ev) for claim, ev in items), return_exceptions=True
        )
        supported = sum(1 for v in verdicts if v is True)
        return round(supported / total_claims, 4)

    async def run(
        self,
        questions: Sequence[EvalQuestion],
        mode: Literal["retrieval", "full"] = "full",
        config: dict[str, object] | None = None,
    ) -> EvalReport:
        results = await asyncio.gather(*(self._one(q, mode) for q in questions))
        ok = [r for r in results if r.error is None]
        by_cat: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in ok:
            by_cat[r.category].append(r)
        stage_values: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            for stage, ms in r.stages_ms.items():
                stage_values[stage].append(ms)
        faith = [r.faithfulness for r in ok if r.faithfulness is not None]
        judged = [r.judge_faithfulness for r in ok if r.judge_faithfulness is not None]
        full = mode == "full"
        return EvalReport(
            mode=mode,
            n_questions=len(questions),
            mrr_at_10=mean([r.mrr for r in ok]),
            ndcg_at_10=mean([r.ndcg for r in ok]),
            recall_at_10=mean([r.recall for r in ok]),
            citation_faithfulness=mean(faith) if full and faith else (None if not full else 0.0),
            judge_faithfulness=mean(judged) if judged else None,
            verification_pass_rate=mean([1.0 if r.passed else 0.0 for r in ok]) if full else None,
            abstain_rate=mean([1.0 if r.abstained else 0.0 for r in ok]) if full else None,
            latency_ms=latency_summary([r.latency_ms for r in ok]),
            stage_latency_ms={k: latency_summary(v) for k, v in sorted(stage_values.items())},
            by_category={
                cat: {
                    "n": float(len(rs)),
                    "mrr": mean([r.mrr for r in rs]),
                    "ndcg": mean([r.ndcg for r in rs]),
                    "recall": mean([r.recall for r in rs]),
                }
                for cat, rs in sorted(by_cat.items())
            },
            errors=len(results) - len(ok),
            config=config or {},
            results=list(results),
        )


def write_report(report: EvalReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
