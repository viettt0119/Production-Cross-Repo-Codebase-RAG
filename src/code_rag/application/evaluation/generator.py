"""Generates cross-repo evaluation questions from the symbol graph.

Pairs of symbols connected across repository boundaries (resolved CALLS/INHERITS edges and the
gRPC client->endpoint<-server bridge) are turned into questions whose answer needs *both* symbols;
both become gold items. Use an LLM QuestionWriter for natural phrasing; review before trusting.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from pathlib import Path

from code_rag.application.evaluation.evaluator import EvalQuestion, GoldItem
from code_rag.domain.ports.indexing import SymbolGraph, VectorIndex
from code_rag.domain.ports.models import QuestionSeed, QuestionWriter
from code_rag.domain.value_objects import SymbolId

MAX_SOURCE_CHARS = 2500


class CrossRepoQuestionGenerator:
    def __init__(self, graph: SymbolGraph, vectors: VectorIndex, writer: QuestionWriter, seed: int = 7) -> None:
        self._graph = graph
        self._vectors = vectors
        self._writer = writer
        self._rng = random.Random(seed)

    async def generate(self, n: int, concurrency: int = 8) -> list[EvalQuestion]:
        pairs = await asyncio.to_thread(self._graph.cross_repo_pairs, n * 5)
        unique = list(dict.fromkeys(pairs))
        self._rng.shuffle(unique)
        chunks = await self._vectors.fetch(sorted({sid for a, b, _ in unique for sid in (a, b)}))
        sem = asyncio.Semaphore(concurrency)

        async def make(i: int, src: str, dst: str, relation: str) -> EvalQuestion | None:
            a, b = chunks.get(SymbolId(src)), chunks.get(SymbolId(dst))
            if a is None or b is None:
                return None
            if a.symbol.coord.repo == b.symbol.coord.repo and not relation.startswith("RPC"):
                return None
            seed = QuestionSeed(
                relation=relation,
                src_repo=a.symbol.coord.repo,
                src_name=a.symbol.qualified_name,
                src_summary=a.summary or "",
                src_source=a.symbol.source[:MAX_SOURCE_CHARS],
                dst_repo=b.symbol.coord.repo,
                dst_name=b.symbol.qualified_name,
                dst_summary=b.summary or "",
                dst_source=b.symbol.source[:MAX_SOURCE_CHARS],
            )
            async with sem:
                question = await self._writer.write(seed)
            if not question:
                return None
            return EvalQuestion(
                id=f"gen-{i:04d}",
                question=question,
                category="cross_repo_rpc" if relation.startswith("RPC") else f"cross_repo_{relation.lower()}",
                gold=[
                    GoldItem(repo=a.symbol.coord.repo, qualified_name=a.symbol.qualified_name, grade=2),
                    GoldItem(repo=b.symbol.coord.repo, qualified_name=b.symbol.qualified_name, grade=2),
                ],
            )

        made = await asyncio.gather(*(make(i, a, b, r) for i, (a, b, r) in enumerate(unique)))
        out = [q for q in made if q is not None]
        return out[:n]


def write_questions(questions: Sequence[EvalQuestion], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(q.model_dump(mode="json")) for q in questions) + "\n")
