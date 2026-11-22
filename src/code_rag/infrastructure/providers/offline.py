"""Deterministic, dependency-free adapters for the `offline` profile.

They make the full pipeline runnable without API keys or GPUs (CI, demos, eval smoke tests)
while exercising exactly the same ports as the cloud/local adapters.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence

from code_rag.domain.ports.models import (
    FileToSummarize,
    InputType,
    QueryExpansion,
    QuestionSeed,
    RerankResult,
)
from code_rag.domain.query import RankedContext
from code_rag.domain.value_objects import SymbolId
from code_rag.infrastructure.text import code_tokens, identifiers_in


class HashingEmbeddingProvider:
    """Feature-hashed bag of code sub-tokens + bigrams, log-TF, L2-normalised."""

    def __init__(self, dimension: int = 512) -> None:
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        toks = code_tokens(text)
        feats = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in itertools.pairwise(toks))
        vec = [0.0] * self._dim
        for feat, tf in feats.items():
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            idx = h % self._dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(tf))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        return [self._vector(t) for t in texts]


_DOC_PATTERNS = (
    re.compile(r'"""\s*(.+?)(?:\n|""")', re.DOTALL),
    re.compile(r"'''\s*(.+?)(?:\n|''')", re.DOTALL),
    re.compile(r"^\s*(?://+|///|#(?![\[!]))\s*(.+)$", re.MULTILINE),
    re.compile(r"/\*\*\s*\n?\s*\*?\s*(.+?)(?:\n|\*/)", re.DOTALL),
)


class HeuristicSummarizer:
    """Uses the symbol's own doc comment / docstring, else its signature."""

    @property
    def model_name(self) -> str:
        return "heuristic-docstring"

    @staticmethod
    def _doc(source: str, file_lines: Sequence[str], start_line: int) -> str | None:
        for pat in _DOC_PATTERNS[:2]:
            m = pat.search(source)
            if m:
                return m.group(1).strip()
        # leading comments directly above the symbol
        above: list[str] = []
        i = start_line - 2
        while i >= 0 and file_lines[i].strip().startswith(("//", "#", "///", "*", "/**")):
            text = file_lines[i].strip().lstrip("/#*").strip()
            if text:
                above.insert(0, text)
            i -= 1
        if above:
            return " ".join(above)
        for pat in _DOC_PATTERNS[2:]:
            m = pat.search(source)
            if m:
                return m.group(1).strip()
        return None

    async def summarize_file(self, item: FileToSummarize) -> dict[SymbolId, str]:
        lines = item.file_source.split("\n")
        out: dict[SymbolId, str] = {}
        for sym in item.symbols:
            doc = self._doc(sym.source, lines, sym.span.start)
            text = (doc or sym.signature).replace("*/", "").strip()
            out[sym.symbol_id] = text[:300]
        return out


class LexicalOverlapReranker:
    """BM25-flavoured token overlap reranker (stand-in for a cross-encoder)."""

    @property
    def model_name(self) -> str:
        return "lexical-overlap"

    async def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[RerankResult]:
        q = Counter(code_tokens(query))
        if not q:
            return [RerankResult(index=i, score=0.0) for i in range(min(top_k, len(documents)))]
        doc_tokens = [Counter(code_tokens(d)) for d in documents]
        n = len(documents) or 1
        df: Counter[str] = Counter()
        for dt in doc_tokens:
            df.update(set(dt))
        avg_len = sum(sum(dt.values()) for dt in doc_tokens) / n or 1.0
        scored: list[RerankResult] = []
        for i, dt in enumerate(doc_tokens):
            dl = sum(dt.values()) or 1
            score = 0.0
            for term in q:
                tf = dt.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * dl / avg_len))
            scored.append(RerankResult(index=i, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class RuleBasedQueryExpander:
    async def expand(self, question: str) -> QueryExpansion:
        idents = identifiers_in(question)
        rewrite = " ".join(dict.fromkeys(code_tokens(question)))
        return QueryExpansion(rewrites=(rewrite,) if rewrite and rewrite != question else (), identifiers=tuple(idents))


class ExtractiveSynthesizer:
    """Builds a cited answer from the top contexts' summaries/signatures. Never invents content."""

    def __init__(self, max_items: int = 5) -> None:
        self._max = max_items

    @property
    def model_name(self) -> str:
        return "extractive"

    async def stream(
        self, question: str, contexts: Sequence[RankedContext], feedback: str | None
    ) -> AsyncIterator[str]:
        if not contexts:
            yield "INSUFFICIENT_CONTEXT: no indexed code matches this question."
            return
        yield "### Relevant code\n\n"
        for ctx in contexts[: self._max]:
            sym = ctx.chunk.symbol
            what = (ctx.chunk.summary or sym.signature).strip().rstrip(".")
            what = what.replace("\n", " ")
            yield f"- `{sym.qualified_name}` ({sym.kind.value}): {what} {sym.citation_anchor()}.\n"


class LexicalEntailmentJudge:
    """Offline judge: a claim is supported if most of its code-ish tokens occur in the cited evidence."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    async def supports(self, claim: str, evidence: Mapping[str, str]) -> bool:
        claim_toks = set(code_tokens(claim))
        if not claim_toks or not evidence:
            return False
        ev_toks: set[str] = set()
        for text in evidence.values():
            ev_toks.update(code_tokens(text))
        return len(claim_toks & ev_toks) / len(claim_toks) >= self._threshold


class TemplateQuestionWriter:
    """Offline question writer: phrases the pair's summaries into a question (no LLM)."""

    async def write(self, seed: QuestionSeed) -> str | None:
        a = (seed.src_summary or seed.src_name).rstrip(".")
        b = (seed.dst_summary or seed.dst_name).rstrip(".")
        if seed.relation.startswith("RPC"):
            return f"How does the client side ({a}) reach the server implementation ({b}) across services?"
        if seed.relation in ("INHERITS", "IMPLEMENTS"):
            return f"What contract does {seed.src_name.rsplit('.', 1)[-1]} inherit, and where is it defined ({b})?"
        return f"When {a}, which shared code from another repository does it rely on ({b})?"
