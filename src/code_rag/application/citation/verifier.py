"""Citation verification: every factual sentence must carry a `(org/repo/path:start-end)` anchor
that (1) names an indexed file, (2) lies within its line count, and (3) overlaps a code chunk that
was actually given to the model. Code snippets must appear verbatim in the retrieved sources."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from code_rag.domain.answer import (
    Citation,
    CitedClaim,
    ClaimCheck,
    ClaimVerdict,
    VerificationReport,
)
from code_rag.domain.ports.indexing import FileLineIndex
from code_rag.domain.query import RankedContext

ABSTAIN_PREFIX = "INSUFFICIENT_CONTEXT"
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*_\[(\"'])")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TRIVIAL_CODE = re.compile(r"^(?:\.\.\.|…|⋯.*|//\s*\.\.\.|#\s*\.\.\.|[{}()\[\];,]+)$")
_MIN_CLAIM_WORDS = 4


@dataclass(frozen=True, slots=True)
class _Unit:
    sentences: list[str]


def _normalize_code_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _drop_block_containing(text: str) -> Callable[[re.Match[str]], str]:
    def repl(m: re.Match[str]) -> str:
        return "" if text in m.group(1) else m.group(0)

    return repl


def is_abstention(answer: str) -> bool:
    return answer.lstrip().startswith(ABSTAIN_PREFIX)


class CitationVerifier:
    def __init__(self, line_index: FileLineIndex, require_in_context: bool = True) -> None:
        self._lines = line_index
        self._require_in_context = require_in_context

    # ---- public --------------------------------------------------------------------
    def verify(self, answer: str, contexts: Sequence[RankedContext]) -> VerificationReport:
        if is_abstention(answer):
            return VerificationReport(checks=(), faithfulness=1.0)
        checks: list[ClaimCheck] = []
        checks.extend(self._check_code_blocks(answer, contexts))
        prose = _FENCE.sub("\n", answer)
        for unit in self._units(prose):
            checks.extend(self._check_unit(unit, contexts))
        verified = sum(1 for c in checks if c.verdict is ClaimVerdict.VERIFIED)
        faithfulness = verified / len(checks) if checks else 0.0
        cited = self._cited_symbols(checks, contexts)
        return VerificationReport(checks=tuple(checks), faithfulness=round(faithfulness, 4), cited_symbol_ids=cited)

    @staticmethod
    def feedback(report: VerificationReport, max_items: int = 12) -> str:
        lines = []
        for c in report.failures()[:max_items]:
            snippet = c.claim.text.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(f'- [{c.verdict.value}] "{snippet}" {c.detail}'.rstrip())
        return "\n".join(lines)

    @staticmethod
    def strip_unverified(answer: str, report: VerificationReport) -> str:
        """Remove sentences/snippets that failed verification (last-resort guardrail)."""
        out = answer
        for c in report.failures():
            text = c.claim.text.strip()
            if not text:
                continue
            if c.detail.startswith("code snippet"):
                out = _FENCE.sub(_drop_block_containing(text), out)
            else:
                out = out.replace(text, "")
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)])\s*$\n?", "", out)
        return out.strip()

    # ---- prose -----------------------------------------------------------------------
    @staticmethod
    def _units(prose: str) -> list[_Unit]:
        units: list[_Unit] = []
        paragraph: list[str] = []

        def flush() -> None:
            if paragraph:
                text = " ".join(paragraph).strip()
                if text:
                    units.append(_Unit([s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]))
                paragraph.clear()

        for raw in prose.split("\n"):
            line = raw.rstrip()
            if not line.strip():
                flush()
                continue
            if line.lstrip().startswith("#") or line.strip().startswith("|"):
                flush()
                continue  # headings and table rows are structure, not claims
            if _BULLET.match(line):
                flush()
                paragraph.append(_BULLET.sub("", line, count=1))
                flush()
                continue
            paragraph.append(line.strip())
        flush()
        return units

    @staticmethod
    def _is_claim(sentence: str) -> bool:
        if sentence.endswith(":"):
            return False
        words = re.sub(Citation.PATTERN, "", sentence).split()
        return len(words) >= _MIN_CLAIM_WORDS

    def _check_unit(self, unit: _Unit, contexts: Sequence[RankedContext]) -> list[ClaimCheck]:
        cites = [Citation.parse_all(s) for s in unit.sentences]
        out: list[ClaimCheck] = []
        for i, sentence in enumerate(unit.sentences):
            own = cites[i]
            if not self._is_claim(sentence) and not own:
                continue
            effective = own
            if not effective:
                # citation-at-end-of-paragraph style: inherit the next cited sentence's anchors
                for j in range(i + 1, len(unit.sentences)):
                    if cites[j]:
                        effective = cites[j]
                        break
            claim = CitedClaim(text=sentence, citations=tuple(effective))
            if not effective:
                out.append(
                    ClaimCheck(
                        claim=claim,
                        verdict=ClaimVerdict.MISSING_CITATION,
                        detail="needs a (org/repo/path:start-end) citation",
                    )
                )
                continue
            out.append(self._check_citations(claim, contexts))
        return out

    def _check_citations(self, claim: CitedClaim, contexts: Sequence[RankedContext]) -> ClaimCheck:
        any_ok = False
        worst: tuple[ClaimVerdict, str] | None = None
        for c in claim.citations:
            verdict, detail = self._check_one(c, contexts)
            if verdict is ClaimVerdict.VERIFIED:
                any_ok = True
            elif verdict in (ClaimVerdict.UNKNOWN_FILE, ClaimVerdict.SPAN_OUT_OF_RANGE):
                worst = (verdict, detail)  # fabricated anchors are hard failures
            elif worst is None:
                worst = (verdict, detail)
        if any_ok and (worst is None or worst[0] is ClaimVerdict.NOT_IN_CONTEXT):
            return ClaimCheck(claim=claim, verdict=ClaimVerdict.VERIFIED)
        assert worst is not None
        return ClaimCheck(claim=claim, verdict=worst[0], detail=worst[1])

    def _check_one(self, c: Citation, contexts: Sequence[RankedContext]) -> tuple[ClaimVerdict, str]:
        total = self._lines.line_count(c.repo, c.path)
        if total is None:
            return ClaimVerdict.UNKNOWN_FILE, f"{c.render()} is not an indexed file"
        if c.span.end > total:
            return ClaimVerdict.SPAN_OUT_OF_RANGE, f"{c.render()} exceeds file length ({total} lines)"
        if self._require_in_context and not any(
            ctx.chunk.symbol.coord.repo == c.repo
            and ctx.chunk.symbol.coord.path == c.path
            and ctx.chunk.symbol.span.overlaps(c.span)
            for ctx in contexts
        ):
            return ClaimVerdict.NOT_IN_CONTEXT, f"{c.render()} does not overlap any provided source block"
        return ClaimVerdict.VERIFIED, ""

    # ---- code blocks -----------------------------------------------------------------
    def _check_code_blocks(self, answer: str, contexts: Sequence[RankedContext]) -> list[ClaimCheck]:
        known: set[str] = set()
        for ctx in contexts:
            for line in ctx.chunk.symbol.source.split("\n"):
                norm = _normalize_code_line(line)
                if norm:
                    known.add(norm)
        out: list[ClaimCheck] = []
        for m in _FENCE.finditer(answer):
            block = m.group(1)
            lines = [_normalize_code_line(x) for x in block.split("\n")]
            lines = [x for x in lines if x and not _TRIVIAL_CODE.match(x)]
            if not lines:
                continue
            missing = [x for x in lines if x not in known]
            claim = CitedClaim(text=lines[0], citations=())
            if missing:
                out.append(
                    ClaimCheck(
                        claim=claim,
                        verdict=ClaimVerdict.NOT_IN_CONTEXT,
                        detail=f"code snippet line not found verbatim in sources: {missing[0][:80]!r}",
                    )
                )
            else:
                out.append(ClaimCheck(claim=claim, verdict=ClaimVerdict.VERIFIED))
        return out

    @staticmethod
    def _cited_symbols(checks: Sequence[ClaimCheck], contexts: Sequence[RankedContext]) -> tuple[str, ...]:
        ids: list[str] = []
        for check in checks:
            if check.verdict is not ClaimVerdict.VERIFIED:
                continue
            for c in check.claim.citations:
                for ctx in contexts:
                    sym = ctx.chunk.symbol
                    if (
                        sym.coord.repo == c.repo
                        and sym.coord.path == c.path
                        and sym.span.overlaps(c.span)
                        and sym.symbol_id not in ids
                    ):
                        ids.append(sym.symbol_id)
        return tuple(ids)
