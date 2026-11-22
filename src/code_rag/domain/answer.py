"""Answer & citation value objects."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, ClassVar

from pydantic import Field

from code_rag.domain.value_objects import FrozenModel, LineSpan

# (org/repo/path/to/file.py:12-40) — repo is always the first two path segments.
CITATION_PATTERN: re.Pattern[str] = re.compile(
    r"\((?P<repo>[\w.-]+/[\w.-]+)/(?P<path>[^\s:()]+):(?P<start>\d+)(?:-(?P<end>\d+))?\)"
)


class Citation(FrozenModel):
    repo: str
    path: str
    span: LineSpan

    PATTERN: ClassVar[re.Pattern[str]] = CITATION_PATTERN

    def render(self) -> str:
        return f"({self.repo}/{self.path}:{self.span.start}-{self.span.end})"

    @classmethod
    def parse_all(cls, text: str) -> list[Citation]:
        out: list[Citation] = []
        for m in CITATION_PATTERN.finditer(text):
            start = int(m.group("start"))
            end = int(m.group("end") or start)
            if start < 1 or end < start:
                continue
            out.append(cls(repo=m.group("repo"), path=m.group("path"), span=LineSpan(start=start, end=end)))
        return out


class CitedClaim(FrozenModel):
    text: str
    citations: tuple[Citation, ...]


class ClaimVerdict(StrEnum):
    VERIFIED = "verified"
    MISSING_CITATION = "missing_citation"
    UNKNOWN_FILE = "unknown_file"
    SPAN_OUT_OF_RANGE = "span_out_of_range"
    NOT_IN_CONTEXT = "not_in_context"


class ClaimCheck(FrozenModel):
    claim: CitedClaim
    verdict: ClaimVerdict
    detail: str = ""


class VerificationReport(FrozenModel):
    checks: tuple[ClaimCheck, ...]
    faithfulness: Annotated[float, Field(ge=0.0, le=1.0)]
    cited_symbol_ids: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.verdict is ClaimVerdict.VERIFIED for c in self.checks)

    def failures(self) -> list[ClaimCheck]:
        return [c for c in self.checks if c.verdict is not ClaimVerdict.VERIFIED]


class CitedAnswer(FrozenModel):
    question: str
    answer_markdown: str
    report: VerificationReport
    attempts: Annotated[int, Field(ge=1)]
    abstained: bool = False
    trace_id: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
