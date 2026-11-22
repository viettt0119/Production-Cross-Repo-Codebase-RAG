"""Incremental re-indexing benchmark: replays the last N commits of a repo one by one and checks the
freshness SLO (default: commits touching <= 50 files must be indexed in < 60 s)."""

from __future__ import annotations

from pydantic import BaseModel

from code_rag.application.evaluation.metrics import latency_summary
from code_rag.application.ingest.workflow import IncrementalIngestWorkflow
from code_rag.domain.changeset import IngestStats
from code_rag.domain.ports.source import SourceRepository


class CommitTiming(BaseModel):
    sha: str
    files_changed: int
    symbols_embedded: int
    symbols_relocated: int
    symbols_deleted: int
    total_ms: float
    stages_ms: dict[str, float]


class IngestBenchReport(BaseModel):
    repo: str
    base_sha: str
    commits: list[CommitTiming]
    latency_ms: dict[str, float]
    slo_files: int
    slo_seconds: float
    slo_commits: int
    slo_violations: list[str]

    @property
    def slo_met(self) -> bool:
        return not self.slo_violations


class IngestBenchmark:
    def __init__(self, workflow: IncrementalIngestWorkflow, source: SourceRepository) -> None:
        self._wf = workflow
        self._source = source

    async def run(self, repo: str, n_commits: int, slo_files: int = 50, slo_seconds: float = 60.0) -> IngestBenchReport:
        base = await self._source.head(repo, f"HEAD~{n_commits}")
        tip = await self._source.head(repo, "HEAD")
        await self._wf.run(repo, to_sha=base, from_sha=None, force_full=True)
        commits = await self._source.commits_between(repo, base, tip)
        timings: list[CommitTiming] = []
        for sha in commits:
            stats: IngestStats = await self._wf.run(repo, to_sha=sha)
            timings.append(
                CommitTiming(
                    sha=sha,
                    files_changed=stats.files_changed,
                    symbols_embedded=stats.symbols_embedded,
                    symbols_relocated=stats.symbols_relocated,
                    symbols_deleted=stats.symbols_deleted,
                    total_ms=stats.timings_ms.get("total", 0.0),
                    stages_ms=stats.timings_ms,
                )
            )
        in_scope = [t for t in timings if t.files_changed <= slo_files]
        violations = [t.sha for t in in_scope if t.total_ms > slo_seconds * 1000]
        return IngestBenchReport(
            repo=repo,
            base_sha=base,
            commits=timings,
            latency_ms=latency_summary([t.total_ms for t in timings]),
            slo_files=slo_files,
            slo_seconds=slo_seconds,
            slo_commits=len(in_scope),
            slo_violations=violations,
        )
