"""arq worker: the single writer of the hash ledger, Tantivy index and graph database.

Jobs:
  * ingest_repo(repo, to_sha)  — enqueued by webhooks; job id `ingest:<repo>:<sha>` dedupes redeliveries.
    Each run diffs from the ledger's *indexed* sha (not the webhook's `before`), so coalesced or
    out-of-order pushes are safe; a per-repo Redis lock serializes runs of the same repo.
  * sync_summaries()           — cron: applies finished Message Batches summary jobs.
"""

from __future__ import annotations

import logging
from typing import Any

from arq import cron
from arq.connections import RedisSettings

from code_rag.application.ingest.workflow import SummaryEnrichment
from code_rag.infrastructure.container import Container
from code_rag.infrastructure.settings import Settings

log = logging.getLogger(__name__)
REPO_LOCK_TIMEOUT_S = 1800


async def startup(ctx: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    if settings.role == "api":
        settings = settings.model_copy(update={"role": "worker"})
    container = Container(settings)
    await container.vectors.ensure_ready()
    ctx["container"] = container


async def shutdown(ctx: dict[str, Any]) -> None:
    container: Container | None = ctx.get("container")
    if container is not None:
        await container.aclose()


async def ingest_repo(ctx: dict[str, Any], repo: str, to_sha: str | None = None) -> dict[str, Any]:
    container: Container = ctx["container"]
    lock = ctx["redis"].lock(f"code-rag:lock:{repo}", timeout=REPO_LOCK_TIMEOUT_S, blocking_timeout=REPO_LOCK_TIMEOUT_S)
    async with lock:
        stats = await container.ingest_workflow.run(repo, to_sha)
    log.info(
        "ingest %s -> %s: embedded=%d relocated=%d deleted=%d total=%.0fms",
        repo,
        stats.to_sha[:10],
        stats.symbols_embedded,
        stats.symbols_relocated,
        stats.symbols_deleted,
        stats.timings_ms["total"],
    )
    return stats.model_dump(mode="json")


async def sync_summaries(ctx: dict[str, Any]) -> dict[str, int]:
    container: Container = ctx["container"]
    return await SummaryEnrichment(container.ingest_dependencies()).sync()


class WorkerSettings:
    functions = [ingest_repo, sync_summaries]
    cron_jobs = [cron(sync_summaries, minute=set(range(0, 60, 5)), run_at_startup=True)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(Settings().redis_url)
    max_jobs = 4
    job_timeout = 3600
    keep_result = 86_400
    max_tries = 3
