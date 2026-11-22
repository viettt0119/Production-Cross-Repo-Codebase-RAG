"""FastAPI application: /api/v1/ask (JSON or SSE), /api/v1/search, /webhooks/git, health."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from code_rag.domain.answer import CitedAnswer
from code_rag.domain.errors import CodeRagError, WebhookAuthError
from code_rag.infrastructure.container import Container
from code_rag.infrastructure.git.webhooks import authenticate, parse_push
from code_rag.infrastructure.settings import Settings
from code_rag.interfaces.api.schemas import ContextOut, QueryRequest, SearchResponse, WebhookAccepted

log = logging.getLogger(__name__)


class JobQueue(Protocol):
    async def enqueue_ingest(self, repo: str, to_sha: str) -> str | None:
        """Schedule an ingest of `repo` up to `to_sha`; returns a job id, or None for a duplicate."""

    async def close(self) -> None:
        """Release queue resources."""


class ArqQueue:
    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._pool: Any = None

    async def _get(self) -> Any:
        if self._pool is None:
            from arq import create_pool
            from arq.connections import RedisSettings

            self._pool = await create_pool(RedisSettings.from_dsn(self._url))
        return self._pool

    async def enqueue_ingest(self, repo: str, to_sha: str) -> str | None:
        pool = await self._get()
        job = await pool.enqueue_job("ingest_repo", repo, to_sha, _job_id=f"ingest:{repo}:{to_sha}")
        return job.job_id if job is not None else None  # None => duplicate delivery, already queued

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()


class InlineQueue:
    """Single-process mode (role=all): run ingests in the API process, serialized per repo."""

    def __init__(self, container: Container) -> None:
        self._c = container
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def enqueue_ingest(self, repo: str, to_sha: str) -> str | None:
        async def run() -> None:
            async with self._locks.setdefault(repo, asyncio.Lock()):
                try:
                    stats = await self._c.ingest_workflow.run(repo, to_sha)
                    log.info("ingested %s@%s: %s", repo, to_sha[:10], stats.timings_ms)
                except CodeRagError:
                    log.exception("inline ingest failed for %s@%s", repo, to_sha)

        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return f"inline:{repo}:{to_sha}"

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()


def _answer_json(answer: CitedAnswer) -> dict[str, Any]:
    data = answer.model_dump(mode="json")
    data["passed"] = answer.report.passed
    return data


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or Container(settings or Settings())
        app.state.container = c
        app.state.queue = ArqQueue(c.settings.redis_url) if c.settings.role == "api" else InlineQueue(c)
        try:
            yield
        finally:
            await app.state.queue.close()
            await c.aclose()

    app = FastAPI(title="code-rag", version="0.1.0", lifespan=lifespan)

    def ctr(request: Request) -> Container:
        c: Container = request.app.state.container
        return c

    @app.exception_handler(CodeRagError)
    async def domain_error(_: Request, exc: CodeRagError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": type(exc).__name__, "detail": str(exc)})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        c = ctr(request)
        try:
            await c.vectors.ensure_ready()
            count = await c.vectors.count()
        except Exception as exc:  # readiness must report, not raise
            raise HTTPException(status_code=503, detail=f"vector store unavailable: {exc}") from exc
        return {"status": "ready", "chunks": count, "profile": c.settings.profile, "role": c.settings.role}

    @app.post("/api/v1/search", response_model=SearchResponse)
    async def search(body: QueryRequest, request: Request) -> SearchResponse:
        c = ctr(request)
        await c.vectors.ensure_ready()
        contexts = await c.retrieval_agent.search(body.to_query())
        return SearchResponse(results=[ContextOut.from_context(x) for x in contexts])

    @app.post("/api/v1/ask", response_model=None)
    async def ask(body: QueryRequest, request: Request) -> Any:
        c = ctr(request)
        await c.vectors.ensure_ready()
        wants_sse = body.stream or "text/event-stream" in request.headers.get("accept", "")
        if not wants_sse:
            return _answer_json(await c.retrieval_agent.ask(body.to_query()))

        async def events() -> AsyncIterator[dict[str, str]]:
            async for ev in c.retrieval_agent.stream(body.to_query()):
                if await request.is_disconnected():
                    return
                if ev["type"] == "final":
                    yield {"event": "final", "data": json.dumps(_answer_json(ev["answer"]))}
                else:
                    yield {
                        "event": str(ev["type"]),
                        "data": json.dumps({k: v for k, v in ev.items() if k != "type"}),
                    }

        return EventSourceResponse(events(), ping=15)

    @app.post("/webhooks/git", response_model=WebhookAccepted, status_code=202)
    async def git_webhook(request: Request) -> WebhookAccepted:
        c = ctr(request)
        body = await request.body()
        wh = c.settings.webhooks
        try:
            authenticate(
                request.headers,
                body,
                wh.github_secret.get_secret_value() if wh.github_secret else None,
                wh.gitlab_secret.get_secret_value() if wh.gitlab_secret else None,
            )
            event = parse_push(request.headers, body)
        except WebhookAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        if event is None:
            return WebhookAccepted(queued=False, reason="ignored event type")
        known = {r.name: r for r in c.repos.repos}
        if event.repo not in known:
            return WebhookAccepted(queued=False, reason=f"repository {event.repo} is not configured")
        branch = wh.allowed_branches.get(event.repo, known[event.repo].default_branch)
        if event.branch != branch or event.is_branch_delete:
            return WebhookAccepted(queued=False, reason=f"ignored ref {event.ref}")
        job_id = await request.app.state.queue.enqueue_ingest(event.repo, event.after)
        return WebhookAccepted(
            queued=job_id is not None, job_id=job_id, reason=None if job_id else "duplicate delivery"
        )

    @app.get("/api/v1/repos")
    async def repos(request: Request) -> list[dict[str, Any]]:
        c = ctr(request)
        return [
            {"repo": r.name, "default_branch": r.default_branch, "indexed_sha": c.ledger.indexed_sha(r.name)}
            for r in c.repos.repos
        ]

    @app.get("/api/v1/runs")
    async def runs(request: Request, limit: int = 20) -> list[dict[str, Any]]:
        return ctr(request).ledger.recent_runs(min(limit, 200))

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn code_rag.interfaces.api.app:app_factory --factory`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app()
