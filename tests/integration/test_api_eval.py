from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from code_rag.application.evaluation.evaluator import BenchmarkEvaluator, load_questions
from code_rag.application.evaluation.generator import CrossRepoQuestionGenerator
from code_rag.application.evaluation.ingest_bench import IngestBenchmark
from code_rag.infrastructure.container import Container
from code_rag.infrastructure.settings import WebhookSettings
from code_rag.interfaces.api.app import create_app
from tests.conftest import CorpusRepos

pytestmark = pytest.mark.integration
QUESTIONS = Path(__file__).parents[2] / "eval" / "questions.fixture.jsonl"


async def _client(c: Container) -> httpx.AsyncClient:
    app = create_app(container=c)
    app.state.container = c
    from code_rag.interfaces.api.app import InlineQueue

    app.state.queue = InlineQueue(c)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_ask_json_and_sse(indexed: Container) -> None:
    async with await _client(indexed) as client:
        r = await client.post("/api/v1/ask", json={"question": "How are money strings parsed?"})
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is True and "acme/shared-py/acme_common/money.py" in body["answer_markdown"]

        r = await client.post("/api/v1/ask", json={"question": "How are money strings parsed?", "stream": True})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [line.removeprefix("event: ").strip() for line in r.text.splitlines() if line.startswith("event:")]
        assert "stage" in events and "token" in events and events[-1] == "final"

        r = await client.post("/api/v1/search", json={"question": "reserve stock inventory"})
        names = [x["qualified_name"] for x in r.json()["results"]]
        assert "checkout.inventory.Inventory.reserve" in names[:3]


async def test_webhook_auth_branch_filter_and_enqueue(indexed: Container, corpus: CorpusRepos) -> None:
    indexed.settings.webhooks = WebhookSettings(github_secret=SecretStr("s3cret"))
    corpus.write("acme/checkout-py", "checkout/extra.py", "def extra_fn():\n    return 42\n")
    sha = corpus.commit("acme/checkout-py")

    def gh(payload: dict[str, object], secret: str = "s3cret") -> tuple[bytes, dict[str, str]]:
        body = json.dumps(payload).encode()
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return body, {
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        }

    push = {
        "ref": "refs/heads/main",
        "before": "a" * 40,
        "after": sha,
        "repository": {"full_name": "acme/checkout-py"},
    }
    async with await _client(indexed) as client:
        body, headers = gh(push, secret="wrong")
        assert (await client.post("/webhooks/git", content=body, headers=headers)).status_code == 401
        body, headers = gh({**push, "ref": "refs/heads/feature"})
        r = await client.post("/webhooks/git", content=body, headers=headers)
        assert r.status_code == 202 and r.json()["queued"] is False
        body, headers = gh(push)
        r = await client.post("/webhooks/git", content=body, headers=headers)
        assert r.status_code == 202 and r.json()["queued"] is True
        queue = client._transport.app.state.queue  # type: ignore[attr-defined]
        for task in list(queue._tasks):
            await task
    assert indexed.ledger.indexed_sha("acme/checkout-py") == sha
    assert any(e.name == "extra_fn" for e in indexed.ledger.file_entries("acme/checkout-py", "checkout/extra.py"))


async def test_benchmark_on_fixture_questions(indexed: Container) -> None:
    questions = load_questions(QUESTIONS)
    assert len(questions) >= 25
    report = await BenchmarkEvaluator(indexed.retrieval_agent, indexed.judge).run(questions, "full")
    assert report.errors == 0
    assert report.mrr_at_10 > 0.5, report.headline()
    assert report.citation_faithfulness is not None and report.citation_faithfulness >= 0.95
    assert {"p50", "p95", "p99"} <= set(report.latency_ms)
    assert "cross_repo_rpc" in report.by_category


async def test_question_generator_uses_cross_repo_edges(indexed: Container) -> None:
    qs = await CrossRepoQuestionGenerator(indexed.graph, indexed.vectors, indexed.question_writer).generate(10)
    assert qs, "fixture corpus has cross-repo calls and RPC bridges"
    for q in qs:
        assert len(q.gold) == 2
    assert any(q.category == "cross_repo_rpc" for q in qs)


async def test_ingest_benchmark_replays_commits(container: Container, corpus: CorpusRepos) -> None:
    repo = "acme/shared-py"
    for i in range(3):
        corpus.write(repo, f"acme_common/util_{i}.py", f"def helper_{i}(x):\n    return x + {i}\n")
        corpus.commit(repo, f"c{i}")
    report = await IngestBenchmark(container.ingest_workflow, container.source).run(repo, 3)
    assert len(report.commits) == 3 and all(t.symbols_embedded == 1 for t in report.commits)
    assert report.slo_met
