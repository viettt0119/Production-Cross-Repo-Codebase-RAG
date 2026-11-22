"""End-to-end demo on the fixture corpus, driven through the real HTTP API (in-process ASGI).

    uv run python scripts/demo.py [--profile offline|cloud]

1. index 6 repos (5 languages)      2. inspect the cross-repo graph (gRPC bridge)
3. ask via JSON and via SSE         4. push a commit -> signed GitHub webhook -> incremental re-index
5. ask again (citations follow the moved lines)   6. citation guardrail rejects fabricated answers
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx
from pydantic import SecretStr
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from fixture_eval import _GIT_ENV, materialize

console = Console()
SECRET = "demo-webhook-secret"


def step(title: str) -> None:
    console.rule(f"[bold cyan]{title}")


def git(cwd: Path, *args: str) -> str:
    import os
    import subprocess

    out = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env={**os.environ, **_GIT_ENV}
    )
    return out.stdout.strip()


async def main(profile: str) -> None:
    from code_rag.application.citation.verifier import CitationVerifier
    from code_rag.domain.query import SearchQuery
    from code_rag.infrastructure.container import Container
    from code_rag.infrastructure.settings import Settings, WebhookSettings
    from code_rag.interfaces.api.app import InlineQueue, create_app

    work = Path(tempfile.mkdtemp(prefix="code-rag-demo-"))
    repos_file = materialize(work)
    settings = Settings(
        profile=profile,
        data_dir=work / "data",
        repos_file=repos_file,
        webhooks=WebhookSettings(github_secret=SecretStr(SECRET)),
    )
    c = Container(settings)
    app = create_app(container=c)
    app.state.container = c
    app.state.queue = InlineQueue(c)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://demo", timeout=120)
    try:
        step("1. Full index of 6 repositories")
        table = Table("repo", "files", "symbols", "edges", "ms")
        for r in c.repos.repos:
            s = await c.ingest_workflow.run(r.name)
            table.add_row(
                r.name,
                str(s.files_changed),
                str(s.symbols_embedded),
                str(s.edges_resolved),
                f"{s.timings_ms['total']:.0f}",
            )
        console.print(table)
        console.print("graph:", c.graph.stats())

        step("2. Cross-repo graph: who is behind BillingClient.charge (Python)?")
        ids = {e.qualified_name: e for e in c.ledger.all_symbols()}
        charge = ids["checkout.billing_client.BillingClient.charge"]
        for n in c.graph.neighbors([charge.symbol_id], limit=10):
            e = c.ledger.symbol_entries([n.neighbor_id]).get(n.neighbor_id)
            if e:
                console.print(
                    f"  [green]{n.relation:28s}[/green] {e.repo:18s} {e.qualified_name}  (cross_repo={n.cross_repo})"
                )

        step("3a. POST /api/v1/ask (JSON)")
        q1 = "How does checkout charge the customer through the billing service?"
        r = (await client.post("/api/v1/ask", json={"question": q1})).json()
        console.print(f"[bold]Q:[/bold] {q1}")
        console.print(Markdown(r["answer_markdown"]))
        console.print(
            f"passed={r['passed']} faithfulness={r['report']['faithfulness']} attempts={r['attempts']} "
            f"stages={ {k: round(v) for k, v in r['timings_ms'].items()} }"
        )

        step("3b. POST /api/v1/ask (SSE stream)")
        q2 = "Where is a refund rejected when it is larger than the original charge?"
        console.print(f"[bold]Q:[/bold] {q2}")
        events: list[str] = []
        async with client.stream("POST", "/api/v1/ask", json={"question": q2, "stream": True}) as resp:
            event = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                    events.append(event)
                elif line.startswith("data:") and event == "stage":
                    console.print(f"  [dim]stage → {json.loads(line[5:])['stage']}[/dim]")
                elif line.startswith("data:") and event == "final":
                    final = json.loads(line[5:])
                    console.print(Markdown(final["answer_markdown"]))
                    console.print(f"passed={final['passed']}")
        console.print(f"SSE events: {len(events)} ({events.count('token')} token events)")

        step("4. Developer pushes a commit → signed GitHub webhook → incremental re-index")
        reverse_id = ids["internal.ledger.Ledger.Reverse"].symbol_id
        before = (await c.vectors.fetch([reverse_id]))[reverse_id].symbol.span
        go_repo = Path(c.source.repo_dir("acme/billing-go"))
        ledger_go = go_repo / "internal/ledger/ledger.go"
        src = ledger_go.read_text()
        src = src.replace(
            'return Entry{}, errors.New("refund exceeds charge")',
            'return Entry{}, fmt.Errorf("refund %d exceeds charge %d", amountMinor, e.AmountMinor)',
        )
        src = src.replace(
            "package ledger\n",
            ("// Package ledger: append-only ledger used by billing.\n// Owner: payments-team\n\npackage ledger\n"),
        )
        ledger_go.write_text(src)
        git(go_repo, "commit", "-qam", "ledger: richer refund error")
        sha = git(go_repo, "rev-parse", "HEAD")
        body = json.dumps(
            {"ref": "refs/heads/main", "before": "0" * 40, "after": sha, "repository": {"full_name": "acme/billing-go"}}
        ).encode()
        sig = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        bad = await client.post(
            "/webhooks/git", content=body, headers={"X-GitHub-Event": "push", "X-Hub-Signature-256": "sha256=forged"}
        )
        console.print(f"forged signature  → HTTP {bad.status_code}")
        t0 = time.perf_counter()
        ok = await client.post(
            "/webhooks/git", content=body, headers={"X-GitHub-Event": "push", "X-Hub-Signature-256": sig}
        )
        console.print(f"valid signature   → HTTP {ok.status_code} {ok.json()}")
        for task in list(app.state.queue._tasks):
            await task
        run = c.ledger.recent_runs(1)[0]
        stats = json.loads(run["stats_json"])
        indexed = (c.ledger.indexed_sha("acme/billing-go") or "")[:10]
        console.print(
            f"re-indexed in {(time.perf_counter() - t0) * 1000:.0f} ms: files={stats['files_changed']} "
            f"embedded={stats['symbols_embedded']} relocated={stats['symbols_relocated']} "
            f"unchanged={stats['symbols_unchanged']}  (indexed_sha={indexed})"
        )

        step("5. Citations follow the moved code")
        after = (await c.vectors.fetch([reverse_id]))[reverse_id].symbol.span
        console.print(f"Ledger.Reverse indexed span: lines {before} → {after} (comment added above, body edited)")
        q3 = "How does Ledger.Reverse reject a refund that exceeds the charge?"
        r = (await client.post("/api/v1/ask", json={"question": q3})).json()
        console.print(f"[bold]Q:[/bold] {q3}")
        console.print(r["answer_markdown"], markup=False)
        anchor = f"(acme/billing-go/internal/ledger/ledger.go:{after.start}-{after.end})"
        console.print(f"new anchor {anchor} cited: {anchor in r['answer_markdown']}  passed={r['passed']}")

        step("6. Guardrail: verifier on fabricated / uncited answers")
        contexts = await c.retrieval_agent.search(SearchQuery(text=q2))
        fake = (
            "Refunds are validated in the ledger (acme/billing-go/internal/ledger/ledger.go:900-910).\n\n"
            "The billing service also retries refunds three times.\n\n"
            "Validation happens in the payments gateway (acme/payments/gateway.py:1-5)."
        )
        report = c.retrieval_agent._d.verifier.verify(fake, contexts)
        for chk in report.checks:
            console.print(f"  [red]{chk.verdict.value:18s}[/red] {chk.claim.text[:70]}")
        console.print(f"passed={report.passed} faithfulness={report.faithfulness}")
        console.print("[dim]feedback sent back to the model on re-ask:[/dim]")
        console.print(CitationVerifier.feedback(report), markup=False)
    finally:
        await client.aclose()
        await c.aclose()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="offline", choices=["offline", "cloud", "local"])
    asyncio.run(main(p.parse_args().profile))
