"""`code-rag` command-line interface."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from code_rag.domain.errors import CodeRagError
from code_rag.domain.query import SearchQuery
from code_rag.infrastructure.container import Container
from code_rag.infrastructure.settings import Settings

app = typer.Typer(add_completion=False, help="Cross-repo semantic code search & citation-enforced RAG.")
eval_app = typer.Typer(help="Evaluation: question generation, quality metrics, ingest SLO benchmark.")
summaries_app = typer.Typer(help="Asynchronous summary enrichment (Message Batches).")
app.add_typer(eval_app, name="eval")
app.add_typer(summaries_app, name="summaries")
console = Console()
err = Console(stderr=True)
T = TypeVar("T")


def _run[T](fn: Callable[[Container], Awaitable[T]], role: str | None = None) -> T:
    async def main() -> T:
        settings = Settings()
        if role:
            settings = settings.model_copy(update={"role": role})
        c = Container(settings)
        try:
            return await fn(c)
        finally:
            await c.aclose()

    try:
        return asyncio.run(main())
    except CodeRagError as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def index(
    repos: Annotated[list[str] | None, typer.Argument(help="org/name; default: all configured repos")] = None,
    full: Annotated[bool, typer.Option(help="Force a full reconcile instead of a diff")] = False,
    to: Annotated[str | None, typer.Option(help="Target commit/ref (default: HEAD)")] = None,
) -> None:
    """Index repositories (incremental by default)."""

    async def go(c: Container) -> None:
        targets = repos or [r.name for r in c.repos.repos]
        if not targets:
            err.print("no repositories configured (see config/repos.yaml)")
            raise typer.Exit(1)
        table = Table("repo", "files", "embedded", "relocated", "deleted", "edges", "summary fails", "total ms")
        for repo in targets:
            sha = await c.source.head(repo, to) if to else None
            s = await c.ingest_workflow.run(repo, to_sha=sha, force_full=full)
            table.add_row(
                repo,
                str(s.files_changed),
                str(s.symbols_embedded),
                str(s.symbols_relocated),
                str(s.symbols_deleted),
                str(s.edges_resolved),
                str(s.summaries_failed),
                f"{s.timings_ms.get('total', 0):.0f}",
            )
        console.print(table)

    _run(go, role="worker")


@app.command()
def ask(
    question: str,
    repo: Annotated[list[str] | None, typer.Option("--repo", "-r")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the CitedAnswer as JSON")] = False,
    include_generated: bool = False,
) -> None:
    """Ask a question; streams the answer and prints citation verification."""

    async def go(c: Container) -> None:
        query = SearchQuery(text=question, repos=tuple(repo or ()), include_generated=include_generated)
        final: Any = None
        async for ev in c.retrieval_agent.stream(query):
            if as_json:
                if ev["type"] == "final":
                    final = ev["answer"]
                continue
            if ev["type"] == "stage":
                err.print(f"[dim]· {ev['stage']}[/dim]")
            elif ev["type"] == "token":
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
            elif ev["type"] == "retry":
                err.print(f"\n[yellow]↻ citation check failed, re-asking:[/yellow]\n{ev['reason']}\n")
            elif ev["type"] == "final":
                final = ev["answer"]
        if as_json:
            console.print_json(final.model_dump_json())
            return
        console.print("\n")
        console.rule("verified answer")
        console.print(Markdown(final.answer_markdown))
        status = "[green]passed[/green]" if final.report.passed else "[red]failed[/red]"
        console.print(
            f"\ncitations {status}  faithfulness={final.report.faithfulness:.2f}  attempts={final.attempts}  "
            f"total={final.timings_ms.get('total', 0):.0f}ms"
        )

    _run(go)


@app.command()
def search(
    query: str,
    repo: Annotated[list[str] | None, typer.Option("--repo", "-r")] = None,
    top_k: int = 10,
) -> None:
    """Hybrid retrieval only (no LLM synthesis)."""

    async def go(c: Container) -> None:
        await c.vectors.ensure_ready()
        contexts = await c.retrieval_agent.search(SearchQuery(text=query, repos=tuple(repo or ()), top_k=top_k))
        table = Table("#", "symbol", "location", "sources", "rerank")
        for i, ctx in enumerate(contexts, 1):
            s = ctx.chunk.symbol
            table.add_row(
                str(i),
                s.qualified_name,
                f"{s.coord.repo}/{s.coord.path}:{s.span}",
                ",".join(x.value for x in ctx.sources),
                f"{ctx.rerank_score:.3f}" if ctx.rerank_score is not None else "-",
            )
        console.print(table)

    _run(go)


@app.command()
def stats() -> None:
    """Index statistics."""

    async def go(c: Container) -> None:
        await c.vectors.ensure_ready()
        data = {
            "profile": c.settings.profile,
            "embedding_model": c.code_embedder.model_name,
            "collection": c.vectors.collection,
            "chunks": await c.vectors.count(),
            "lexical_docs": c.lexical.num_docs(),
            "graph": c.graph.stats(),
            "repos": {r: c.ledger.indexed_sha(r) for r in c.ledger.repos()},
        }
        console.print_json(json.dumps(data))

    _run(go, role="worker")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080, workers: int = 1) -> None:
    """Run the HTTP API (uses CODE_RAG_ROLE: 'all' = single process, 'api' = read-only + Redis queue)."""
    import uvicorn

    uvicorn.run("code_rag.interfaces.api.app:app_factory", factory=True, host=host, port=port, workers=workers)


@app.command()
def worker() -> None:
    """Run the arq ingest worker (single writer)."""
    from arq.worker import run_worker

    from code_rag.interfaces.worker.main import WorkerSettings

    run_worker(WorkerSettings)  # type: ignore[arg-type]


@summaries_app.command("sync")
def summaries_sync() -> None:
    """Apply finished summary batches to the index."""
    from code_rag.application.ingest.workflow import SummaryEnrichment

    async def go(c: Container) -> None:
        applied = await SummaryEnrichment(c.ingest_dependencies()).sync()
        console.print_json(json.dumps(applied))

    _run(go, role="worker")


@eval_app.command("run")
def eval_run(
    questions: Annotated[Path, typer.Option(help="JSONL question set")] = Path("eval/questions.jsonl"),
    out: Annotated[Path, typer.Option(help="Report path")] = Path("eval/reports/latest.json"),
    retrieval_only: Annotated[bool, typer.Option(help="Skip synthesis (ranking metrics + latency)")] = False,
    judge: Annotated[bool, typer.Option(help="LLM entailment judge for faithfulness")] = False,
    concurrency: int = 4,
    min_mrr: Annotated[float | None, typer.Option(help="Fail (exit 2) if MRR@10 is below this")] = None,
) -> None:
    """Run the benchmark: MRR@10, nDCG@10, Recall@10, citation faithfulness, p50/p95/p99 latency."""
    from code_rag.application.evaluation.evaluator import BenchmarkEvaluator, load_questions, write_report

    async def go(c: Container) -> None:
        await c.vectors.ensure_ready()
        qs = load_questions(questions)
        evaluator = BenchmarkEvaluator(c.retrieval_agent, c.judge if judge else None, concurrency)
        config: dict[str, object] = {
            "profile": c.settings.profile,
            "embedding": c.code_embedder.model_name,
            "reranker": c.reranker.model_name,
            "synthesizer": c.synthesizer.model_name,
            "retrieval": c.settings.retrieval.model_dump(),
        }
        report = await evaluator.run(qs, "retrieval" if retrieval_only else "full", config)
        write_report(report, out)
        table = Table("category", "n", "MRR@10", "nDCG@10", "R@10")
        for cat, m in report.by_category.items():
            table.add_row(cat, f"{m['n']:.0f}", f"{m['mrr']:.3f}", f"{m['ndcg']:.3f}", f"{m['recall']:.3f}")
        console.print(table)
        console.print(report.headline())
        console.print(f"report written to {out}")
        if min_mrr is not None and report.mrr_at_10 < min_mrr:
            err.print(f"[red]MRR@10 {report.mrr_at_10:.3f} < required {min_mrr}[/red]")
            raise typer.Exit(2)

    _run(go)


@eval_app.command("generate")
def eval_generate(
    n: int = 100,
    out: Annotated[Path, typer.Option()] = Path("eval/questions.generated.jsonl"),
) -> None:
    """Generate cross-repo questions from graph edges (review them before use)."""
    from code_rag.application.evaluation.generator import CrossRepoQuestionGenerator, write_questions

    async def go(c: Container) -> None:
        await c.vectors.ensure_ready()
        gen = CrossRepoQuestionGenerator(c.graph, c.vectors, c.question_writer)
        qs = await gen.generate(n)
        write_questions(qs, out)
        console.print(f"wrote {len(qs)} questions to {out}")

    _run(go, role="worker")


@eval_app.command("ingest-bench")
def eval_ingest_bench(
    repo: str,
    commits: Annotated[int, typer.Option(help="Replay the last N commits")] = 20,
    slo_files: int = 50,
    slo_seconds: float = 60.0,
    out: Annotated[Path, typer.Option()] = Path("eval/reports/ingest_bench.json"),
) -> None:
    """Replay recent commits incrementally and check the freshness SLO."""
    from code_rag.application.evaluation.ingest_bench import IngestBenchmark

    async def go(c: Container) -> None:
        report = await IngestBenchmark(c.ingest_workflow, c.source).run(repo, commits, slo_files, slo_seconds)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2))
        table = Table("commit", "files", "embedded", "relocated", "deleted", "total ms")
        for t in report.commits:
            table.add_row(
                t.sha[:10],
                str(t.files_changed),
                str(t.symbols_embedded),
                str(t.symbols_relocated),
                str(t.symbols_deleted),
                f"{t.total_ms:.0f}",
            )
        console.print(table)
        verdict = "[green]MET[/green]" if report.slo_met else f"[red]VIOLATED[/red] {report.slo_violations}"
        console.print(f"latency {report.latency_ms}  SLO (<= {slo_files} files in < {slo_seconds}s): {verdict}")
        if not report.slo_met:
            raise typer.Exit(2)

    _run(go, role="worker")


@app.command("activate-collection")
def activate_collection() -> None:
    """Point the `<prefix>_active` Qdrant alias at the current embedding model's collection (A/B cut-over)."""

    async def go(c: Container) -> None:
        await c.vectors.ensure_ready()
        await c.vectors.activate()
        console.print(f"{c.vectors.alias} -> {c.vectors.collection}")

    _run(go, role="worker")


if __name__ == "__main__":
    app()
