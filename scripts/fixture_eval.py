"""Index the bundled multi-repo fixture corpus and run the curated cross-repo benchmark.

uv run python scripts/fixture_eval.py --profile offline            # no keys, deterministic
uv run python scripts/fixture_eval.py --profile cloud --judge      # Voyage + Claude
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "corpus"
QUESTIONS = ROOT / "eval" / "questions.fixture.jsonl"
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "fx",
    "GIT_AUTHOR_EMAIL": "fx@example.com",
    "GIT_COMMITTER_NAME": "fx",
    "GIT_COMMITTER_EMAIL": "fx@example.com",
}


def materialize(workdir: Path) -> Path:
    repos = []
    for d in sorted(CORPUS.iterdir()):
        dst = workdir / "src" / d.name
        shutil.copytree(d, dst)
        for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "fixture"]):
            subprocess.run(["git", *args], cwd=dst, check=True, env={**os.environ, **_GIT_ENV})
        repos.append({"name": d.name.replace("__", "/"), "path": str(dst)})
    repos_file = workdir / "repos.yaml"
    repos_file.write_text(yaml.safe_dump({"repos": repos}))
    return repos_file


async def main(args: argparse.Namespace) -> int:
    from code_rag.application.evaluation.evaluator import BenchmarkEvaluator, load_questions, write_report
    from code_rag.infrastructure.container import Container
    from code_rag.infrastructure.settings import Settings

    workdir = Path(tempfile.mkdtemp(prefix="code-rag-fixture-"))
    repos_file = materialize(workdir)
    settings = Settings(profile=args.profile, data_dir=workdir / "data", repos_file=repos_file)
    c = Container(settings)
    try:
        for repo in c.repos.repos:
            stats = await c.ingest_workflow.run(repo.name)
            print(
                f"indexed {repo.name:20s} symbols={stats.symbols_embedded:3d} edges={stats.edges_resolved:3d} "
                f"{stats.timings_ms['total']:.0f}ms"
            )
        print("graph:", json.dumps(c.graph.stats()))
        questions = load_questions(args.questions)
        evaluator = BenchmarkEvaluator(c.retrieval_agent, c.judge if args.judge else None, args.concurrency)
        config: dict[str, object] = {
            "profile": args.profile,
            "embedding": c.code_embedder.model_name,
            "reranker": c.reranker.model_name,
            "synthesizer": c.synthesizer.model_name,
            "corpus": "tests/fixtures/corpus",
        }
        report = await evaluator.run(questions, "retrieval" if args.retrieval_only else "full", config)
        out = Path(args.out or ROOT / "eval" / "reports" / f"fixture-{args.profile}.json")
        write_report(report, out)
        for cat, m in report.by_category.items():
            print(f"  {cat:18s} n={m['n']:.0f}  MRR@10={m['mrr']:.3f}  nDCG@10={m['ndcg']:.3f}  R@10={m['recall']:.3f}")
        print(report.headline())
        print(f"report: {out}")
        if args.min_mrr is not None and report.mrr_at_10 < args.min_mrr:
            print(f"FAIL: MRR@10 {report.mrr_at_10:.3f} < {args.min_mrr}", file=sys.stderr)
            return 2
        return 0
    finally:
        await c.aclose()
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="offline", choices=["offline", "cloud", "local"])
    p.add_argument("--questions", default=str(QUESTIONS))
    p.add_argument("--out")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--min-mrr", type=float)
    p.add_argument("--keep", action="store_true", help="keep the temp workdir for inspection")
    sys.exit(asyncio.run(main(p.parse_args())))
