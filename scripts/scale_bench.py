"""Synthetic-scale incremental indexing benchmark (freshness SLO).

Generates a repository of `--files` Python modules (~`--funcs` functions each), indexes it fully,
then creates `--commits` commits that each modify `--changed` files (one function body + one
line-shift per file) and measures incremental re-index time per commit.

    uv run python scripts/scale_bench.py --profile offline --files 2000 --changed 50 --commits 5
    uv run python scripts/scale_bench.py --profile cloud   --files 2000 --changed 50 --commits 3 \
        --qdrant-url http://localhost:6333

The offline profile isolates pipeline overhead (git diff, parse, hash diff, store writes, graph);
the cloud profile adds real summarization/embedding latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "b",
    "GIT_AUTHOR_EMAIL": "b@example.com",
    "GIT_COMMITTER_NAME": "b",
    "GIT_COMMITTER_EMAIL": "b@example.com",
}
REPO = "bench/monorepo"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env={**os.environ, **_GIT_ENV}
    ).stdout.strip()


def module_source(i: int, funcs: int, variant: int = 0) -> str:
    lines = [f'"""Service module {i}: order, billing and inventory helpers."""', "", "import math", ""]
    for f in range(funcs):
        callee = f"helper_{i}_{(f + 1) % funcs}"
        lines += [
            "",
            f"def helper_{i}_{f}(amount: float, qty: int = 1) -> float:",
            f'    """Compute adjusted amount #{f} for module {i} (rounding, tax and quantity)."""',
            f"    base = amount * qty + {f}",
            "    if base < 0:",
            '        raise ValueError("negative amount")',
            f"    taxed = base * (1 + 0.0{(f % 9) + 1})",
            f"    return round(taxed, 2) + {variant}"
            if f == 0
            else f"    return {callee}(taxed) if qty > 99 else taxed",
        ]
    lines += ["", "", f"class Service{i}:", f'    """Facade over module {i} helpers."""', ""]
    lines += ["    def run(self, amount: float) -> float:", f"        return helper_{i}_0(amount)", ""]
    return "\n".join(lines) + "\n"


async def main(args: argparse.Namespace) -> int:
    from qdrant_client import models as qm

    from code_rag.infrastructure.container import Container
    from code_rag.infrastructure.settings import QdrantSettings, Settings

    work = Path(tempfile.mkdtemp(prefix="code-rag-scale-"))
    repo_dir = work / "monorepo"
    (repo_dir / "svc").mkdir(parents=True)
    for i in range(args.files):
        (repo_dir / "svc" / f"mod_{i:05d}.py").write_text(module_source(i, args.funcs))
    git(repo_dir, "init", "-q", "-b", "main")
    git(repo_dir, "add", "-A")
    git(repo_dir, "commit", "-qm", "initial")
    loc = sum(len(p.read_text().splitlines()) for p in (repo_dir / "svc").iterdir())
    (work / "repos.yaml").write_text(yaml.safe_dump({"repos": [{"name": REPO, "path": str(repo_dir)}]}))
    overrides: dict[str, object] = {}
    if args.qdrant_url:
        # isolated, disposable collection on a real Qdrant server (embedded mode is for tests only)
        overrides["qdrant"] = QdrantSettings(url=args.qdrant_url, prefix=f"bench{int(time.time())}")
    settings = Settings(profile=args.profile, data_dir=work / "data", repos_file=work / "repos.yaml", **overrides)
    c = Container(settings)
    rng = random.Random(42)
    results: list[dict[str, object]] = []
    try:
        t0 = time.perf_counter()
        full = await c.ingest_workflow.run(REPO)
        print(
            f"full index: files={args.files} LOC={loc} symbols={full.symbols_embedded} "
            f"in {time.perf_counter() - t0:.1f}s  stages={full.timings_ms}"
        )
        for n in range(1, args.commits + 1):
            for i in rng.sample(range(args.files), args.changed):
                path = repo_dir / "svc" / f"mod_{i:05d}.py"
                src = module_source(i, args.funcs, variant=n)
                path.write_text("# touched\n" * n + src)  # body change in helper_0 + line shift for the rest
            git(repo_dir, "add", "-A")
            git(repo_dir, "commit", "-qm", f"change {n}")
            stats = await c.ingest_workflow.run(REPO)
            results.append(stats.model_dump(mode="json"))
            print(
                f"commit {n}: files={stats.files_changed} embedded={stats.symbols_embedded} "
                f"relocated={stats.symbols_relocated} total={stats.timings_ms['total'] / 1000:.2f}s "
                f"stages={ {k: round(v) for k, v in stats.timings_ms.items()} }"
            )
        worst = max(float(r["timings_ms"]["total"]) for r in results)  # type: ignore[index]
        verdict = "MET" if worst < args.slo_seconds * 1000 else "VIOLATED"
        print(f"SLO <{args.slo_seconds:.0f}s for {args.changed}-file commits: {verdict} (worst {worst / 1000:.2f}s)")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(
                    {
                        "profile": args.profile,
                        "files": args.files,
                        "loc": loc,
                        "changed_per_commit": args.changed,
                        "full_index": full.model_dump(mode="json"),
                        "commits": results,
                        "worst_ms": worst,
                        "slo": verdict,
                    },
                    indent=2,
                )
            )
        return 0 if verdict == "MET" else 2
    finally:
        if args.qdrant_url:
            await c.qdrant_client.delete_collection(c.vectors.collection)
            await c.qdrant_client.update_collection_aliases(
                change_aliases_operations=[
                    qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=c.vectors.alias))
                ]
            )
        await c.aclose()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="offline", choices=["offline", "cloud", "local"])
    p.add_argument("--files", type=int, default=1000)
    p.add_argument("--funcs", type=int, default=12)
    p.add_argument("--changed", type=int, default=50)
    p.add_argument("--commits", type=int, default=3)
    p.add_argument("--slo-seconds", type=float, default=60.0)
    p.add_argument("--out")
    p.add_argument("--qdrant-url", help="e.g. http://localhost:6333 (recommended: embedded mode is slow)")
    sys.exit(asyncio.run(main(p.parse_args())))
