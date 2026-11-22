"""Per-request stage timing collection (context-local), independent of any tracing backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_timings: ContextVar[dict[str, float] | None] = ContextVar("code_rag_timings", default=None)


@contextmanager
def collect_timings() -> Iterator[dict[str, float]]:
    """Collect `span name -> elapsed ms` for every span opened inside this context."""
    bucket: dict[str, float] = {}
    token = _timings.set(bucket)
    try:
        yield bucket
    finally:
        _timings.reset(token)


def record_timing(name: str, elapsed_ms: float) -> None:
    bucket = _timings.get()
    if bucket is not None:
        bucket[name] = round(bucket.get(name, 0.0) + elapsed_ms, 2)
