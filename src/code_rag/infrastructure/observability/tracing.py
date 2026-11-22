"""Tracer adapters: Langfuse (OTel-based SDK), plain OpenTelemetry, and a recording tracer.

The recording tracer is always active underneath so per-stage latencies can be returned to API
callers and aggregated by the benchmark even when no tracing backend is configured.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from code_rag.application.timing import collect_timings, record_timing
from code_rag.domain.ports.observability import Span

__all__ = ["LangfuseTracer", "OtelTracer", "RecordingTracer", "collect_timings"]

log = logging.getLogger(__name__)


def _safe(value: Any, limit: int = 4000) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text[:limit]


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.output: Any = None
        self.error: BaseException | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_output(self, value: Any) -> None:
        self.output = value

    def record_error(self, error: BaseException) -> None:
        self.error = error


class RecordingTracer:
    """No backend: only timing collection."""

    @contextmanager
    def _span(self, name: str, input: Any = None, **attributes: Any) -> Iterator[Span]:
        span = _RecordingSpan()
        start = time.perf_counter()
        try:
            yield span
        finally:
            record_timing(name, (time.perf_counter() - start) * 1000)

    def span(self, name: str, *, input: Any = None, **attributes: Any) -> AbstractContextManager[Span]:
        return self._span(name, input, **attributes)

    @property
    def current_trace_id(self) -> str | None:
        return None

    def flush(self) -> None:
        return None


class _LangfuseSpanAdapter:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._metadata: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self._metadata[key] = _safe(value, 500)
        self._inner.update(metadata=self._metadata)

    def set_output(self, value: Any) -> None:
        self._inner.update(output=_safe(value))

    def record_error(self, error: BaseException) -> None:
        self._inner.update(level="ERROR", status_message=f"{type(error).__name__}: {error}"[:500])


class LangfuseTracer:
    def __init__(self, public_key: str, secret_key: str, host: str, environment: str = "production") -> None:
        from langfuse import Langfuse

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=host, environment=environment)

    @contextmanager
    def _span(self, name: str, input: Any = None, **attributes: Any) -> Iterator[Span]:
        as_type = attributes.pop("as_type", "span")
        start = time.perf_counter()
        with self._client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=_safe(input),
            metadata={k: _safe(v, 500) for k, v in attributes.items()},
        ) as inner:
            adapter = _LangfuseSpanAdapter(inner)
            try:
                yield adapter
            except BaseException as exc:
                adapter.record_error(exc)
                raise
            finally:
                record_timing(name, (time.perf_counter() - start) * 1000)

    def span(self, name: str, *, input: Any = None, **attributes: Any) -> AbstractContextManager[Span]:
        return self._span(name, input, **attributes)

    @property
    def current_trace_id(self) -> str | None:
        try:
            return self._client.get_current_trace_id()
        except Exception:  # SDK raises when no span is active; trace id is best-effort metadata
            return None

    def flush(self) -> None:
        self._client.flush()


class _OtelSpanAdapter:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def set_attribute(self, key: str, value: Any) -> None:
        self._inner.set_attribute(f"code_rag.{key}", _safe(value, 500))

    def set_output(self, value: Any) -> None:
        self._inner.set_attribute("code_rag.output", _safe(value, 2000))

    def record_error(self, error: BaseException) -> None:
        from opentelemetry.trace import Status, StatusCode

        self._inner.record_exception(error)
        self._inner.set_status(Status(StatusCode.ERROR, str(error)[:200]))


class OtelTracer:
    """Uses the globally configured OpenTelemetry TracerProvider (OTLP exporter configured via env)."""

    def __init__(self, service_name: str = "code-rag") -> None:
        from opentelemetry import trace

        self._trace = trace
        self._tracer = trace.get_tracer(service_name)

    @contextmanager
    def _span(self, name: str, input: Any = None, **attributes: Any) -> Iterator[Span]:
        start = time.perf_counter()
        with self._tracer.start_as_current_span(name) as inner:
            adapter = _OtelSpanAdapter(inner)
            if input is not None:
                inner.set_attribute("code_rag.input", _safe(input, 2000))
            for k, v in attributes.items():
                adapter.set_attribute(k, v)
            try:
                yield adapter
            except BaseException as exc:
                adapter.record_error(exc)
                raise
            finally:
                record_timing(name, (time.perf_counter() - start) * 1000)

    def span(self, name: str, *, input: Any = None, **attributes: Any) -> AbstractContextManager[Span]:
        return self._span(name, input, **attributes)

    @property
    def current_trace_id(self) -> str | None:
        ctx = self._trace.get_current_span().get_span_context()
        return f"{ctx.trace_id:032x}" if ctx.is_valid else None

    def flush(self) -> None:
        provider = self._trace.get_tracer_provider()
        force = getattr(provider, "force_flush", None)
        if callable(force):
            force()
