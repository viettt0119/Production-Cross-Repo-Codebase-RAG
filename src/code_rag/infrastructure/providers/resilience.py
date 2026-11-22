"""Retry with exponential backoff + jitter, token-bucket rate limiting, bounded concurrency."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

import httpx

from code_rag.domain.errors import ProviderError

T = TypeVar("T")
log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    jitter: float = 0.25

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.max_delay_s)
        backoff = min(self.max_delay_s, self.base_delay_s * 2.0 ** (attempt - 1))
        return backoff * (1 + random.uniform(-self.jitter, self.jitter))


def is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, ProviderError):
        return exc.retryable
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in (408, 409, 425, 429) or code >= 500
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[BaseException], bool] = is_retryable_http,
    op_name: str = "call",
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay(attempt, retry_after_seconds(exc))
            log.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.2fs",
                op_name,
                attempt,
                policy.max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


class AsyncRateLimiter:
    """Token bucket: `rate` permits per `per_seconds`, burst up to `capacity`."""

    def __init__(self, rate: float, per_seconds: float = 60.0, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._fill_per_s = rate / per_seconds
        self._capacity = capacity if capacity is not None else max(1.0, rate / 10)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, permits: float = 1.0) -> None:
        permits = min(permits, self._capacity)
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._fill_per_s)
                self._updated = now
                if self._tokens >= permits:
                    self._tokens -= permits
                    return
                wait = (permits - self._tokens) / self._fill_per_s
            await asyncio.sleep(wait)


class Throttle:
    """Bounded concurrency + optional request-rate limiting, used by every remote adapter."""

    def __init__(self, max_concurrency: int, requests_per_minute: float | None = None) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        self._limiter = AsyncRateLimiter(requests_per_minute) if requests_per_minute else None

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._sem:
            if self._limiter is not None:
                await self._limiter.acquire()
            yield


def batched[T](
    items: list[T], max_items: int, max_chars: int | None = None, size: Callable[[T], int] | None = None
) -> list[list[T]]:
    """Split into batches bounded by count and (optionally) by an approximate char budget."""
    out: list[list[T]] = []
    current: list[T] = []
    chars = 0
    for item in items:
        n = size(item) if size else 0
        if current and (len(current) >= max_items or (max_chars is not None and chars + n > max_chars)):
            out.append(current)
            current, chars = [], 0
        current.append(item)
        chars += n
    if current:
        out.append(current)
    return out
