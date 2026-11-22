from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import httpx
import pytest

from code_rag.application.retrieval.fusion import RankedList, rrf_fuse
from code_rag.domain.errors import WebhookAuthError
from code_rag.domain.ports.indexing import ScoredId
from code_rag.domain.query import HitSource
from code_rag.domain.value_objects import SymbolId
from code_rag.infrastructure.git.webhooks import authenticate, parse_push
from code_rag.infrastructure.providers.resilience import AsyncRateLimiter, RetryPolicy, batched, with_retry
from code_rag.infrastructure.text import code_tokens, identifiers_in, split_identifier


def _ids(*names: str) -> list[ScoredId]:
    return [ScoredId(SymbolId(n), 1.0) for n in names]


def test_rrf_rewards_agreement_and_respects_weights() -> None:
    fused = rrf_fuse(
        [
            RankedList(HitSource.DENSE_CODE, _ids("a", "b", "c")),
            RankedList(HitSource.BM25, _ids("b", "d", "b")),
        ],
        k=60,
    )
    assert fused[0].symbol_id == "b"
    assert set(fused[0].sources) == {HitSource.DENSE_CODE, HitSource.BM25}
    weighted = rrf_fuse(
        [
            RankedList(HitSource.DENSE_CODE, _ids("a"), weight=0.1),
            RankedList(HitSource.BM25, _ids("z"), weight=1.0),
        ]
    )
    assert weighted[0].symbol_id == "z"


def test_identifier_tokenization() -> None:
    assert split_identifier("getUserByID") == ["get", "user", "by", "id"]
    assert split_identifier("parse_money") == ["parse", "money"]
    assert "billingserviceclient" in code_tokens("new BillingServiceClient()")
    assert identifiers_in("Where does `place_order` call ChargeClient and to_minor_units?") == [
        "place_order",
        "ChargeClient",
        "to_minor_units",
    ]


def test_batched_respects_count_and_budget() -> None:
    items = ["x" * 10] * 7
    assert [len(b) for b in batched(items, 3)] == [3, 3, 1]
    assert [len(b) for b in batched(items, 100, max_chars=25, size=len)] == [2, 2, 2, 1]


async def test_retry_backs_off_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    assert await with_retry(flaky, policy=RetryPolicy(max_attempts=4, base_delay_s=0.001)) == "ok"
    assert calls["n"] == 3


async def test_retry_does_not_retry_client_errors() -> None:
    calls = {"n": 0}

    async def bad() -> None:
        calls["n"] += 1
        request = httpx.Request("POST", "http://x")
        raise httpx.HTTPStatusError("bad", request=request, response=httpx.Response(400, request=request))

    with pytest.raises(httpx.HTTPStatusError):
        await with_retry(bad, policy=RetryPolicy(max_attempts=4, base_delay_s=0.001))
    assert calls["n"] == 1


async def test_rate_limiter_paces_requests() -> None:
    limiter = AsyncRateLimiter(rate=20, per_seconds=1.0, capacity=1)
    start = time.monotonic()
    await asyncio.gather(*(limiter.acquire() for _ in range(5)))
    assert time.monotonic() - start >= 0.18


def _gh_headers(body: bytes, secret: str, event: str = "push") -> dict[str, str]:
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"X-GitHub-Event": event, "X-Hub-Signature-256": sig, "X-GitHub-Delivery": "d1"}


def test_github_webhook_auth_and_parse() -> None:
    body = json.dumps(
        {
            "ref": "refs/heads/main",
            "before": "0" * 40,
            "after": "f" * 40,
            "repository": {"full_name": "acme/checkout-py"},
        }
    ).encode()
    headers = _gh_headers(body, "s3cret")
    authenticate(headers, body, "s3cret", None)
    event = parse_push(headers, body)
    assert event is not None and event.repo == "acme/checkout-py" and event.before is None and event.branch == "main"
    with pytest.raises(WebhookAuthError):
        authenticate(_gh_headers(body, "wrong"), body, "s3cret", None)
    assert parse_push(_gh_headers(body, "s3cret", event="ping"), body) is None


def test_gitlab_webhook_auth_and_parse() -> None:
    body = json.dumps(
        {
            "object_kind": "push",
            "ref": "refs/heads/main",
            "before": "a" * 40,
            "after": "b" * 40,
            "project": {"path_with_namespace": "acme/billing-go"},
        }
    ).encode()
    headers = {"X-Gitlab-Event": "Push Hook", "X-Gitlab-Token": "tok"}
    authenticate(headers, body, None, "tok")
    event = parse_push(headers, body)
    assert event is not None and event.provider == "gitlab" and event.before == "a" * 40
    with pytest.raises(WebhookAuthError):
        authenticate({**headers, "X-Gitlab-Token": "nope"}, body, None, "tok")
