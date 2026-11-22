"""Git provider webhook verification + payload normalisation (GitHub and GitLab)."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any

from code_rag.domain.errors import WebhookAuthError
from code_rag.domain.value_objects import FrozenModel

ZERO_SHA = "0" * 40


class PushEvent(FrozenModel):
    provider: str
    repo: str
    ref: str
    before: str | None
    after: str
    delivery_id: str | None = None

    @property
    def branch(self) -> str:
        return self.ref.removeprefix("refs/heads/")

    @property
    def is_branch_delete(self) -> bool:
        return self.after == ZERO_SHA


def verify_github(secret: str, body: bytes, signature_header: str | None) -> None:
    if not signature_header or not signature_header.startswith("sha256="):
        raise WebhookAuthError("missing X-Hub-Signature-256")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise WebhookAuthError("invalid GitHub signature")


def verify_gitlab(secret: str, token_header: str | None) -> None:
    if not token_header or not hmac.compare_digest(secret, token_header):
        raise WebhookAuthError("invalid X-Gitlab-Token")


def parse_push(headers: Mapping[str, str], body: bytes) -> PushEvent | None:
    """Return a PushEvent for push hooks, None for other event types (ping, PRs, ...)."""
    lower = {k.lower(): v for k, v in headers.items()}
    try:
        payload: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        raise WebhookAuthError(f"invalid JSON body: {exc}") from exc
    if "x-github-event" in lower:
        if lower["x-github-event"] != "push":
            return None
        before = payload.get("before")
        return PushEvent(
            provider="github",
            repo=payload["repository"]["full_name"],
            ref=payload["ref"],
            before=None if before in (None, ZERO_SHA) else before,
            after=payload["after"],
            delivery_id=lower.get("x-github-delivery"),
        )
    if "x-gitlab-event" in lower:
        if payload.get("object_kind") != "push":
            return None
        before = payload.get("before")
        return PushEvent(
            provider="gitlab",
            repo=payload["project"]["path_with_namespace"],
            ref=payload["ref"],
            before=None if before in (None, ZERO_SHA) else before,
            after=payload["after"],
            delivery_id=lower.get("x-gitlab-event-uuid"),
        )
    return None


def authenticate(headers: Mapping[str, str], body: bytes, github_secret: str | None, gitlab_secret: str | None) -> None:
    lower = {k.lower(): v for k, v in headers.items()}
    if "x-github-event" in lower:
        if not github_secret:
            raise WebhookAuthError("GitHub webhooks not configured")
        verify_github(github_secret, body, lower.get("x-hub-signature-256"))
    elif "x-gitlab-event" in lower:
        if not gitlab_secret:
            raise WebhookAuthError("GitLab webhooks not configured")
        verify_gitlab(gitlab_secret, lower.get("x-gitlab-token"))
    else:
        raise WebhookAuthError("unknown webhook provider")
