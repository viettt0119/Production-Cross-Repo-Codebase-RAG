"""Self-hosted LLM adapters over vLLM's HTTP chat-completions endpoint (local profile)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from code_rag.domain.errors import ProviderError
from code_rag.domain.ports.models import FileToSummarize, QueryExpansion
from code_rag.domain.query import RankedContext
from code_rag.domain.value_objects import SymbolId
from code_rag.infrastructure.providers.prompts import (
    EXPAND_SCHEMA,
    EXPAND_SYSTEM,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    SUMMARY_SCHEMA,
    SUMMARY_SYSTEM,
    SYNTH_SYSTEM,
    summary_user_prompt,
    synth_user_prompt,
)
from code_rag.infrastructure.providers.resilience import RetryPolicy, Throttle, with_retry


class VllmChat:
    def __init__(self, base_url: str, model: str, max_concurrency: int = 16, timeout_s: float = 180.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self.model = model
        self._throttle = Throttle(max_concurrency)
        self._retry = RetryPolicy(max_attempts=4)

    async def json_completion(
        self, system: str, user: str, schema: dict[str, object], max_tokens: int
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        }

        async def call() -> dict[str, Any]:
            async with self._throttle.slot():
                resp = await self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderError("vllm", f"invalid JSON: {exc}", retryable=True) from exc
            if not isinstance(parsed, dict):
                raise ProviderError("vllm", "structured output is not an object")
            return parsed

        try:
            return await with_retry(call, policy=self._retry, op_name="vllm.json")
        except httpx.HTTPError as exc:
            raise ProviderError("vllm", str(exc), retryable=True) from exc

    async def stream_text(self, system: str, user: str, max_tokens: int) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            async with (
                self._throttle.slot(),
                self._client.stream("POST", "/v1/chat/completions", json=payload) as resp,
            ):
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError("vllm", str(exc), retryable=True) from exc


class VllmSummarizer:
    def __init__(self, chat: VllmChat) -> None:
        self._chat = chat

    @property
    def model_name(self) -> str:
        return self._chat.model

    async def summarize_file(self, item: FileToSummarize) -> dict[SymbolId, str]:
        if not item.symbols:
            return {}
        parsed = await self._chat.json_completion(
            SUMMARY_SYSTEM, summary_user_prompt(item), SUMMARY_SCHEMA, 400 + 120 * len(item.symbols)
        )
        allowed = {s.symbol_id for s in item.symbols}
        out: dict[SymbolId, str] = {}
        for row in parsed.get("summaries", []):
            sid, summary = str(row.get("id", "")), str(row.get("summary", "")).strip()
            if sid in allowed and summary:
                out[SymbolId(sid)] = summary[:400]
        return out


class VllmQueryExpander:
    def __init__(self, chat: VllmChat) -> None:
        self._chat = chat

    async def expand(self, question: str) -> QueryExpansion:
        parsed = await self._chat.json_completion(EXPAND_SYSTEM, question, EXPAND_SCHEMA, 512)
        return QueryExpansion(
            rewrites=tuple(str(r) for r in parsed.get("rewrites", []))[:3],
            identifiers=tuple(str(r) for r in parsed.get("identifiers", []))[:12],
        )


class VllmSynthesizer:
    def __init__(self, chat: VllmChat, max_tokens: int = 4000) -> None:
        self._chat = chat
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._chat.model

    def stream(self, question: str, contexts: Sequence[RankedContext], feedback: str | None) -> AsyncIterator[str]:
        return self._chat.stream_text(SYNTH_SYSTEM, synth_user_prompt(question, contexts, feedback), self._max_tokens)


class VllmJudge:
    def __init__(self, chat: VllmChat) -> None:
        self._chat = chat

    async def supports(self, claim: str, evidence: Mapping[str, str]) -> bool:
        ev = "\n\n".join(f'<evidence anchor="{k}">\n{v}\n</evidence>' for k, v in evidence.items())
        parsed = await self._chat.json_completion(JUDGE_SYSTEM, f"{ev}\n\nClaim: {claim}", JUDGE_SCHEMA, 64)
        return bool(parsed.get("supported", False))
