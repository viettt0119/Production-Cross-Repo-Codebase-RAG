"""Claude adapters: symbol summarizer (realtime + Message Batches), query expander, answer
synthesizer (streaming, server-side refusal fallback) and evaluation judge."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal

import anthropic

from code_rag.domain.errors import ProviderError, ProviderRefusalError
from code_rag.domain.ports.models import FileToSummarize, QueryExpansion, QuestionSeed
from code_rag.domain.query import RankedContext
from code_rag.domain.value_objects import SymbolId
from code_rag.infrastructure.providers.prompts import (
    EXPAND_SCHEMA,
    EXPAND_SYSTEM,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    QUESTION_SCHEMA,
    QUESTION_SYSTEM,
    SUMMARY_SCHEMA,
    SUMMARY_SYSTEM,
    SYNTH_SYSTEM,
    question_user_prompt,
    summary_user_prompt,
    synth_user_prompt,
)
from code_rag.infrastructure.providers.resilience import RetryPolicy, Throttle, with_retry

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, ProviderError):
        return exc.retryable
    if isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError | anthropic.APITimeoutError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 529
    return False


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    return ProviderError("anthropic", f"{type(exc).__name__}: {exc}", retryable=_retryable(exc))


def _json_text(message: Any) -> dict[str, Any]:
    if message.stop_reason == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        raise ProviderRefusalError("anthropic", f"model declined (category={category})")
    if message.stop_reason == "max_tokens":
        raise ProviderError("anthropic", "structured output truncated at max_tokens", retryable=False)
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError("anthropic", f"invalid JSON output: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("anthropic", "structured output is not an object")
    return parsed


def _summaries_from(parsed: Mapping[str, Any], allowed: set[str]) -> dict[SymbolId, str]:
    out: dict[SymbolId, str] = {}
    for row in parsed.get("summaries", []):
        sid = str(row.get("id", ""))
        summary = str(row.get("summary", "")).strip()
        if sid in allowed and summary:
            out[SymbolId(sid)] = summary[:400]
    return out


class _ClaudeBase:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        max_concurrency: int,
        requests_per_minute: float | None,
    ) -> None:
        self._client = client
        self._model = model
        self._throttle = Throttle(max_concurrency, requests_per_minute)
        self._retry = RetryPolicy(max_attempts=4, base_delay_s=1.0)

    @property
    def model_name(self) -> str:
        return self._model

    async def _structured(
        self, system: str, user: str, schema: dict[str, object], max_tokens: int, effort: str = "low"
    ) -> dict[str, Any]:
        async def call() -> dict[str, Any]:
            async with self._throttle.slot():
                try:
                    request: dict[str, Any] = {
                        "model": self._model,
                        "max_tokens": max_tokens,
                        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                        "messages": [{"role": "user", "content": user}],
                        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                    }
                    msg = await self._client.messages.create(**request)
                except anthropic.APIError as exc:
                    raise _translate(exc) from exc
            return _json_text(msg)

        return await with_retry(call, policy=self._retry, is_retryable=_retryable, op_name=f"claude:{self._model}")


class ClaudeSummarizer(_ClaudeBase):
    """Per-file structured summarization. The large stable system prompt is cached across files."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str = "claude-haiku-5-5",
        max_concurrency: int = 32,
        requests_per_minute: float | None = 2000,
    ) -> None:
        super().__init__(client, model, max_concurrency, requests_per_minute)

    async def summarize_file(self, item: FileToSummarize) -> dict[SymbolId, str]:
        if not item.symbols:
            return {}
        max_tokens = min(16_000, 400 + 120 * len(item.symbols))
        parsed = await self._structured(SUMMARY_SYSTEM, summary_user_prompt(item), SUMMARY_SCHEMA, max_tokens)
        return _summaries_from(parsed, {s.symbol_id for s in item.symbols})


class ClaudeBatchSummarizer:
    """Bulk summarization through the Message Batches API (50% cheaper, asynchronous)."""

    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-haiku-5-5") -> None:
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def submit(self, items: Sequence[FileToSummarize]) -> str:
        requests: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            if not item.symbols:
                continue
            requests.append(
                {
                    "custom_id": f"f{i}",
                    "params": {
                        "model": self._model,
                        "max_tokens": min(16_000, 400 + 120 * len(item.symbols)),
                        "system": [{"type": "text", "text": SUMMARY_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        "messages": [{"role": "user", "content": summary_user_prompt(item)}],
                        "output_config": {
                            "effort": "low",
                            "format": {"type": "json_schema", "schema": SUMMARY_SCHEMA},
                        },
                    },
                }
            )
        if not requests:
            raise ProviderError("anthropic-batch", "nothing to summarize")
        try:
            batch = await self._client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
        except anthropic.APIError as exc:
            raise _translate(exc) from exc
        return batch.id

    async def poll(self, batch_id: str) -> Literal["in_progress", "ended"]:
        try:
            batch = await self._client.messages.batches.retrieve(batch_id)
        except anthropic.APIError as exc:
            raise _translate(exc) from exc
        return "ended" if batch.processing_status == "ended" else "in_progress"

    async def collect(self, batch_id: str) -> dict[SymbolId, str]:
        out: dict[SymbolId, str] = {}
        failed = 0
        try:
            results = await self._client.messages.batches.results(batch_id)
            async for result in results:
                if result.result.type != "succeeded":
                    failed += 1
                    continue
                try:
                    parsed = _json_text(result.result.message)
                except ProviderError:
                    failed += 1
                    continue
                for row in parsed.get("summaries", []):
                    sid, summary = str(row.get("id", "")), str(row.get("summary", "")).strip()
                    if len(sid) == 40 and summary:
                        out[SymbolId(sid)] = summary[:400]
        except anthropic.APIError as exc:
            raise _translate(exc) from exc
        if failed:
            log.warning("batch %s: %d requests failed; affected symbols keep no summary", batch_id, failed)
        return out


class ClaudeQueryExpander(_ClaudeBase):
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-haiku-5-5") -> None:
        super().__init__(client, model, max_concurrency=16, requests_per_minute=None)

    async def expand(self, question: str) -> QueryExpansion:
        parsed = await self._structured(EXPAND_SYSTEM, question, EXPAND_SCHEMA, max_tokens=1024)
        rewrites = tuple(str(r).strip() for r in parsed.get("rewrites", []) if str(r).strip())[:3]
        idents = tuple(str(r).strip() for r in parsed.get("identifiers", []) if str(r).strip())[:12]
        return QueryExpansion(rewrites=rewrites, identifiers=idents)


class ClaudeJudge(_ClaudeBase):
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-haiku-5-5") -> None:
        super().__init__(client, model, max_concurrency=16, requests_per_minute=None)

    async def supports(self, claim: str, evidence: Mapping[str, str]) -> bool:
        ev = "\n\n".join(f'<evidence anchor="{k}">\n{v}\n</evidence>' for k, v in evidence.items())
        parsed = await self._structured(JUDGE_SYSTEM, f"{ev}\n\nClaim: {claim}", JUDGE_SCHEMA, max_tokens=512)
        return bool(parsed.get("supported", False))


class ClaudeQuestionWriter(_ClaudeBase):
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = "claude-opus-5-5") -> None:
        super().__init__(client, model, max_concurrency=8, requests_per_minute=None)

    async def write(self, seed: QuestionSeed) -> str | None:
        try:
            parsed = await self._structured(
                QUESTION_SYSTEM, question_user_prompt(seed), QUESTION_SCHEMA, max_tokens=2048, effort="medium"
            )
        except ProviderRefusalError:
            return None
        question = str(parsed.get("question", "")).strip()
        return question or None


class ClaudeSynthesizer:
    """Streams a citation-constrained answer. Uses server-side `fallbacks` so a policy refusal is
    retried on a fallback model inside the same call; a final refusal raises ProviderRefusalError."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 8000,
        enable_fallback: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._enable_fallback = enable_fallback

    @property
    def model_name(self) -> str:
        return self._model

    async def stream(
        self, question: str, contexts: Sequence[RankedContext], feedback: str | None
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": [{"type": "text", "text": SYNTH_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": synth_user_prompt(question, contexts, feedback)}],
            "output_config": {"effort": self._effort},
        }
        if self._enable_fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        try:
            async with self._client.beta.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise _translate(exc) from exc
        if final.stop_reason == "refusal":
            raise ProviderRefusalError("anthropic", "answer synthesis declined by all models in the fallback chain")
        if final.stop_reason == "max_tokens":
            yield "\n\n[answer truncated: max_tokens reached]"
