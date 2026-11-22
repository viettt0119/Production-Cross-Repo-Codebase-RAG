"""HTTP embedding/rerank adapters: Voyage AI (cloud) and HuggingFace TEI (self-hosted)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from code_rag.domain.errors import ProviderError
from code_rag.domain.ports.models import InputType, RerankResult
from code_rag.infrastructure.providers.resilience import RetryPolicy, Throttle, batched, with_retry

VOYAGE_BASE_URL = "https://api.voyageai.com/v1"


class _HttpBase:
    def __init__(
        self,
        provider: str,
        base_url: str,
        headers: dict[str, str],
        timeout_s: float,
        max_concurrency: int,
        requests_per_minute: float | None,
        retry: RetryPolicy,
    ) -> None:
        self._provider = provider
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout_s)
        self._throttle = Throttle(max_concurrency, requests_per_minute)
        self._retry = retry

    async def _post(self, path: str, payload: dict[str, object]) -> object:
        async def call() -> object:
            async with self._throttle.slot():
                resp = await self._client.post(path, json=payload)
            if resp.status_code >= 400:
                retryable = resp.status_code in (408, 409, 429) or resp.status_code >= 500
                if retryable:
                    resp.raise_for_status()
                raise ProviderError(self._provider, f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()

        try:
            return await with_retry(call, policy=self._retry, op_name=f"{self._provider}{path}")
        except httpx.HTTPError as exc:
            raise ProviderError(self._provider, str(exc), retryable=True) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


class VoyageEmbeddingProvider(_HttpBase):
    """voyage-code-3 / voyage-3.5 embeddings. Batches by count and approximate token budget."""

    MAX_BATCH = 128
    MAX_BATCH_CHARS = 240_000  # ~60k tokens, well under per-request limits

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-code-3",
        dimension: int = 1024,
        max_concurrency: int = 8,
        requests_per_minute: float | None = 1500,
        timeout_s: float = 60.0,
        base_url: str = VOYAGE_BASE_URL,
    ) -> None:
        super().__init__(
            "voyage",
            base_url,
            {"Authorization": f"Bearer {api_key}"},
            timeout_s,
            max_concurrency,
            requests_per_minute,
            RetryPolicy(),
        )
        self._model = model
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        if not texts:
            return []
        indexed = list(enumerate(texts))
        batches = batched(indexed, self.MAX_BATCH, self.MAX_BATCH_CHARS, size=lambda it: len(it[1]))

        async def run(batch: list[tuple[int, str]]) -> list[tuple[int, list[float]]]:
            data = await self._post(
                "/embeddings",
                {
                    "input": [t for _, t in batch],
                    "model": self._model,
                    "input_type": input_type,
                    "output_dimension": self._dim,
                    "truncation": True,
                },
            )
            rows = data["data"] if isinstance(data, dict) else []
            if len(rows) != len(batch):
                raise ProviderError("voyage", f"expected {len(batch)} embeddings, got {len(rows)}")
            ordered = sorted(rows, key=lambda r: int(r["index"]))
            return [(batch[i][0], [float(x) for x in row["embedding"]]) for i, row in enumerate(ordered)]

        results = await asyncio.gather(*(run(b) for b in batches))
        out: list[list[float]] = [[] for _ in texts]
        for part in results:
            for idx, vec in part:
                out[idx] = vec
        return out


class VoyageReranker(_HttpBase):
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-2.5",
        max_concurrency: int = 8,
        requests_per_minute: float | None = 1000,
        timeout_s: float = 30.0,
        base_url: str = VOYAGE_BASE_URL,
    ) -> None:
        super().__init__(
            "voyage-rerank",
            base_url,
            {"Authorization": f"Bearer {api_key}"},
            timeout_s,
            max_concurrency,
            requests_per_minute,
            RetryPolicy(max_attempts=3),
        )
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[RerankResult]:
        if not documents:
            return []
        data = await self._post(
            "/rerank",
            {
                "query": query,
                "documents": list(documents),
                "model": self._model,
                "top_k": min(top_k, len(documents)),
                "truncation": True,
            },
        )
        rows = data["data"] if isinstance(data, dict) else []
        return [RerankResult(index=int(r["index"]), score=float(r["relevance_score"])) for r in rows]


class TeiEmbeddingProvider(_HttpBase):
    """text-embeddings-inference server (e.g. nomic-embed-code on a local GPU)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int,
        query_prefix: str = "Represent this query for searching relevant code: ",
        max_batch: int = 32,
        max_concurrency: int = 4,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__("tei-embed", base_url, {}, timeout_s, max_concurrency, None, RetryPolicy())
        self._model = model
        self._dim = dimension
        self._query_prefix = query_prefix
        self._max_batch = max_batch

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        prepared = [f"{self._query_prefix}{t}" if input_type == "query" else t for t in texts]
        batches = batched(prepared, self._max_batch)

        async def run(batch: list[str]) -> list[list[float]]:
            data = await self._post("/embed", {"inputs": batch, "truncate": True, "normalize": True})
            if not isinstance(data, list) or len(data) != len(batch):
                raise ProviderError("tei-embed", "unexpected /embed response shape")
            return [[float(x) for x in row] for row in data]

        parts = await asyncio.gather(*(run(b) for b in batches))
        return [vec for part in parts for vec in part]


class TeiReranker(_HttpBase):
    """TEI /rerank endpoint (e.g. BAAI/bge-reranker-v2-m3)."""

    def __init__(self, base_url: str, model: str = "BAAI/bge-reranker-v2-m3", timeout_s: float = 60.0) -> None:
        super().__init__("tei-rerank", base_url, {}, timeout_s, 4, None, RetryPolicy(max_attempts=3))
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[RerankResult]:
        if not documents:
            return []
        data = await self._post("/rerank", {"query": query, "texts": list(documents), "truncate": True})
        if not isinstance(data, list):
            raise ProviderError("tei-rerank", "unexpected /rerank response shape")
        rows = sorted(data, key=lambda r: float(r["score"]), reverse=True)[:top_k]
        return [RerankResult(index=int(r["index"]), score=float(r["score"])) for r in rows]
