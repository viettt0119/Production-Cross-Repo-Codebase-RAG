"""Composition root: builds adapters for the configured profile and role."""

from __future__ import annotations

import asyncio
import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient

from code_rag.domain.errors import ConfigurationError
from code_rag.domain.ports.models import (
    AnswerSynthesizer,
    BatchSummarizer,
    EmbeddingProvider,
    EntailmentJudge,
    QueryExpander,
    QuestionWriter,
    Reranker,
    SymbolSummarizer,
)
from code_rag.domain.ports.observability import Tracer
from code_rag.infrastructure.git.local_git import LocalGitSource
from code_rag.infrastructure.observability.tracing import LangfuseTracer, OtelTracer, RecordingTracer
from code_rag.infrastructure.parsing.treesitter_parser import ParallelParser, TreeSitterParser
from code_rag.infrastructure.providers.offline import (
    ExtractiveSynthesizer,
    HashingEmbeddingProvider,
    HeuristicSummarizer,
    LexicalEntailmentJudge,
    LexicalOverlapReranker,
    RuleBasedQueryExpander,
)
from code_rag.infrastructure.settings import ReposFile, Settings
from code_rag.infrastructure.stores.graph_store import EmbeddedGraphStore
from code_rag.infrastructure.stores.qdrant_index import QdrantVectorIndex
from code_rag.infrastructure.stores.sqlite_hash_ledger import SqliteHashLedger
from code_rag.infrastructure.stores.tantivy_index import TantivyLexicalIndex

if TYPE_CHECKING:
    import anthropic

    from code_rag.application.ingest.workflow import IncrementalIngestWorkflow, IngestDependencies
    from code_rag.application.retrieval.agent import HybridRetrievalAgent
    from code_rag.infrastructure.providers.vllm_llm import VllmChat


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.data_dir = Path(self.settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()

    @property
    def readonly(self) -> bool:
        return self.settings.role == "api"

    # ---- configuration -----------------------------------------------------------
    @cached_property
    def repos(self) -> ReposFile:
        return self.settings.load_repos()

    # ---- stores ------------------------------------------------------------------
    @cached_property
    def ledger(self) -> SqliteHashLedger:
        return SqliteHashLedger(self.data_dir / "ledger.sqlite3")

    @cached_property
    def lexical(self) -> TantivyLexicalIndex:
        return TantivyLexicalIndex(self.data_dir / "tantivy", readonly=self.readonly)

    @cached_property
    def graph(self) -> EmbeddedGraphStore:
        return EmbeddedGraphStore(self.data_dir / "graph", readonly=self.readonly)

    @cached_property
    def qdrant_client(self) -> AsyncQdrantClient:
        q = self.settings.qdrant
        if q.url:
            return AsyncQdrantClient(url=q.url, api_key=q.api_key.get_secret_value() if q.api_key else None)
        return AsyncQdrantClient(path=str(self.data_dir / "qdrant"))

    @cached_property
    def vectors(self) -> QdrantVectorIndex:
        return QdrantVectorIndex(
            self.qdrant_client,
            embedding_model=self.code_embedder.model_name,
            code_dimension=self.code_embedder.dimension,
            summary_dimension=self.summary_embedder.dimension,
            prefix=self.settings.qdrant.prefix,
            server_side_boost=self.settings.qdrant.server_side_boost,
            on_disk=bool(self.settings.qdrant.url),
            quantize=bool(self.settings.qdrant.url),
            payload_indexes=bool(self.settings.qdrant.url),
        )

    # ---- source & parsing ----------------------------------------------------------
    @cached_property
    def source(self) -> LocalGitSource:
        remotes = {r.name: r.remote for r in self.repos.repos if r.remote}
        local = {r.name: r.path for r in self.repos.repos if r.path}
        return LocalGitSource(self.data_dir / "repos", remotes=remotes, local_paths=local)

    @cached_property
    def parser(self) -> ParallelParser:
        gen, exc = self.repos.generated_globs, self.repos.exclude_globs
        return ParallelParser(TreeSitterParser(gen, exc), self.settings.parse_workers, gen, exc)

    # ---- model providers ---------------------------------------------------------------
    def _anthropic(self) -> anthropic.AsyncAnthropic:
        import anthropic

        key = self.settings.cloud.anthropic_api_key
        return anthropic.AsyncAnthropic(api_key=key.get_secret_value() if key else None, max_retries=3)

    @cached_property
    def anthropic_client(self) -> anthropic.AsyncAnthropic:
        return self._anthropic()

    def _voyage_key(self) -> str:
        key = self.settings.cloud.voyage_api_key
        value = key.get_secret_value() if key else os.environ.get("VOYAGE_API_KEY")
        if not value:
            raise ConfigurationError("cloud profile requires CODE_RAG_CLOUD__VOYAGE_API_KEY or VOYAGE_API_KEY")
        return value

    @cached_property
    def code_embedder(self) -> EmbeddingProvider:
        p = self.settings.profile
        if p == "cloud":
            from code_rag.infrastructure.providers.http_models import VoyageEmbeddingProvider

            c = self.settings.cloud
            return VoyageEmbeddingProvider(self._voyage_key(), c.code_embedding_model, c.embedding_dim)
        if p == "local":
            from code_rag.infrastructure.providers.http_models import TeiEmbeddingProvider

            loc = self.settings.local
            return TeiEmbeddingProvider(loc.tei_code_embed_url, loc.code_embedding_model, loc.code_embedding_dim)
        return HashingEmbeddingProvider(self.settings.offline.embedding_dim)

    @cached_property
    def summary_embedder(self) -> EmbeddingProvider:
        p = self.settings.profile
        if p == "cloud":
            from code_rag.infrastructure.providers.http_models import VoyageEmbeddingProvider

            c = self.settings.cloud
            return VoyageEmbeddingProvider(self._voyage_key(), c.summary_embedding_model, c.embedding_dim)
        if p == "local":
            from code_rag.infrastructure.providers.http_models import TeiEmbeddingProvider

            loc = self.settings.local
            return TeiEmbeddingProvider(
                loc.tei_summary_embed_url,
                loc.summary_embedding_model,
                loc.summary_embedding_dim,
                query_prefix="search_query: ",
            )
        return self.code_embedder

    @cached_property
    def summarizer(self) -> SymbolSummarizer:
        p = self.settings.profile
        if p == "cloud":
            from code_rag.infrastructure.providers.anthropic_llm import ClaudeSummarizer

            c = self.settings.cloud
            return ClaudeSummarizer(self.anthropic_client, c.summarizer_model, c.summarizer_concurrency)
        if p == "local":
            from code_rag.infrastructure.providers.vllm_llm import VllmSummarizer

            return VllmSummarizer(self._vllm_small)
        return HeuristicSummarizer()

    @cached_property
    def batch_summarizer(self) -> BatchSummarizer | None:
        if self.settings.profile == "cloud" and self.settings.cloud.use_batches_for_bulk:
            from code_rag.infrastructure.providers.anthropic_llm import ClaudeBatchSummarizer

            return ClaudeBatchSummarizer(self.anthropic_client, self.settings.cloud.summarizer_model)
        return None

    @cached_property
    def _vllm(self) -> VllmChat:
        from code_rag.infrastructure.providers.vllm_llm import VllmChat

        loc = self.settings.local
        return VllmChat(loc.vllm_url, loc.vllm_model)

    @cached_property
    def _vllm_small(self) -> VllmChat:
        from code_rag.infrastructure.providers.vllm_llm import VllmChat

        loc = self.settings.local
        return VllmChat(loc.vllm_small_url, loc.vllm_small_model) if loc.vllm_small_url else self._vllm

    @cached_property
    def reranker(self) -> Reranker:
        p = self.settings.profile
        if p == "cloud":
            from code_rag.infrastructure.providers.http_models import VoyageReranker

            return VoyageReranker(self._voyage_key(), self.settings.cloud.rerank_model)
        if p == "local":
            from code_rag.infrastructure.providers.http_models import TeiReranker

            return TeiReranker(self.settings.local.tei_rerank_url, self.settings.local.rerank_model)
        return LexicalOverlapReranker()

    @cached_property
    def expander(self) -> QueryExpander:
        p = self.settings.profile
        if p == "cloud":
            from code_rag.infrastructure.providers.anthropic_llm import ClaudeQueryExpander

            return ClaudeQueryExpander(self.anthropic_client, self.settings.cloud.expander_model)
        if p == "local":
            from code_rag.infrastructure.providers.vllm_llm import VllmQueryExpander

            return VllmQueryExpander(self._vllm_small)
        return RuleBasedQueryExpander()

    @cached_property
    def synthesizer(self) -> AnswerSynthesizer:
        p = self.settings.profile
        if p == "cloud":
            from code_rag.infrastructure.providers.anthropic_llm import ClaudeSynthesizer

            c = self.settings.cloud
            return ClaudeSynthesizer(
                self.anthropic_client,
                c.synth_model,
                c.synth_effort,
                enable_fallback=c.enable_refusal_fallback,
            )
        if p == "local":
            from code_rag.infrastructure.providers.vllm_llm import VllmSynthesizer

            return VllmSynthesizer(self._vllm)
        return ExtractiveSynthesizer()

    @cached_property
    def judge(self) -> EntailmentJudge:
        p = self.settings.profile
        if p == "cloud":
            from code_rag.infrastructure.providers.anthropic_llm import ClaudeJudge

            return ClaudeJudge(self.anthropic_client, self.settings.cloud.judge_model)
        if p == "local":
            from code_rag.infrastructure.providers.vllm_llm import VllmJudge

            return VllmJudge(self._vllm_small)
        return LexicalEntailmentJudge()

    @cached_property
    def question_writer(self) -> QuestionWriter:
        if self.settings.profile == "cloud":
            from code_rag.infrastructure.providers.anthropic_llm import ClaudeQuestionWriter

            return ClaudeQuestionWriter(self.anthropic_client, self.settings.cloud.synth_model)
        from code_rag.infrastructure.providers.offline import TemplateQuestionWriter

        return TemplateQuestionWriter()

    @cached_property
    def tracer(self) -> Tracer:
        t = self.settings.tracing
        if t.backend == "langfuse":
            if not (t.langfuse_public_key and t.langfuse_secret_key):
                raise ConfigurationError("langfuse tracing requires public and secret keys")
            return LangfuseTracer(
                t.langfuse_public_key,
                t.langfuse_secret_key.get_secret_value(),
                t.langfuse_host,
                t.environment,
            )
        if t.backend == "otel":
            return OtelTracer()
        return RecordingTracer()

    # ---- application services ------------------------------------------------------------
    def ingest_dependencies(self) -> IngestDependencies:
        from code_rag.application.ingest.workflow import IngestDependencies

        if self.readonly:
            raise ConfigurationError("ingestion is not available in the read-only 'api' role")
        return IngestDependencies(
            source=self.source,
            parser=self.parser,
            ledger=self.ledger,
            vectors=self.vectors,
            lexical=self.lexical,
            graph=self.graph,
            code_embedder=self.code_embedder,
            summary_embedder=self.summary_embedder,
            summarizer=self.summarizer,
            batch_summarizer=self.batch_summarizer,
            tracer=self.tracer,
            write_lock=self._write_lock,
        )

    @cached_property
    def ingest_workflow(self) -> IncrementalIngestWorkflow:
        from code_rag.application.ingest.workflow import IncrementalIngestWorkflow

        return IncrementalIngestWorkflow(self.ingest_dependencies())

    @cached_property
    def retrieval_agent(self) -> HybridRetrievalAgent:
        from code_rag.application.citation.verifier import CitationVerifier
        from code_rag.application.retrieval.agent import HybridRetrievalAgent, RetrievalDependencies

        r = self.settings.retrieval
        deps = RetrievalDependencies(
            vectors=self.vectors,
            lexical=self.lexical,
            graph=self.graph,
            code_embedder=self.code_embedder,
            summary_embedder=self.summary_embedder,
            reranker=self.reranker,
            expander=self.expander if r.query_expansion else None,
            synthesizer=self.synthesizer,
            verifier=CitationVerifier(self.ledger),
            tracer=self.tracer,
        )
        return HybridRetrievalAgent(deps, r)

    async def aclose(self) -> None:
        if "parser" in self.__dict__:
            self.parser.close()
        if "lexical" in self.__dict__ and not self.readonly:
            self.lexical.close()
        if "qdrant_client" in self.__dict__:
            await self.qdrant_client.close()
        if "graph" in self.__dict__:
            self.graph.close()
        if "tracer" in self.__dict__:
            self.tracer.flush()
