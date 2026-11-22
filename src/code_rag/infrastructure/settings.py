"""Typed configuration: TOML file (CODE_RAG_CONFIG) + environment overrides (CODE_RAG_*, `__` nesting)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from code_rag.application.retrieval.config import RetrievalConfig
from code_rag.domain.errors import ConfigurationError

Profile = Literal["cloud", "local", "offline"]
Role = Literal["all", "api", "worker"]


class QdrantSettings(BaseModel):
    url: str | None = None  # None => embedded on-disk mode under data_dir (single process)
    api_key: SecretStr | None = None
    prefix: str = "chunks"
    server_side_boost: bool = True


class CloudSettings(BaseModel):
    voyage_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    code_embedding_model: str = "voyage-code-3"
    summary_embedding_model: str = "voyage-3.5"
    embedding_dim: int = 1024
    rerank_model: str = "rerank-2.5"
    summarizer_model: str = "claude-haiku-5-5"
    expander_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-haiku-5-5"
    synth_model: str = "claude-opus-5-5"
    synth_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    enable_refusal_fallback: bool = True
    use_batches_for_bulk: bool = True
    summarizer_concurrency: int = 32


class LocalSettings(BaseModel):
    tei_code_embed_url: str = "http://localhost:8081"
    code_embedding_model: str = "nomic-ai/nomic-embed-code"
    code_embedding_dim: int = 3584
    tei_summary_embed_url: str = "http://localhost:8082"
    summary_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    summary_embedding_dim: int = 768
    tei_rerank_url: str = "http://localhost:8083"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    vllm_url: str = "http://localhost:8000"
    vllm_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    vllm_small_url: str | None = None
    vllm_small_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"


class OfflineSettings(BaseModel):
    embedding_dim: int = 512


RetrievalSettings = RetrievalConfig


class TracingSettings(BaseModel):
    backend: Literal["none", "langfuse", "otel"] = "none"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    environment: str = "production"


class WebhookSettings(BaseModel):
    github_secret: SecretStr | None = None
    gitlab_secret: SecretStr | None = None
    allowed_branches: dict[str, str] = Field(default_factory=dict)  # repo -> branch; default 'main'


class RepoConfig(BaseModel):
    name: str
    remote: str | None = None
    path: str | None = None
    default_branch: str = "main"


class ReposFile(BaseModel):
    repos: list[RepoConfig] = Field(default_factory=list)
    generated_globs: list[str] = Field(default_factory=list)
    exclude_globs: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODE_RAG_", env_nested_delimiter="__", extra="ignore")

    profile: Profile = "offline"
    role: Role = "all"
    data_dir: Path = Path("./data")
    repos_file: Path = Path("config/repos.yaml")
    redis_url: str = "redis://localhost:6379/0"
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    local: LocalSettings = Field(default_factory=LocalSettings)
    offline: OfflineSettings = Field(default_factory=OfflineSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    webhooks: WebhookSettings = Field(default_factory=WebhookSettings)
    parse_workers: int | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        config_path = os.environ.get("CODE_RAG_CONFIG")
        if config_path:
            if not Path(config_path).exists():
                raise ConfigurationError(f"CODE_RAG_CONFIG points to a missing file: {config_path}")
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=config_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    def load_repos(self) -> ReposFile:
        if not self.repos_file.exists():
            return ReposFile()
        raw = yaml.safe_load(self.repos_file.read_text()) or {}
        return ReposFile.model_validate(raw)
