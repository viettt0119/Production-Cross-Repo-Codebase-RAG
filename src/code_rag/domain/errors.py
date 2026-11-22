"""Domain error hierarchy. Adapters translate vendor exceptions into these."""

from __future__ import annotations


class CodeRagError(Exception):
    """Base error."""


class ParseError(CodeRagError):
    """A source file could not be parsed into an AST."""


class UnsupportedLanguageError(ParseError):
    """No grammar/extractor exists for the file's language."""


class SourceUnavailableError(CodeRagError):
    """A blob/commit could not be read from the source repository."""


class ProviderError(CodeRagError):
    """Upstream model provider failed after retries."""

    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable


class ProviderRefusalError(ProviderError):
    """Model declined the request (stop_reason == 'refusal')."""


class IndexStoreError(CodeRagError):
    """A vector/lexical/graph store rejected an operation."""


class WebhookAuthError(CodeRagError):
    """A webhook failed signature/token verification or is malformed."""


class ConfigurationError(CodeRagError):
    """Settings are missing or inconsistent for the selected profile/role."""
