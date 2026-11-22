"""Code-aware tokenization shared by BM25 indexing and offline adapters."""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+|[0-9]+")
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "is",
        "it",
        "for",
        "on",
        "with",
        "as",
        "at",
        "by",
        "be",
        "this",
        "that",
        "from",
        "how",
        "what",
        "where",
        "which",
        "does",
        "do",
        "when",
        "why",
        "who",
        "self",
        "return",
        "def",
        "func",
        "fn",
        "let",
        "const",
        "var",
        "import",
        "pub",
        "if",
        "else",
        "none",
        "null",
        "true",
        "false",
        "nil",
        "new",
        "class",
        "struct",
    }
)


def split_identifier(token: str) -> list[str]:
    """`getUserByID` -> [get, user, by, id]; `parse_money` -> [parse, money]."""
    parts: list[str] = []
    for chunk in token.split("_"):
        if chunk:
            parts.extend(m.group(0).lower() for m in _CAMEL.finditer(chunk))
    return parts


def code_tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for m in _IDENT.finditer(text):
        raw = m.group(0)
        subs = split_identifier(raw)
        low = raw.lower()
        if len(subs) > 1:
            out.append(low)
        out.extend(subs)
    if keep_stopwords:
        return out
    return [t for t in out if t not in _STOP and len(t) > 1]


def expand_identifiers(text: str) -> str:
    """Text for the BM25 'code' analyzer: original identifiers plus their sub-tokens."""
    return " ".join(code_tokens(text, keep_stopwords=True))


_IDENT_IN_QUESTION = re.compile(r"`([^`]+)`|\b([A-Za-z_]+(?:[A-Z][a-z0-9]+)+|[a-z0-9]+(?:_[a-z0-9]+)+)\b")


def identifiers_in(question: str) -> list[str]:
    """Extract likely code identifiers (backticked, camelCase, snake_case) from natural language."""
    found: list[str] = []
    for m in _IDENT_IN_QUESTION.finditer(question):
        tok = (m.group(1) or m.group(2) or "").strip()
        if tok and tok not in found:
            found.append(tok)
    return found
