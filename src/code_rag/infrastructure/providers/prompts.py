"""Prompt templates and context formatting shared by LLM adapters (Claude and vLLM)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from code_rag.domain.entities import SymbolNode
from code_rag.domain.ports.models import FileToSummarize, QuestionSeed
from code_rag.domain.query import RankedContext

MAX_FILE_CHARS_FOR_SUMMARY = 60_000
MAX_SYMBOL_CHARS_FOR_SUMMARY = 6_000
MAX_CONTEXT_SOURCE_CHARS = 6_000

SUMMARY_SYSTEM = """You write one-sentence summaries of source-code symbols for a code search index.

For every symbol you are given, produce exactly one sentence (max 35 words) that states:
1. its public contract: what it takes and what it returns or guarantees, in domain terms;
2. its notable side effects: I/O, network/RPC calls, database or ledger writes, mutation of shared
   state, raised errors, retries, locking;
3. nothing else. No praise, no restating the name, no "This function", no markdown, no code.

Write for an engineer searching a large multi-repository codebase: prefer concrete nouns that a
search query would contain (e.g. "idempotency key", "refund", "gRPC BillingService.Charge",
"inventory reservation", "currency conversion") over vague verbs ("handles", "manages", "processes").
If a symbol is a type (class/struct/interface/trait/enum/message), describe what it represents and
its key invariants. If it is a module-level block, describe the configuration or constants it defines.
If a symbol is trivial (getter, re-export), say so in a few words.

Examples of good summaries:
- "Captures funds for an order via the ledger and returns a charge id; repeated idempotency keys
  return the original charge instead of charging twice."
- "Reserves stock for a SKU by decrementing the in-memory count, raising OutOfStockError when the
  requested quantity exceeds availability."
- "Immutable money value object with Decimal amount and ISO currency; converts to integer minor units
  using banker's rounding."
- "gRPC client wrapper around BillingServiceStub that adds a per-call deadline and forwards
  idempotency keys on Charge and Refund."

Return JSON matching the provided schema: one entry per requested symbol id, using the ids exactly
as given. Never invent symbols that were not requested."""

SUMMARY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["id", "summary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["summaries"],
    "additionalProperties": False,
}


def summary_user_prompt(item: FileToSummarize) -> str:
    header = f"Repository: {item.repo}\nFile: {item.path}\nLanguage: {item.language}\n"
    if len(item.file_source) <= MAX_FILE_CHARS_FOR_SUMMARY:
        body = f"<file>\n{item.file_source}\n</file>\n"
        listing = "\n".join(
            f"- id={s.symbol_id} kind={s.kind.value} name={s.qualified_name} lines={s.span}" for s in item.symbols
        )
    else:
        body = ""
        listing = "\n".join(
            f"- id={s.symbol_id} kind={s.kind.value} name={s.qualified_name} lines={s.span}\n"
            f"<source>\n{s.source[:MAX_SYMBOL_CHARS_FOR_SUMMARY]}\n</source>"
            for s in item.symbols
        )
    return f"{header}{body}\nSummarize these symbols:\n{listing}"


EXPAND_SYSTEM = """You rewrite developer questions about a multi-repository codebase into search queries.
Return JSON: up to 3 short alternative phrasings that use likely code vocabulary (function, class,
service, RPC and file names; synonyms such as charge/payment/billing), and a list of concrete code
identifiers that probably appear in the answer (camelCase, snake_case, PascalCase). Do not answer."""

EXPAND_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "rewrites": {"type": "array", "items": {"type": "string"}},
        "identifiers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["rewrites", "identifiers"],
    "additionalProperties": False,
}

SYNTH_SYSTEM = """You answer questions about a multi-repository codebase using ONLY the <source> blocks provided.

Citation rules (strictly enforced by an automated verifier; violations cause the answer to be rejected):
- Every sentence or bullet that states a fact about the code MUST end with at least one citation of the
  exact form (org/repo/path/to/file.ext:START-END), for example (acme/checkout-py/checkout/service.py:27-40).
- The repo is always the first two segments (org/name). Copy it from the source block's anchor.
- START-END must lie inside the line range of a source block you were given. Prefer the narrowest range
  that supports the claim. Never cite a file or lines you were not shown.
- Code snippets must be copied verbatim from the sources (you may omit lines, but never alter them).
- If the sources do not contain enough information, reply with a line starting "INSUFFICIENT_CONTEXT:"
  followed by what is missing. Do not guess.

Style: concise GitHub-flavoured markdown. Explain cross-repository flows step by step (caller -> callee),
naming the services and RPCs involved."""

JUDGE_SYSTEM = """You verify whether cited source code supports a claim about that code.
Answer with JSON {"supported": true|false}. Supported means the code shown directly substantiates the
claim; plausible-but-not-shown is false."""

JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}},
    "required": ["supported"],
    "additionalProperties": False,
}

_SKELETON_MARKER = re.compile(r"⋯ body of .+: lines (\d+)-(\d+)$")


def numbered_source(sym: SymbolNode, max_chars: int = MAX_CONTEXT_SOURCE_CHARS) -> str:
    out: list[str] = []
    line_no = sym.span.start
    used = 0
    for raw in sym.source.split("\n"):
        marker = _SKELETON_MARKER.search(raw.strip())
        if marker:
            out.append(f"     | {raw}")
            line_no = int(marker.group(2)) + 1
            continue
        rendered = f"{line_no:>4} | {raw}"
        used += len(rendered) + 1
        if used > max_chars:
            out.append("     | [truncated]")
            break
        out.append(rendered)
        line_no += 1
    return "\n".join(out)


def format_contexts(contexts: Sequence[RankedContext]) -> str:
    blocks: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        sym = ctx.chunk.symbol
        summary = f'\nsummary="{ctx.chunk.summary}"' if ctx.chunk.summary else ""
        blocks.append(
            f'<source id="{i}" anchor="{sym.citation_anchor()}" symbol="{sym.qualified_name}" '
            f'kind="{sym.kind.value}" language="{sym.language.value}"{summary}>\n'
            f"{numbered_source(sym)}\n</source>"
        )
    return "\n\n".join(blocks)


def synth_user_prompt(question: str, contexts: Sequence[RankedContext], feedback: str | None) -> str:
    parts = [format_contexts(contexts), f"\nQuestion: {question.strip()}"]
    if feedback:
        parts.append(
            "\nYour previous answer was rejected by the citation verifier:\n"
            f"{feedback}\nRewrite the complete answer, fixing every listed problem."
        )
    return "\n".join(parts)


QUESTION_SYSTEM = """You write evaluation questions for a code-search system over many repositories.
Given two related code symbols (often in different repositories and languages, linked by a call,
inheritance or a gRPC client/server relationship), write ONE realistic question a developer would ask
whose complete answer requires looking at BOTH symbols. Do not mention file paths or line numbers and do
not quote identifiers verbatim more than once; describe behaviour the way an engineer would. If the pair
is trivial or unrelated, return an empty string. Return JSON {"question": "..."}."""

QUESTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


def question_user_prompt(seed: QuestionSeed) -> str:
    return (
        f"Relation: {seed.relation}\n\n"
        f"Symbol A ({seed.src_repo}) {seed.src_name}\nSummary: {seed.src_summary}\n"
        f"<code>\n{seed.src_source}\n</code>\n\n"
        f"Symbol B ({seed.dst_repo}) {seed.dst_name}\nSummary: {seed.dst_summary}\n"
        f"<code>\n{seed.dst_source}\n</code>"
    )
