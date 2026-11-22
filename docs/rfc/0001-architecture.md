# RFC-0001: Cross-Repo Semantic Code Search & RAG — Architecture

Status: **Accepted, implemented** · Scope: 2M LOC across 10+ repositories; incremental re-index of a
50-file commit in < 60 s.

## 1. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Profiles behind ports: `cloud` (Voyage + Claude, default), `local` (TEI + vLLM), `offline` (deterministic, CI) | quality without GPUs by default; air-gapped option; full pipeline testable without keys |
| D2 | AST chunks per symbol; containers above 40 lines become skeletons | no split functions; no duplicated member text |
| D3 | Two hashes: `content_hash` (normalised AST tokens + leading doc comments) vs location | formatting/line shifts never cost an API call, yet citations stay exact |
| D4 | `symbol_id = sha1(repo|path|qualified_name|kind|ordinal)`; Qdrant id = `uuid5(symbol_id)` | stable across edits; idempotent upserts; deletes = `old_ids − new_ids` per file |
| D5 | References stored by name, resolved late (imports, suffix match, receiver heuristics, ambiguity cap) and re-resolved when target names appear/disappear | cross-repo edges self-heal regardless of indexing order |
| D6 | gRPC modelled through `RpcEndpoint` nodes (client `CALLS_RPC`, server `SERVES_RPC`, proto `DECLARES_RPC`) | languages are indexed independently and still meet at the endpoint |
| D7 | Single writer (worker) + atomic graph snapshots for read-only API processes | embedded graph DBs and Tantivy allow one writer |
| D8 | Stores written before the ledger; `indexed_sha` advanced last; diffs computed from ledger sha | crash-safe, replay-safe, out-of-order-webhook-safe |
| D9 | Per-file summarization with cached system prompt; Batches API for bulk | 50 requests instead of ~600 for a 50-file commit; 50% cost for initial indexing |
| D10 | Citation verifier: indexed file + within line count + overlaps a provided chunk; verbatim code snippets; re-ask with feedback; prune or abstain | zero tolerance for fabricated anchors |
| D11 | LadybugDB instead of Kùzu | `kuzudb/kuzu` archived 2025-10-10; Ladybug is the maintained fork with the same API |

## 2. Latency budget (cloud, 50-file commit, worst case ~600 changed symbols)

| Step | Budget |
|---|---|
| git diff + read blobs + parse (process pool) | 1–2 s |
| hash diff | < 0.1 s |
| summaries: 50 per-file requests, 32 concurrent | 6–12 s (critical path) |
| code embeddings (parallel with summaries) | 2–4 s |
| summary embeddings | 2–4 s |
| Qdrant (server) + Tantivy + graph writes, ledger | 1–3 s |
| resolution + snapshot publish | 1–2 s |
| **total** | **~15–25 s** |

Measured offline (no model latency) on 216k LOC with embedded Qdrant: 12–17 s per 50-file commit,
dominated by embedded-Qdrant writes (see README).

## 3. Query pipeline (LangGraph)

`expand_query → retrieve_hybrid (dense code ∥ dense summary ∥ BM25F, per original + rewrites) →
fuse_rrf (weighted, k=60) → expand_graph (RPC bridge first, cross-repo calls, parents) → rerank
(cross-encoder; generated-code penalty; container demotion; 2 slots reserved for graph neighbours of
the top-3) → synthesize (streamed) ⇄ verify (≤3 attempts) → finalize (prune / abstain)`.

## 4. Schemas

See `src/code_rag/infrastructure/stores/` — `graph_store.DDL`, `tantivy_index._schema`,
`qdrant_index._payload`, `sqlite_hash_ledger._SCHEMA` are the authoritative definitions.
