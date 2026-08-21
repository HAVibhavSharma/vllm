# `plan/` — design & reference docs (index)

These are **design/implementation plans and reference notes**, not always
as-built documentation. Several predate the API refactor and can drift from the
shipped code. When a doc and the code disagree, **the code wins** — verify
against the referenced source files.

## Authoritative code-level docs (start here for the serving layer)

The current serving/entrypoint architecture is documented *next to the code*,
not in this folder:

- **[`vllm/entrypoints/openai/cached_chat/ARCHITECTURE.md`](../vllm/entrypoints/openai/cached_chat/ARCHITECTURE.md)**
  — the `cached_chat` manual KV-cache-reuse endpoint
  (`POST /v1/cached_chat/completions`): fingerprint → capture prefix KV blocks
  to pinned host memory → inject back into fresh GPU blocks on the next
  identical prompt. Includes a live **Status** table of what works vs. what is
  still missing (as of writing, the prefill-skip step is not yet landed). This
  is the most current source for the serving layer and supersedes the
  entrypoint sections of the plans below.

Related shipped entrypoint packages (no standalone doc yet — read the code):
`vllm/entrypoints/openai/agent_chat/`, `.../chunked_chat/`, `.../generate/`.

## Documents in this folder

| Doc | What it covers | As-built accuracy |
|---|---|---|
| [`agent_prefetch_plan.md`](agent_prefetch_plan.md) | Agent-scoped KV prefix prefetch (`/v1/agents/prefetch`, registry, phantom prefetches). | **Implemented with drift** — has an "As-built deltas" table at the top; trust that over the body. |
| [`agent_apis.md`](agent_apis.md) | Agent API surface / usage. | Verify endpoints against `agent_chat/api_router.py`. |
| [`architecture.md`](architecture.md) | Overall LMCache-integration architecture. | Design-level; references live files. |
| [`block_lifecycle.md`](block_lifecycle.md) | KV block lifecycle. | Reference. |
| [`l0_eviction.md`](l0_eviction.md) | L0 / APC eviction behavior. | Reference. |
| [`prefill_computation.md`](prefill_computation.md) | Prefill token-split computation. | Recent (tracks current commits). |
| [`num_computed_tokens_output.md`](num_computed_tokens_output.md) | `num_computed_tokens` accounting / job stats. | Recent (tracks current commits). |
| [`kv_reuse_provenance.md`](kv_reuse_provenance.md) | Inter-question KV reuse: which job's prefill paid for the blocks a request hits (`VLLM_KV_PROVENANCE`, `kv_reuse` line, `GET /v1/kv_metrics`). | **As-built.** |
| [`prefix_token_sweep.md`](prefix_token_sweep.md) | Prefix token sweep experiment. | Reference. |
| [`commands.md`](commands.md) | Run/serve commands used during development. | Verify flags against current CLI. |
