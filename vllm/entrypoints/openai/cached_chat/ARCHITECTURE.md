# cached_chat — Manual KV-Cache Reuse

A custom vLLM endpoint (`/v1/cached_chat/completions`) that stores prefix
KV-cache blocks in pinned host memory after one prompt is prefilled, then
copies them back into freshly-allocated GPU blocks the next time the same
prompt arrives — the goal being to skip repeated prefill of identical
prompts when vLLM's built-in prefix cache is disabled.

## Status

| Stage                                    | Working? |
| ---------------------------------------- | -------- |
| Fingerprint a request                    | yes      |
| Reserve a handle and arm capture         | yes      |
| Capture KV blocks → pinned host memory   | yes      |
| Recall handle on second hit              | yes      |
| Inject host blob → freshly-allocated GPU | yes      |
| **Actually skip the prefill forward**    | **NO**   |

The first five components are wired up correctly (the recent dim-1 fix
made capture/inject stop crashing). What's still missing is the piece
that actually saves time. See *What's missing* at the bottom.

## Files

```
vllm/entrypoints/openai/cached_chat/
├── api_router.py    # FastAPI route /v1/cached_chat/completions
├── protocol.py      # Pydantic types (mostly extends ChatCompletionRequest)
├── serving.py       # OpenAIServingCachedChat — fingerprint + dispatch
├── store.py         # ManualKVStore — fp → handle table on the API server
└── worker_ext.py    # ManualKVWorkerExtension — capture/inject on the worker
```

`benchmarks/cached_chat/test_cached_chat.py` is the driver that fires
warmup → MISS → HIT → HIT and asserts a 1.5× speedup on the second call.

## Components

### 1. `OpenAIServingCachedChat` (serving.py)

API-server side. One instance per server. Subclasses
`OpenAIServingChat` so the wire format is identical to
`/v1/chat/completions`.

Responsibilities:

- **Fingerprint** the request (sha256 of `model + messages + tools +
  tool_choice`).
- Look up that fingerprint in `ManualKVStore`.
- **MISS** path — call `kv_store.reserve(fp)` to pre-allocate a handle,
  then stamp `manual_kv_handle=<handle>` and `manual_kv_capture=1` into
  `request.vllm_xargs`. Forward the request normally.
- **HIT** path — stamp `manual_kv_handle=<handle>` and
  `manual_kv_prefilled_tokens=<N>` into `vllm_xargs`. Forward the request
  normally.
- After the response comes back, if it was a MISS, mark the store entry
  populated using `usage.prompt_tokens` rounded down to a block boundary
  (so the next request asks for exactly what was actually captured).

Discovers the worker's `block_size` lazily via
`engine_client.collective_rpc("manual_kv_install")`. That same RPC also
installs the worker-side state.

### 2. `ManualKVStore` (store.py)

Pure Python LRU table on the API server. Maps:

```
fingerprint (sha256 hex) → ManualKVHandleEntry(handle, prefix_len, hits, populated)
```

- `reserve(fp)` — allocate a fresh handle string `mkv-<8 hex>` and put a
  placeholder entry (populated=False).
- `populate(fp, prefix_len)` — flip populated=True with the captured
  prefix length.
- LRU evict at `max_entries=64`. On evict, `on_evict` callback fires
  `manual_kv_drop` over `collective_rpc` so the worker can free the
  pinned host memory for that handle.

Note: this lives on the API process, not the engine process. It does
**not** hold the KV blob itself — only the handle string and metadata.

### 3. `ManualKVWorkerExtension` (worker_ext.py)

Worker-side. One per GPU worker. Holds the actual KV blobs:

```python
self._blobs: dict[str, list[torch.Tensor]]   # handle → [layer0_kv, layer1_kv, ...]
self._blob_lengths: dict[str, int]           # handle → tokens stored
self._pending_captures: dict[str, tuple[handle, num_prompt_tokens]]
```

Each layer's tensor is shape `(2, num_prefix_blocks, block_size,
num_kv_heads, head_size)` (FlashAttention) or
`(num_prefix_blocks, ...)` (MLA), allocated with `pin_memory=True` so
GPU↔CPU copies are async-friendly.

`process_pending(scheduler_output)` is the single hook into the
scheduler. It runs once per engine step, before
`_prepare_inputs`. It does two things:

#### a) Inject for new requests (HIT path)

For each entry in `scheduler_output.scheduled_new_reqs` whose
`sampling_params.extra_args` carries `manual_kv_prefilled_tokens > 0`:

1. Pull the blob list for that handle.
2. Clamp the requested prefix to what's actually stored.
3. Read `new_req_data.block_ids[:num_blocks]` — these are the GPU block
   IDs the scheduler just allocated for this request.
4. For each layer, `layer_kv.index_copy_(block_dim, gpu_index, host_buf)`
   with `block_dim = 1 if layout has the K/V split else 0`.

This *writes the bytes* into the GPU KV cache before the model's forward
pass for this step.

#### b) Capture after prefill (MISS path)

When `manual_kv_capture=1`, the request id gets put into
`_pending_captures` keyed by `req_id`, holding `(handle,
num_prompt_tokens)`. Each subsequent step the hook checks
`req_state.num_computed_tokens >= num_prompt_tokens`. The first step
that's true:

1. Look up the request's allocated `block_ids[:num_prompt_tokens //
   block_size]`.
2. For each layer, `layer_kv.index_select(block_dim, prefix_block_ids)`
   onto a pinned host buffer.
3. Stash in `self._blobs[handle]`, set `self._blob_lengths[handle]`,
   drop from `_pending_captures`.

#### c) RPC entry points

- `manual_kv_install()` → returns `block_size`, called once at startup.
- `manual_kv_drop(handle)` → free both `_blobs[handle]` and
  `_blob_lengths[handle]`.

The dim-1 fix (recent change) lives in two spots: `_capture_request`
(line 259-area) and `_do_inject` (line 233-area). Both now use:

```python
block_dim = 1 if sample.shape[0] == 2 else 0
```

…to handle FA's `(2, num_blocks, ...)` layout vs MLA's `(num_blocks,
...)` layout.

### 4. `api_router.py` / `protocol.py`

Just wiring. The router mounts `/v1/cached_chat/completions` on the same
FastAPI app and dispatches into `OpenAIServingCachedChat`. The protocol
extends `ChatCompletionRequest` only by way of the `vllm_xargs`
mechanism — no new fields.

## Request lifecycle (HIT case)

```
POST /v1/cached_chat/completions {messages, ...}
    │
    ▼
OpenAIServingCachedChat.create_chat_completion
    │   fp = sha256(messages...)
    │   entry = kv_store.get(fp)               ──── HIT ────►
    │   xargs[manual_kv_handle]   = entry.handle
    │   xargs[manual_kv_prefilled_tokens] = entry.prefix_len
    │
    ▼
OpenAIServingChat.create_chat_completion (super)
    │
    ▼
AsyncLLM → EngineCore → Scheduler
    │   Scheduler allocates fresh KV blocks for this request
    │   (no special handling — looks like an ordinary request)
    │
    ▼ scheduler_output.scheduled_new_reqs[i] carries the xargs
ModelRunner._update_states (vllm-side hook)
    │
    ▼ calls into ▼
ManualKVWorkerExtension.process_pending
    │   sees manual_kv_prefilled_tokens=N
    │   reads new_req_data.block_ids
    │   index_copy_  pinned host blob ──► GPU KV blocks   ◄── memory now populated
    │
    ▼
ModelRunner._prepare_inputs / forward
    │   Scheduler still thinks num_computed_tokens=0,
    │   so it schedules the FULL prompt for prefill.
    │   Forward pass runs over all 101 tokens, writes KV
    │   into the same blocks ────► overwrites the inject ◄── this is the problem.
    ▼
Response → API server
    │
    ▼ response.usage.prompt_tokens
kv_store.populate(fp, prompt_tokens // block_size * block_size)
```

## What's missing (why the test fails)

The benchmark sees `1.0×` speedup, not `1.5×`, because the inject does
not actually save any work. Three things must happen *together* on the
step a HIT request enters the batch:

| # | Required | Currently |
| - | -------- | --------- |
| 1 | KV memory contains valid bytes for the cached prefix | ✓ done by `_do_inject` |
| 2 | `req_state.num_computed_tokens` advanced to `prefilled` | ✗ never set |
| 3 | `scheduler_output.num_scheduled_tokens[req_id]` reduced from `prompt_len` to `prompt_len - prefilled` (or 1 if you want to start decoding immediately) | ✗ never reduced |

Without (2) and (3), `_prepare_inputs` lays out a full-prompt forward
pass, the model recomputes the prefix, and the freshly-injected KV gets
overwritten with the same values — no time saved.

### Two ways forward

**A. Quick scheduler-output mutation (prototype only).**
Inside `process_pending`, after a successful inject:

```python
req_state = self._runner.requests[new_req_data.req_id]
req_state.num_computed_tokens = prefilled
new_req_data.num_computed_tokens = prefilled
# scheduler_output.num_scheduled_tokens is per-req; reduce it
scheduler_output.num_scheduled_tokens[new_req_data.req_id] = (
    num_prompt_tokens - prefilled
)
```

Brittle — chunked prefill, preemption, prefix caching, and async
scheduling each interact differently. Useful only to confirm the speedup
hypothesis end-to-end before investing in (B).

**B. Proper `KVConnectorBase_V1` integration.**
The vLLM v1 engine has a first-class hook for exactly this use case:
external KV providers (NIXL, LMCache, MoonCake) all use it. The
connector reports `external_computed_tokens` to the *scheduler*, which
then plans the request as a partial prefill, allocates blocks
correctly, and lets the worker fill them. See
`vllm/distributed/kv_transfer/kv_connector/v1/` for examples.

This is the right long-term shape: the worker extension stays
responsible for the host↔GPU memory mechanics, and a new connector
class handles the scheduler-side accounting.

## Required server flags

```bash
vllm serve <model> \
  --no-enable-prefix-caching \
  --worker-extension-cls vllm.entrypoints.openai.cached_chat.worker_ext.ManualKVWorkerExtension
```

Prefix caching is disabled because the design assumes the engine has no
prior knowledge of the prefix — `cached_chat` *is* the cache.
