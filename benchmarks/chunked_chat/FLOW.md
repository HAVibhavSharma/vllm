# chunked_chat end-to-end flow

The chunked_chat benchmark exercises a non-trivial path through vLLM:

```
client ─► /v1/chunked_chat/completions ─► serving (chunked_chat)
                                              │
                                              ├─ render chat template
                                              ├─ compute per-chunk token spans
                                              ├─ stamp vllm_xargs.anchor_pool_spans
                                              └─ forward to standard chat path
                                                       │
                                                       ▼
                                              engine + worker (forward)
                                                       │
                                              ┌────────┴─────────┐
                                              │  GPUModelRunner  │
                                              │  _update_states  │ ─► AnchorPoolRunnerState
                                              │  hook fires      │     .process_pending
                                              └──────────────────┘            │
                                                                              ▼
                                       capture_placeholder_base   (first observation)
                                          OR
                                       predict_as_anchor  →  capture_anchor (admit only)
```

This document covers the request lifecycle, the components involved, and
how to run the benchmark. The capture pipeline is wired and exercised by
the benchmark; **the reuse path (skipping prefill on cache hits) is not
implemented** — see *Limitations* at the end.

---

## Components

### 1. `/v1/chunked_chat/completions` route
File: `vllm/entrypoints/openai/chunked_chat/api_router.py`

Standard FastAPI route. Validates JSON, calls the serving handler,
returns either a `JSONResponse` or a streaming response.

### 2. `OpenAIServingChunkedChat`
File: `vllm/entrypoints/openai/chunked_chat/serving.py`

Subclass of `OpenAIServingChat`. Inherits all of its rendering, tool,
and response-formatting machinery; adds chunk-aware orchestration on
top.

Per request:

1. **Lazy install.** First request triggers
   `engine_client.collective_rpc("anchor_pool_install")` which forces
   the worker extension to attach state and returns the cache /
   tokenizer dims. Cached after the first success.
2. **Render and span-compute.** Builds a single user message from
   `"".join(chunks)`, renders via `tokenizer.apply_chat_template(...,
   tokenize=False, add_generation_prompt=True)`, then walks each chunk
   in order. For each chunk listed in `anchor_indices`, computes its
   `(t_start, num_tokens)` by tokenizing the rendered prefix up to its
   start and end character offsets.
3. **Stamp xargs.** Inserts
   `vllm_xargs["anchor_pool_spans"] = [{"chunk_hash", "t_start",
   "num_tokens"}, ...]` plus tuning knobs (`anchor_pool_entropy_threshold`,
   `anchor_pool_top_p`).
4. **Forward** to `OpenAIServingChat.create_chat_completion`. The
   downstream chat pipeline tokenizes through the same chat template,
   so spans line up with the engine's view of the prompt.

`chunk_hash` is `sha256(chunk_text)` — content-stable across runs and
across requests, which is exactly what an anchor pool keyed by static
content needs.

### 3. `AnchorPoolWorkerExtension`
File: `vllm/entrypoints/openai/cached_chat/anchor_pool_worker_ext.py`

Worker-side mixin loaded via `--worker-extension-cls`. Holds the
manager and prefix store, exposes RPCs, and **owns the engine-loop
orchestration**.

The orchestration is `_AnchorPoolRunnerState.process_pending(scheduler_output)`,
mirroring the pattern used by `cached_chat/worker_ext.py`:

1. **Drop pending entries** for any request the scheduler just finished.
2. **Queue new requests** that carry `anchor_pool_spans` in their
   `sampling_params.extra_args`. Record `(spans, num_prompt_tokens,
   threshold, top_p)` keyed by `req_id`.
3. **For each pending request**, check if `req_state.num_computed_tokens
   >= num_prompt_tokens`. If so, prefill is complete and we can safely
   gather KV for each span.
4. **Per span**, dispatch to `_handle_span`:
   - If the pool for this `chunk_hash` doesn't exist yet, or exists but
     has no base: `capture_placeholder_base`. The captured K/V is
     de-rotated to canonical position 0 so it's reusable across
     different surrounding contexts.
   - Otherwise (pool + base exist): `predict_as_anchor_for_request`
     computes Shannon entropy of the softmax(−L2-distance) distribution
     over existing anchors. If `entropy > threshold · log2(N)`, admits a
     new anchor via `capture_anchor` (prefix-less, since the benchmark
     has no static prefix between anchors). If it doesn't admit, the
     top-p subset of similar anchors gets their `activation_count`
     bumped — which protects them from LFU eviction next time the pool
     hits its cap of 20.

The runner hook that calls `process_pending`:

```python
# vllm/v1/worker/gpu_model_runner.py, in _update_states
anchor_pool_state = getattr(self, "_anchor_pool_state", None)
if anchor_pool_state is not None:
    anchor_pool_state.process_pending(scheduler_output)
```

Lazy attach means the hook is a no-op until the first RPC touches the
runner; servers not using chunked_chat see zero overhead.

### 4. `AnchorPool` + `AnchorPoolManager`
File: `vllm/entrypoints/openai/cached_chat/anchor_pool.py`

- One `AnchorPool` per static chunk, keyed by `chunk_hash`. Locks
  `num_tokens` at construction time.
- Each pool owns one canonical `(K_base, V_base)` (de-rotated, position
  0) and an LFU-evicted dict of `AnchorEntry`. Cap default 20.
- Each `AnchorEntry` holds six tensors:
  - `placeholder_key_embedding`, `placeholder_value_embedding` — real
    K/V observed during this anchor's specific surrounding context.
    Used as similarity keys for entropy-gated admission.
  - `placeholder_key_delta`, `placeholder_value_delta` — `real − base`.
  - `prefix_key_delta`, `prefix_value_delta` — left zero in this
    benchmark (no static prefix segments).
  - `activation_count` — incremented when this anchor is in the top-p
    nearest set during a non-admitted prediction.

`AnchorPoolManager` is a thin façade owning the pools by hash, with
idempotent `register_pool(...)` that validates `num_tokens` on re-entry.

---

## Per-request lifecycle (3 stages)

### Stage 1 — first request, anchor 0 (prefix.txt) seen for the first time

```
serving:    spans = [{hash=A, t_start=12, num_tokens=160},
                     {hash=S, t_start=812, num_tokens=42}]
            xargs.anchor_pool_spans = spans
serving:    forward to chat completion
runner:     scheduler schedules full prefill (no cache hit, normal path)
hook:       process_pending sees new req → queue spans
           [next step] num_computed_tokens >= num_prompt_tokens
hook:       _handle_span(A):  pool(A) missing  → capture_placeholder_base
                                                gather KV @ t_start=12, len=160
                                                de-rotate K  → set_base(K_base, V_base)
            _handle_span(S):  pool(S) missing  → capture_placeholder_base
                                                gather KV @ t_start=812, len=42
                                                de-rotate K  → set_base(...)
response:   normal chat completion returned
state:      pool(A) base set, 0 anchors.  pool(S) base set, 0 anchors.
```

### Stage 2 — second request, same anchors but new dynamic content

```
serving:    spans recomputed (anchor token positions can shift because
            the dynamic chunk between them changed length)
hook:       _handle_span(A): pool(A) has base → predict_as_anchor
                                                gather V @ new t_start, len=160
                                                N=0 anchors → admit=True
                                              → capture_anchor (prefix-less)
                                                store ph_embedding, ph_delta,
                                                leave pf_delta zero
            _handle_span(S): pool(S) has base → admit=True (still N=0)
                                              → capture_anchor
state:      pool(A) 1 anchor.  pool(S) 1 anchor.
```

### Stage 3 — third request

```
hook:       _handle_span(A): predict_as_anchor with N=1
                              entropy = 0, max_entropy = 0 (special-cased
                              to 0 for N=1), threshold * max = 0
                              entropy > 0 ? → no → admit=False
                              top-p subset = [the one anchor] → activation++
            _handle_span(S): same path
state:      pool(A) 1 anchor with activation_count=1.  Same for S.
```

### Stages 4+ — pool grows as new contexts produce diverse candidates

For each subsequent request, `predict_as_anchor` evaluates the current
candidate against the stored anchors' `placeholder_value_embedding`
distribution. Diverse contexts produce high-entropy distributions
(no clear nearest neighbour) → admit. Redundant contexts produce
peaked distributions (one anchor is obviously the closest) → don't admit,
just bump that anchor's activation count.

Once the pool hits the cap (20), LFU eviction picks the least-activated
anchor for replacement. Insertion order breaks ties so the oldest at the
lowest count loses first.

---

## How to run the benchmark

### 1. Start the server with both worker extensions

```bash
VLLM_USE_DEEP_GEMM=0 vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --dtype half --quantization awq_marlin \
    --no-enable-prefix-caching \
    --worker-extension-cls vllm.entrypoints.openai.cached_chat.worker_ext.ManualKVWorkerExtension \
    --worker-extension-cls vllm.entrypoints.openai.cached_chat.anchor_pool_worker_ext.AnchorPoolWorkerExtension
```

The two `--worker-extension-cls` flags mix both extensions into the
worker class. They don't share attribute names, so no conflict.

Prefix caching stays disabled because we want the anchor pipeline to
own the reuse story; otherwise the engine cache would mask whether
anchors are doing anything.

### 2. Run the benchmark

```bash
python benchmarks/chunked_chat/benchmark_chunked_chat.py \
    --templates-dir benchmarks/chunked_chat/templates/photosynthesis \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --warmup
```

You should see, in the server log:

```
[chunked_chat] anchor pool installed: {'block_size': 16, ...}
[chunked_chat] anchor span chunk_idx=0 hash=<prefix-hash> t_start=... num_tokens=...
[chunked_chat] anchor span chunk_idx=2 hash=<suffix-hash> t_start=... num_tokens=...
[anchor-pool] queued req=<id> spans=2 prompt_tokens=...
[anchor-pool] base captured chunk=<prefix-hash> tokens=...
[anchor-pool] base captured chunk=<suffix-hash> tokens=...
```

And on subsequent stages:

```
[anchor-pool] predict chunk=<prefix-hash> admit=True entropy=... n=0
[anchor-pool] anchor captured chunk=<prefix-hash> anchor=anc-...
...
[anchor-pool] predict chunk=<suffix-hash> admit=False entropy=... n=3 activated=2
```

### 3. Inspect pool state at any time

```bash
python -c "
import json, urllib.request
req = urllib.request.Request(
    'http://localhost:8000/v1/chunked_chat/...', method='POST')
# Or use a direct RPC if exposed. For now, the status RPC isn't routed
# through HTTP — read it from server logs."
```

(The `anchor_pool_status` RPC exists at the worker level but isn't
wired to an HTTP route yet; it's accessible from inside the engine
process.)

---

## Knobs

xargs (set per-request via `vllm_xargs`):
- `anchor_pool_spans` — list of `{chunk_hash, t_start, num_tokens}`.
- `anchor_pool_entropy_threshold` — default 0.3. Lower = more admissive.
- `anchor_pool_top_p` — default 0.9. Larger = more anchors get
  activated per non-admission.

Server-level defaults (in serving.py):
- `self._anchor_pool_threshold = 0.3`
- `self._anchor_pool_top_p = 0.9`

`AnchorPool` constants:
- `max_entries = 20` per pool (matches the reference).
- LFU eviction with insertion-order tiebreak.

---

## Prefill skipping via `AnchorPoolConnector` (option A)

Enable with the standard vLLM `--kv-transfer-config` flag. The
connector lives at
`vllm.distributed.kv_transfer.kv_connector.v1.anchor_pool_connector` and
is registered in the factory as `"AnchorPoolConnector"`.

### How it splits work

```
                                 ┌──────────────────────────────┐
                                 │ AnchorPoolConnector          │
                                 │ (SCHEDULER role)             │
                                 ├──────────────────────────────┤
serving stamps xargs ─────────►  │ get_num_new_matched_tokens   │
                                 │   reads anchor_pool_spans    │
                                 │   returns longest contiguous │
                                 │   leading-prefix of cached   │
                                 │   tokens                     │
                                 │ build_connector_meta         │
                                 │   resolves block_ids and     │
                                 │   emits inject ops           │
                                 │ request_finished             │
                                 │   marks hashes populated     │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │ AnchorPoolConnector          │
                                 │ (WORKER role)                │
                                 ├──────────────────────────────┤
                  pre-forward ─► │ start_load_kv                │
                                 │   for each inject op:        │
                                 │     base = pool.base()       │
                                 │     K = rerotate(K_base,t)   │
                                 │     scatter into block_ids   │
                                 │ save_kv_layer/wait_for_save  │
                                 │   no-ops (capture is in      │
                                 │   the worker_ext hook)       │
                                 └──────────────────────────────┘
```

The scheduler and worker sides share state via:
- `chunk_hash` populated set: scheduler tracks it; updates on
  `request_finished`. This is **optimistic** — assumes the worker
  extension's `process_pending` successfully captured the bases. Tighten
  later by passing the actual capture confirmation back through
  `build_connector_worker_meta`.
- Inject metadata: scheduler-side `build_connector_meta` emits an
  `AnchorPoolConnectorMetadata` instance carrying the per-step inject
  ops (with block_ids resolved). Worker reads via the standard
  `_get_connector_metadata()` hook.

### What gets injected

Currently **just the pool's base K/V**, re-rotated to the request's
position offset. No blending — blending requires the post-prefill
`real_ph_kv`, which by definition doesn't exist when we're skipping
prefill. The reference flow blends because it ran a (cheap) prefill
first; we're avoiding that. Trade-off: the inject reproduces the
canonical no-prefix KV for this chunk, which is accurate when the
surrounding context is similar to what produced the base. Sufficient
quality is an empirical question for downstream evaluation.

### How to enable

```bash
VLLM_USE_DEEP_GEMM=0 vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --dtype half --quantization awq_marlin \
    --no-enable-prefix-caching \
    --worker-extension-cls vllm.entrypoints.openai.cached_chat.anchor_pool_worker_ext.AnchorPoolWorkerExtension \
    --kv-transfer-config '{
        "kv_connector": "AnchorPoolConnector",
        "kv_role": "kv_both"
    }'
```

(You can keep the cached_chat `ManualKVWorkerExtension` flag too if
you're also using `/v1/cached_chat`.)

### Expected log signatures

First request (no cache):
```
AnchorPoolConnector initialized role=SCHEDULER
AnchorPoolConnector initialized role=WORKER
[chunked_chat] dispatching with 2 anchor spans
AnchorPoolConnector: claim req=... tokens=0 ops=0     ← nothing populated yet
[anchor-pool] base captured chunk=...
AnchorPoolConnector: mark populated 2 hashes (total=2)
```

Second request (anchor for the *leading* prefix chunk is populated):
```
AnchorPoolConnector: claim req=... tokens=<N> ops=1
AnchorPoolConnector: dispatching 1 inject ops to worker
AnchorPoolConnector: injected req=... chunk=... t_start=0 tokens=<N>
[anchor-pool] predict chunk=... admit=...
```

Because the connector only skips a contiguous leading prefix, the
suffix anchor (chunk index 2) cannot be skipped on its own — its tokens
sit after the dynamic body, which the scheduler must still prefill. The
suffix's KV is still useful for the *next* surrounding context, and the
predict/capture pipeline still grows the pool for it.

## Limitations

1. **Optimistic population tracking.** The scheduler marks hashes
   populated when a request finishes, not when the worker confirms
   capture. A race-condition or failed capture leaves a stale entry
   that would cause an inject to silently skip with a warning. For
   benchmark workloads this isn't observable; tighten via worker→
   scheduler ack metadata before any production use.

2. **Token-span computation is approximate** for tokenizers whose
   sub-substring boundaries don't align cleanly. For Qwen2.5 and the
   provided photosynthesis template (chunks separated by clear newline
   boundaries), spans land on the correct token positions. Other
   templates may produce off-by-one spans; this would manifest as the
   wrong slice of KV being captured.

3. **RoPE de-rotation is implemented for the standard llama-style
   `RotaryEmbeddingBase` only.** Models with exotic rotary variants
   (DeepSeek scaling, mRoPE) will find a module but the inverse-
   rotation math has only been verified against the plain RoPE path.

4. **Multi-worker (TP > 1) hasn't been verified.** Each worker
   independently maintains its own pool keyed by `chunk_hash`; the
   pools should converge on the same content across workers, but the
   collective_rpc fan-out behaviour at install time is untested.

---

## Files touched

| Path | Role |
| ---- | ---- |
| `vllm/entrypoints/openai/chunked_chat/serving.py` | chunk-aware orchestration; computes spans, stamps xargs |
| `vllm/entrypoints/openai/cached_chat/anchor_pool.py` | `AnchorPool`, `AnchorEntry` (6 tensors + activation), `PrefixSegmentStore` |
| `vllm/entrypoints/openai/cached_chat/anchor_pool_worker_ext.py` | worker extension, `process_pending` hook, RoPE de-rotation, blend math |
| `vllm/v1/worker/gpu_model_runner.py` | added `_anchor_pool_state.process_pending(...)` hook call |
| `benchmarks/chunked_chat/templates/photosynthesis/manifest.json` | benchmark manifest (was missing) |
