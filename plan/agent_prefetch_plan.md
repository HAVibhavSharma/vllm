# Agent-Scoped KV Prefix Prefetch — Design & Implementation Plan

## 1. Goal

Add a new vLLM API path that:

1. Accepts a request shaped like `/v1/chat/completions` plus two extra fields:
   - `agent_id` (string, required) — identifies a logical caller / session / agent.
   - `prefetch_top_k` (int, optional, default 20) — how many of this agent's most
     recent prefix hashes to proactively preload into the GPU paged KV cache
     *before* the request's prefill phase starts.
2. Maintains a per-agent registry of recently used prefix hashes
   (token-id sequences hashed at LMCache chunk granularity).
3. On each call, looks up the registry, fires "phantom" prefetch jobs that
   populate vLLM's APC (Automatic Prefix Cache) directly from LMCache CPU L1,
   then services the real chat completion which is now expected to hit APC
   and skip both CPU→GPU retrieve and prefill for the cached prefixes.
4. Updates the registry with the prefix the current call actually used,
   keeping at most the last N (configurable, default 20) per agent under LRU.

This builds on the existing `LMCacheMPConnector` integration described in
`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`.

---

## 2. Non-goals

- No changes to LMCache MP server (`lmcache server`) wire protocol. The server
  already exposes RETRIEVE; we reuse it as-is.
- No support for the in-process LMCache integration (`internal_api_server`)
  — MP mode only.
- No persistent storage of the agent → hashes registry across vLLM restarts in
  v1. (Optional follow-up: back it with Redis or LMCache's own metadata.)
- No multi-tenant authentication. `agent_id` is treated as an opaque key.
  Auth/quotas should live in a fronting layer.
- Hybrid KV cache manager is **not** supported (existing connector constraint
  via `reformat_block_ids` at `lmcache_mp_connector.py:69`). Server must be
  started with `--disable-hybrid-kv-cache-manager`.

---

## 3. High-level architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Caller (your app)                                                     │
│   POST /v1/agents/chat/completions                                     │
│   { agent_id, prefetch_top_k, messages, ... }                          │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  AgentPrefetchRouter (new)               │
        │   1. tokenize prompt → token_ids         │
        │   2. lookup AgentPrefixRegistry[agent_id]│
        │      → up to N recent prefix hashes      │
        │   3. fan out PhantomPrefetchRequest per  │
        │      hashed prefix (async, fire&forget)  │
        │   4. submit real Request to engine       │
        │   5. on completion, push new prefix hash │
        │      into registry (LRU eviction)        │
        └─────┬────────────────────────────────┬──┘
              │                                │
              ▼                                ▼
   ┌────────────────────────┐    ┌────────────────────────────────┐
   │ AgentPrefixRegistry    │    │ vLLM Engine (existing)         │
   │  agent_id -> deque[20] │    │  ─ Scheduler                   │
   │  of PrefixDescriptor   │    │  ─ KVCacheManager / BlockPool  │
   │  in-mem, LRU, locked   │    │  ─ LMCacheMPConnector          │
   └────────────────────────┘    │  ─ Workers                     │
                                 └────────────────────────────────┘
```

### Two request types submitted to the engine

- **Real request** — normal chat completion. Carries `kv_transfer_params =
  {"cache_salt": <agent_id-derived>, "agent_id": agent_id}`. No code change in
  the engine path required.
- **Phantom prefetch request** — synthetic, `max_tokens=0`, carries
  `kv_transfer_params = {"prefetch_only": True, "cache_salt": <salt>}`. Goes
  through the scheduler exactly like a normal request, gets blocks allocated,
  `start_load_kv` fires a RETRIEVE to LMCache, then the request is marked
  finished *before any forward pass runs*. Blocks are released to APC's
  free-but-cached pool so the next real request hits APC.

The phantom path is Option A from the prior discussion. It rides on the
existing async-load mechanism of the connector — the new code is just a flag
that says "skip the prefill stage once the load completes" and a tiny scheduler
branch that honors it.

---

## 4. Data model

### 4.1 `PrefixDescriptor`

```python
@dataclass(frozen=True)
class PrefixDescriptor:
    # Token ids covering exactly one or more full LMCache chunks
    # (length is a multiple of chunk_size, e.g. 16).
    token_ids: tuple[int, ...]
    # Stable hash over (model_name, cache_salt, token_ids). Used as the
    # registry key and as a quick dedup check before re-submitting prefetches.
    prefix_hash: bytes
    # Per-agent cache salt forwarded to LMCache key derivation.
    cache_salt: str
    # Monotonic last-use timestamp (for LRU within the agent's deque).
    last_used_ns: int
```

`prefix_hash` is computed with the same hash family LMCache uses
(`TokenHasher.hash_to_bytes` chained, plus model_name + cache_salt) so the
hash we cache equals the chunk-key suffix LMCache will look up. This lets us
short-circuit a prefetch if LMCache obviously can't have it.

### 4.2 `AgentPrefixRegistry`

```python
class AgentPrefixRegistry:
    """
    Per-agent LRU of recent prefix descriptors.

    Thread-safe.  In-memory only.  Bounded total size to prevent runaway
    memory growth (default: 10k agents × 20 entries × ~few KB).
    """

    def __init__(self,
                 default_top_k: int = 20,
                 max_agents: int = 10_000,
                 max_per_agent: int = 64):
        self._lock = threading.RLock()
        # agent_id -> OrderedDict[prefix_hash, PrefixDescriptor]
        # OrderedDict gives O(1) LRU promotion via move_to_end.
        self._by_agent: OrderedDict[str, OrderedDict[bytes, PrefixDescriptor]] = ...

    def record(self, agent_id: str, desc: PrefixDescriptor) -> None: ...
    def top_k(self, agent_id: str, k: int) -> list[PrefixDescriptor]: ...
    def evict_agent(self, agent_id: str) -> None: ...
    def stats(self) -> dict: ...
```

Eviction:
- Within an agent: LRU once `max_per_agent` is exceeded.
- Across agents: when `max_agents` is exceeded, drop the least-recently-touched
  agent entirely.

### 4.3 Splitting an incoming prompt into recordable prefixes

A single request typically has one "prefix" worth recording — the chunk-aligned
prefix of the prompt. The registry stores **one descriptor per (agent, unique
prefix)** keyed by `prefix_hash`. On a repeat call, `record()` does
`move_to_end(...)` instead of inserting a new entry. We do **not** record every
possible sub-prefix; only the chunk-aligned prefix of length `floor(prompt_len /
chunk_size) * chunk_size`.

---

## 5. New HTTP endpoint

**Route:** `POST /v1/agents/chat/completions`

**Why a new route instead of extending `/v1/chat/completions`:**
keeps the OpenAI-compatible schema untouched and lets us evolve the agent
semantics independently. The new route's body is a strict superset.

### 5.1 Request schema (Pydantic)

```python
class AgentChatCompletionRequest(ChatCompletionRequest):
    agent_id: str = Field(..., min_length=1, max_length=128)
    prefetch_top_k: int | None = Field(default=None, ge=0, le=64)
    # Optional: pass a separate cache_salt; defaults to f"agent::{agent_id}".
    agent_cache_salt: str | None = None
    # Optional: caller can opt out of registry updates for this call.
    record_in_registry: bool = True
```

### 5.2 Response

Identical schema to `ChatCompletionResponse`, with two extra fields under a
`prefetch_meta` key:

```jsonc
{
  // ...standard chat completion fields...
  "prefetch_meta": {
    "submitted": 17,          // phantom prefetches submitted
    "skipped_already_cached": 2,
    "apc_hit_tokens": 1856,   // tokens covered by APC on the real call
    "lmcache_extra_tokens": 0 // tokens served from LMCache beyond APC
  }
}
```

`apc_hit_tokens` and `lmcache_extra_tokens` come from existing engine metrics
(`num_lmcache_extra_cached_tokens` is already exposed by the connector at
`lmcache_mp_connector.py:914-957`).

### 5.3 Request lifecycle

```
1. Parse request, tokenize messages → prompt_token_ids.
2. cache_salt = req.agent_cache_salt or f"agent::{req.agent_id}"
3. k = req.prefetch_top_k if not None else registry.default_top_k
4. recent = registry.top_k(req.agent_id, k)         # synchronous, O(k)
5. For each desc in recent:
      submit_phantom_prefetch(desc.token_ids, desc.cache_salt)   # fire-and-forget
   (No await here. We deliberately do NOT block the real call on these.)
6. Submit real chat completion to engine with kv_transfer_params=
      {"cache_salt": cache_salt, "agent_id": req.agent_id}
7. Stream/wait for real response.
8. If req.record_in_registry:
      desc = build_descriptor(prompt_token_ids, cache_salt, ...)
      registry.record(req.agent_id, desc)
9. Return response + prefetch_meta.
```

Why fire-and-forget in step 5: blocking the real call until prefetches finish
adds latency on the cold path. The phantom requests race the real one through
the scheduler; if a phantom completes its load first the blocks land in APC
before the real request reaches prefill — that is the win. If the real call
beats them, no harm done (the connector's own async load still fires for
whatever LMCache has).

---

## 6. Engine-side changes (Option A: phantom request)

This section is the core of the change.

### 6.1 `LMCacheMPRequestTracker` flag

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`

Add:
```python
@dataclass
class LMCacheMPRequestTracker:
    ...
    prefetch_only: bool = False
```

Populate it in `_get_or_create_request_tracker` from
`request.kv_transfer_params.get("prefetch_only", False)`.

### 6.2 Skip "minus one" cap for phantom prefetches

**File:** `vllm/v1/core/kv_cache_manager.py:208`

Today:
```python
max_cache_hit_length = request.num_tokens - 1
```
This forces the last block to recompute for logits. For phantom prefetches we
don't need logits. Add:
```python
if getattr(request, "kv_transfer_params", None) and \
        request.kv_transfer_params.get("prefetch_only"):
    max_cache_hit_length = request.num_tokens
else:
    max_cache_hit_length = request.num_tokens - 1
```

### 6.3 Scheduler hook: finish phantom requests post-load, pre-forward

**File:** `vllm/v1/core/sched/scheduler.py`

In the step where the connector reports finished loads (the existing
`connector.get_finished()` consumer), branch on `prefetch_only`:

```python
if request.kv_transfer_params and \
        request.kv_transfer_params.get("prefetch_only"):
    # Make sure APC owns the loaded blocks so future real requests find them.
    self.kv_cache_manager.cache_blocks(
        request, num_computed_tokens=request.num_external_computed_tokens)
    # Finish without ever entering the forward batch.
    self._finish_request(request, finish_reason=FinishReason.STOP)
    continue
```

Important: this must run **before** the request is added to the forward batch
for this step. Look at the order in the scheduler's main `schedule()` to place
the branch correctly — likely after async loads complete and before any
prefill chunk is scheduled.

### 6.4 Worker forward path

No change expected. If Step 6.3 runs in the same step as the load completion,
the request never enters `scheduled_new_reqs` for a forward pass. If timing
forces it to enter the forward batch, add an early-out in
`gpu_model_runner.py` keyed off `prefetch_only` (skip attention call, sample 0
tokens, return).

### 6.5 Request build path

**File:** `vllm/entrypoints/openai/protocol.py` or wherever
`SamplingParams`/`Request` are built from the HTTP body.

The new agent endpoint constructs phantom requests with:
- `prompt_token_ids = desc.token_ids`
- `sampling_params = SamplingParams(max_tokens=0, temperature=0)`
- `kv_transfer_params = {"prefetch_only": True, "cache_salt": desc.cache_salt}`
- Unique `request_id`, e.g. `f"prefetch::{agent_id}::{hex(prefix_hash)[:16]}"`

If `max_tokens=0` is not currently accepted by the engine, special-case it for
the prefetch path (or use `max_tokens=1` and rely on Step 6.3 to finish before
sampling — preferred to avoid touching sampling validation).

### 6.6 Connector retrieve path

No changes needed. Existing `start_load_kv` (`lmcache_mp_connector.py:558`)
and `worker_adapter.batched_submit_retrieve_requests` handle it.

### 6.7 Eviction / pinning (optional, v2)

If your workload runs many other requests between prefetch and the real call,
add an optional pin:
- `kv_cache_manager.pin_blocks(block_hashes, ttl_ms)` and `unpin_blocks(...)`.
- In Step 6.3, pin the just-cached blocks with `ttl_ms=req.pin_ttl_ms` (carried
  on the phantom request).
- A background task unpins when TTL elapses.

Skip for v1 unless measurements show eviction churn.

---

## 7. New module layout

```
vllm/
├── entrypoints/
│   └── openai/
│       └── agent_api.py            # NEW. Route + Pydantic models.
└── distributed/
    └── kv_transfer/
        └── kv_connector/
            └── v1/
                └── lmcache_mp_connector.py   # +prefetch_only flag

vllm/v1/
├── core/
│   ├── kv_cache_manager.py         # max_cache_hit_length tweak
│   └── sched/
│       └── scheduler.py            # finish phantom requests early
└── agent_prefetch/                  # NEW package
    ├── __init__.py
    ├── registry.py                 # AgentPrefixRegistry + PrefixDescriptor
    ├── hashing.py                  # prefix_hash() consistent with LMCache
    └── submitter.py                # submit_phantom_prefetch(...)
```

### 7.1 `vllm/v1/agent_prefetch/hashing.py`

```python
def compute_prefix_hash(model_name: str,
                       cache_salt: str,
                       token_ids: Sequence[int],
                       chunk_size: int) -> bytes:
    """
    Hash chain matching LMCache's chunk-key derivation. token_ids MUST be
    truncated to a multiple of chunk_size by the caller.
    """
```

Implementation must mirror `lmcache.utils.TokenHasher.hash_to_bytes` (or wrap
it directly via the public LMCache python import to avoid drift).

### 7.2 `vllm/v1/agent_prefetch/submitter.py`

```python
class PhantomPrefetchSubmitter:
    """
    Thin wrapper that builds and submits prefetch_only Requests into the
    engine's input queue.  Owned by the API server, not the engine.
    """

    def __init__(self, engine_client: EngineClient): ...

    async def submit(self,
                     token_ids: Sequence[int],
                     cache_salt: str,
                     agent_id: str) -> None:
        request_id = f"prefetch::{agent_id}::{hashlib.sha1(...).hexdigest()[:16]}"
        # Dedup: skip if an in-flight prefetch for this request_id exists.
        if self._inflight.add(request_id):
            return
        await self.engine_client.submit_request(
            request_id=request_id,
            prompt={"prompt_token_ids": list(token_ids)},
            params=SamplingParams(max_tokens=0, temperature=0),
            kv_transfer_params={"prefetch_only": True, "cache_salt": cache_salt},
        )
        # Don't await completion. Fire-and-forget.
```

Dedup is important: a single agent often shares the same prefix across many
near-simultaneous calls; we don't want to submit 20 copies.

### 7.3 `vllm/entrypoints/openai/agent_api.py`

```python
@router.post("/v1/agents/chat/completions",
             response_model=AgentChatCompletionResponse)
async def agent_chat_completions(req: AgentChatCompletionRequest):
    salt = req.agent_cache_salt or f"agent::{req.agent_id}"
    k = req.prefetch_top_k if req.prefetch_top_k is not None \
        else REGISTRY.default_top_k

    # 1. Fire phantom prefetches (non-blocking).
    descriptors = REGISTRY.top_k(req.agent_id, k)
    for d in descriptors:
        asyncio.create_task(
            SUBMITTER.submit(d.token_ids, d.cache_salt, req.agent_id)
        )

    # 2. Submit the real request through the existing chat completion path,
    #    forwarding agent_id + salt as kv_transfer_params.
    real_resp = await chat_completion_pipeline(
        req.to_chat_completion_request(),
        extra_kv_transfer_params={"cache_salt": salt,
                                  "agent_id": req.agent_id},
    )

    # 3. Record this prompt's chunk-aligned prefix in the registry.
    if req.record_in_registry:
        prefix_tokens = chunk_align(prompt_token_ids,
                                    chunk_size=CHUNK_SIZE)
        if prefix_tokens:
            desc = PrefixDescriptor(
                token_ids=tuple(prefix_tokens),
                prefix_hash=compute_prefix_hash(MODEL, salt,
                                                prefix_tokens, CHUNK_SIZE),
                cache_salt=salt,
                last_used_ns=time.monotonic_ns(),
            )
            REGISTRY.record(req.agent_id, desc)

    # 4. Decorate response with prefetch_meta and return.
    return assemble_response(real_resp, prefetch_meta=...)
```

---

## 8. Configuration

New CLI / config flags (added to `EngineArgs` / serve args):

| Flag | Default | Purpose |
|---|---|---|
| `--agent-prefetch-enabled` | `False` | Master switch. When off, the new endpoint 404s. |
| `--agent-prefetch-default-top-k` | `20` | Default `prefetch_top_k`. |
| `--agent-prefetch-max-top-k` | `64` | Upper bound to cap abusive callers. |
| `--agent-prefetch-max-agents` | `10000` | Eviction ceiling. |
| `--agent-prefetch-max-per-agent` | `64` | Per-agent LRU depth. |
| `--agent-prefetch-pin-ttl-ms` | `0` (off) | If >0, pin loaded blocks for N ms. v2. |

The new endpoint is gated by `--agent-prefetch-enabled` and the connector
being `LMCacheMPConnector` (assertion at startup).

---

## 9. Observability

Add Prometheus metrics:

- `vllm_agent_prefetch_submitted_total{agent_id}` (counter)
- `vllm_agent_prefetch_dedup_total{agent_id}` (counter)
- `vllm_agent_prefetch_registry_size{agent_id}` (gauge)
- `vllm_agent_prefetch_registry_agents` (gauge)
- `vllm_agent_prefetch_apc_hit_ratio` (histogram, real call only)
- `vllm_agent_prefetch_lmcache_hit_ratio` (histogram, real call only)

Add a debug log line per real call summarizing
`(submitted, dedup, apc_hit_tokens, lmcache_extra_tokens)`.

---

## 10. Failure modes & their handling

| Scenario | Behavior |
|---|---|
| LMCache server down | Phantom requests fail in connector heartbeat → no APC populated → real call falls back to normal prefill. Logged, but call still succeeds. |
| Agent unknown / empty registry | Step 5 is a no-op. Real call runs normally. |
| `prefetch_top_k = 0` | Disables prefetch for this call; registry update still happens. |
| Prefix evicted from LMCache between record and prefetch | Phantom request gets `num_new_matched_tokens = 0`, finishes immediately as a no-op. |
| `cache_salt` mismatch (registry has old salt, request has new) | Phantom retrieves miss in LMCache; no APC pop. No correctness issue. |
| Phantom blocks evicted from APC before real call arrives | Real call falls back to connector's normal load path (CPU→GPU on the fly). |
| Two concurrent calls from same agent | Dedup in submitter prevents duplicate phantom submissions. Registry record is RLock-guarded. |

---

## 11. Testing strategy

### 11.1 Unit
- `AgentPrefixRegistry`: LRU eviction within an agent, across-agent eviction,
  thread-safety under concurrent `record/top_k`.
- `compute_prefix_hash` parity with LMCache's chunk-key derivation.
- `PhantomPrefetchSubmitter.submit` dedup.

### 11.2 Integration (with running LMCache server)
- End-to-end: agent calls twice with same prefix; second call shows
  `apc_hit_tokens >= len(prefix) - block_size` in `prefetch_meta`.
- Eviction: fill GPU APC; verify second call falls back to LMCache load.
- Failure injection: stop `lmcache server` mid-test; new agent calls should
  still succeed (no prefetch, normal prefill).

### 11.3 Benchmark
- Compare TTFT and TPOT for agentic workload (1000 calls, 90% prefix reuse):
  - Baseline: `/v1/chat/completions` with LMCache connector.
  - Warmup-request approach (dummy `max_tokens=1`).
  - New `/v1/agents/chat/completions` with `prefetch_top_k=20`.
- Expect new endpoint to match or beat the warmup-request approach on TTFT
  while saving compute equal to ~1 forward pass per warmed prefix.

---

## 12. Rollout phases

**Phase 1 — Spike (1–2 days):**
Hand-build a phantom request with `prefetch_only=True` and the
`max_cache_hit_length` tweak. Submit one manually via a curl test endpoint.
Verify APC hit on the next real call via logs/metrics.

**Phase 2 — Engine plumbing (3–5 days):**
Land §6 changes (tracker flag, scheduler branch, kv_cache_manager tweak).
Behind a feature flag, no public endpoint yet.

**Phase 3 — Registry + submitter (2–3 days):**
Land §7.1, §7.2, §7.3. Plus unit tests in §11.1.

**Phase 4 — Public API (2 days):**
Land the `/v1/agents/chat/completions` route with full request/response
schema. Integration tests in §11.2.

**Phase 5 — Observability + docs (1–2 days):**
Metrics from §9. Docs page describing the endpoint and the registry's
semantics.

**Phase 6 — Benchmark + tune (2–3 days):**
§11.3. Tune default `top_k`, decide whether v2 (pinning) is needed.

Total estimate: ~2–3 engineer-weeks for a single owner.

---

## 13. Open questions for you

1. **Tokenizer location.** Should the registry store token_ids (what I've
   assumed) or raw text? Token ids are stable to chat template changes only if
   you control the template. If you ever swap templates, token-id-based prefixes
   silently miss. Confirm tokenizer is frozen for the agent's lifetime.

2. **Cross-restart persistence.** Should the registry survive a vLLM restart?
   If yes, easiest implementation is writing it to LMCache itself as a
   reserved-prefix key. Out of scope for v1 unless you say otherwise.

3. **Per-agent quota.** Should `agent_id` carry an LMCache `quota` (see
   `lmcache/v1/multiprocess/http_apis/quota_api.py`)? If so, the new endpoint
   should propagate a quota header to the MP server on first use of each
   agent. Optional v2.

4. **What counts as a "prefix"?** Currently I record one chunk-aligned prefix
   per call (the full prompt rounded down). Should we also record sub-prefixes
   so an agent that reuses *partial* overlaps benefits? Adds bookkeeping cost;
   skip for v1.

5. **Streaming.** Should `/v1/agents/chat/completions` support SSE streaming
   like the OpenAI endpoint? Yes — reuse the existing streaming path; the
   `prefetch_meta` goes in the final chunk.

Please flag any of the above that diverge from your intent before
implementation starts.
