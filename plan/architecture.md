# Agent-Scoped KV Prefix Prefetch — Implemented Architecture

This document describes what was built. For the original design discussion,
see `agent_prefetch_plan.md`; for runnable commands, see `commands.md`.

---

## 1. Goal

Add a new HTTP endpoint, `POST /v1/agents/chat/completions`, that
proactively warms vLLM's GPU prefix cache (APC) from LMCache CPU L1
**before the real chat completion's prefill stage runs**. The endpoint
takes the same body as `/v1/chat/completions` plus an `agent_id` and a
`prefetch_top_k`, and uses a per-agent registry of recent prefixes to
decide what to warm.

In the demo workflow (3 agents × multiple persona variants × long
preambles), this turns the second-onward call of each agent from a
~950 ms cold prefill into a ~100-150 ms APC hit, despite the GPU
prefix cache having been explicitly invalidated.

---

## 2. File map

Files added or modified, grouped by role.

### 2.1 New package — agent prefetch primitives

```
vllm/v1/agent_prefetch/
├── __init__.py          # public surface
├── hashing.py           # compute_prefix_hash, chunk_align
├── registry.py          # PrefixDescriptor, AgentPrefixRegistry (per-agent LRU)
└── submitter.py         # PhantomPrefetchSubmitter (fire-and-forget engine submission)
```

### 2.2 New HTTP routes

```
vllm/entrypoints/openai/agent_chat/
├── __init__.py
├── protocol.py          # AgentChatCompletionRequest (superset of ChatCompletionRequest)
└── api_router.py        # 3 endpoints + lazy app.state init
```

Registered from:

```
vllm/entrypoints/openai/generate/api_router.py
  └── register_generate_api_routers(app)
        └── register_agent_chat_api_router(app)
```

### 2.3 Engine-side changes

```
vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py
  └── LMCacheMPRequestTracker.prefetch_only flag (read from request.kv_transfer_params)

vllm/v1/core/sched/scheduler.py
  ├── FinishReason import
  ├── _update_from_kv_xfer_finished — now takes outputs dict; branches on prefetch_only
  ├── _is_prefetch_only_request — static helper
  └── _finalize_prefetch_only_request — registers loaded blocks in APC,
                                         marks request finished, emits terminal output
```

### 2.4 Tests

```
tests/v1/agent_prefetch/
├── __init__.py
├── test_hashing.py      # determinism, salt isolation, chunk alignment
└── test_registry.py     # per-agent LRU, cross-agent eviction, thread safety smoke
```

### 2.5 Demo client

```
examples/online_serving/agent_prefetch_workflow.py
  ├── predetermined call schedule printer
  ├── 3 modes: baseline / warmup / prefetch
  ├── per-call inferred cache-hit tagging
  └── matplotlib plots (--plot): timeline + per-agent bars
```

---

## 3. Architecture overview

```mermaid
graph TB
    subgraph Client["Demo workflow client"]
        DEMO[agent_prefetch_workflow.py<br/>fixed-order schedule]
    end

    subgraph VLLM["vLLM serve process"]
        direction TB

        subgraph HTTP["FastAPI routes"]
            EP_AGENT["/v1/agents/chat/completions"]
            EP_RESET["/v1/agents/reset_prefix_cache"]
            EP_STATS["/v1/agents/registry_stats"]
            EP_CHAT["/v1/chat/completions<br/>(delegated to internally)"]
        end

        subgraph Pkg["agent_prefetch package"]
            REG[AgentPrefixRegistry<br/>per-agent LRU]
            SUB[PhantomPrefetchSubmitter<br/>asyncio tasks]
            HASH[compute_prefix_hash<br/>+ chunk_align]
        end

        subgraph Engine["vLLM engine"]
            SCHED[Scheduler<br/>+ _finalize_prefetch_only_request]
            APC[GPU paged-KV prefix tree]
            MR[Model runner]
        end

        CONN[LMCacheMPConnector]
    end

    subgraph LMC["lmcache server (:10903)"]
        SERVER[MP server]
        L1[CPU L1 pool, 20 GB, LRU]
    end

    GPU[(GPU HBM)]

    DEMO -->|messages,<br/>agent_id, prefetch_top_k| EP_AGENT
    DEMO -.->|between runs| EP_RESET
    DEMO -.->|diagnostic| EP_STATS

    EP_AGENT -->|top_k| REG
    EP_AGENT -->|fan out phantoms| SUB
    EP_AGENT -->|delegate real call| EP_CHAT
    EP_AGENT -->|record prefix| REG
    SUB -->|prefetch_only:true| SCHED
    EP_CHAT --> SCHED

    SCHED <--> CONN
    SCHED <--> APC
    SCHED <--> MR
    MR <--> GPU
    APC -.points to.-> GPU
    CONN <-->|ZMQ STORE / RETRIEVE| SERVER
    SERVER <--> L1
    SERVER <==>|CUDA memcpy<br/>via IPC handles| GPU

    EP_RESET -->|reset_prefix_cache| SCHED
    EP_STATS -->|stats| REG

    classDef new fill:#9f9,stroke:#333,stroke-width:2px
    class EP_AGENT,EP_RESET,EP_STATS,REG,SUB,HASH new
```

Green = new code added for the agent prefetch.

---

## 4. Per-call request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant EP as agent endpoint
    participant REG as Registry
    participant SUB as Submitter
    participant ENG as Engine + Connector
    participant LMC as lmcache server
    participant APC as GPU APC

    C->>EP: POST agent_id, prefetch_top_k, messages
    EP->>EP: render_chat_request → prompt_token_ids
    EP->>REG: top_k(agent_id)
    REG-->>EP: K descriptors

    par Fire and forget
        EP->>SUB: submit(desc_i) for each
        SUB-)ENG: generate(prompt_token_ids,<br/>kv_transfer_params={prefetch_only:true})
    and Real call
        EP->>ENG: chat_handler.create_chat_completion(inner)
    end

    Note over ENG,LMC: Phantoms and real call admitted concurrently

    ENG->>LMC: LMCache lookup for each
    LMC-->>ENG: hit / miss per request

    alt phantom hit (async-load)
        ENG->>ENG: allocate blocks,<br/>status=WAITING_FOR_REMOTE_KVS
        LMC->>APC: CPU→GPU memcpy into blocks
        LMC-->>ENG: finished_recving
        ENG->>APC: cache_blocks (register prefix)
        ENG->>ENG: _finalize_prefetch_only_request<br/>(emit STOP, free request)
    else phantom miss (cache_salt mismatch / evicted)
        ENG->>ENG: full prefill (~22K tokens)<br/>then sample 1 token
    end

    Note over ENG: For the real call:<br/>by the time get_num_new_matched_tokens<br/>runs, APC may already have the prefix<br/>(if phantom finished first)

    ENG-->>EP: stream chunks (real call)
    EP->>REG: record(agent_id, this_call's_prefix)
    EP-->>C: stream

    Note over C: TTFT ≈ 100-150 ms<br/>(vs. ~950 ms baseline cold)
```

---

## 5. Component reference

### 5.1 `vllm/v1/agent_prefetch/hashing.py`

Two helpers.

- **`compute_prefix_hash(model_name, cache_salt, token_ids) -> bytes`** —
  SHA-256 over length-prefixed (model_name, cache_salt, token_ids). Used
  only as a registry identity key. Does **not** need to match LMCache's
  internal chunk hashing — LMCache hashes server-side.
- **`chunk_align(token_ids, chunk_size=16) -> list[int]`** — floors the
  token sequence to a multiple of the LMCache chunk size. Returns `[]` if
  shorter than one chunk. Matches LMCache's "only full chunks store" rule.

### 5.2 `vllm/v1/agent_prefetch/registry.py`

- **`PrefixDescriptor`** — frozen dataclass with `token_ids`,
  `prefix_hash`, `cache_salt`, `last_used_ns`. Identity is `prefix_hash`.
- **`AgentPrefixRegistry`** — `OrderedDict[agent_id, OrderedDict[hash, desc]]`.
  - `record(agent_id, desc)` — O(1) insert/promote, evicts LRU within
    agent when `max_per_agent` exceeded, evicts LRU agent across all
    when `max_agents` exceeded.
  - `top_k(agent_id, k) -> list[desc]` — **side-effect-free** read,
    newest-first. (Important: fanning out phantoms must not perturb LRU
    order on the prefixes being read.)
  - Caps: `default_top_k=20`, `max_agents=10_000`, `max_per_agent=64`.
  - Thread-safe via `threading.RLock`.

### 5.3 `vllm/v1/agent_prefetch/submitter.py`

- **`PhantomPrefetchSubmitter`** — fire-and-forget submission to the
  engine via `engine_client.generate(...)`.
  - Per-agent in-flight dedup keyed on
    `request_id = f"prefetch::{agent_id}::{short_hex(prefix_hash)}"` so
    concurrent fan-outs for the same prefix collapse to one engine
    request.
  - Phantom request shape:
    - `prompt = {"prompt_token_ids": list[int]}`
    - `SamplingParams(max_tokens=1, temperature=0)`
    - `extra_args = {"kv_transfer_params": {"prefetch_only": True, "cache_salt": ...}}`
  - Cap: `max_inflight_per_agent=64`.
  - Errors are absorbed (best-effort).

### 5.4 `vllm/entrypoints/openai/agent_chat/protocol.py`

- **`AgentChatCompletionRequest`** — extends `ChatCompletionRequest` with
  `agent_id` (required), `prefetch_top_k` (optional int, default ←
  registry default), `agent_cache_salt` (optional), `record_in_registry`
  (default True).
- `to_chat_completion_request()` — strips the agent-only fields and
  validates back to a plain `ChatCompletionRequest` for delegation.

### 5.5 `vllm/entrypoints/openai/agent_chat/api_router.py`

Three routes, plus lazy `app.state` initialization for the registry and
submitter on first call.

| Route | Method | Purpose |
|---|---|---|
| `/v1/agents/chat/completions` | POST | Main endpoint. Fan out phantoms, delegate real call, record on completion. |
| `/v1/agents/reset_prefix_cache` | POST | Reset APC and/or registry and/or connector cache. Returns rich JSON with before/after state. |
| `/v1/agents/registry_stats` | GET | Inspect per-agent registry sizes. |

The agent endpoint flow:

1. `_tokenize_prompt` — uses `chat_handler.render_chat_request(inner)`
   to apply the chat template and get `prompt_token_ids`. Same path the
   real call will internally use.
2. `_fan_out_prefetches` — `top_k(agent_id, k)` → `submitter.submit(...)`
   for each descriptor. Awaited only to receive the per-task booleans;
   actual phantom completion is *not* awaited.
3. Delegate to `chat_handler.create_chat_completion(inner, raw_request)`.
4. `_record_in_registry` — chunk-align the just-tokenized prompt, hash,
   record under `agent_id`. Happens before the streaming response
   finishes, so subsequent calls see this prefix immediately.
5. Return the chat handler's `StreamingResponse | ChatCompletionResponse |
   ErrorResponse` unchanged.

### 5.6 `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`

- **`LMCacheMPRequestTracker.prefetch_only: bool`** — initialized from
  `request.kv_transfer_params["prefetch_only"]` in `__init__`. Currently
  observable for diagnostics; the scheduler reads
  `request.kv_transfer_params` directly to make routing decisions.
- **`request_finished` is unchanged in semantics** (returns
  `(True, params)` for all requests including prefetch_only). This is
  load-bearing: the cache-miss phantom path issues a STORE that the
  worker needs to finalize before blocks can be released. Returning False
  for prefetch_only here was tried during development and caused two
  classes of bugs — leaked blocks for the async-load phantom path and
  stale-data races for the cache-miss path. Both went away when this was
  reverted to the default.

### 5.7 `vllm/v1/core/sched/scheduler.py`

The injection point lives inside `_update_from_kv_xfer_finished`. When
the worker reports `finished_recving=[req_id, ...]` for a request whose
status is still `WAITING_FOR_REMOTE_KVS`, the scheduler branches on
`_is_prefetch_only_request`:

- Normal request → existing path (add to `finished_recving_kv_req_ids`
  for next-step promotion to `WAITING` and prefill of the loaded blocks).
- Phantom (`kv_transfer_params.prefetch_only == True`) →
  `_finalize_prefetch_only_request`, which:
  1. Calls `kv_cache_manager.cache_blocks(request, request.num_computed_tokens)`.
     This registers the just-loaded blocks in APC under the prompt's hash
     chain. **This is the act that makes the next real request with the
     same prefix an APC hit.**
  2. Removes the request from both waiting queues
     (`self.waiting`, `self.skipped_waiting`).
  3. Sets `status = FINISHED_STOPPED`.
  4. Calls `self._free_request(request)` — which goes through the
     connector's `request_finished` (returns `(True, params)`, delays
     block free) and adds the request id to `self.finished_req_ids`.
  5. Appends a terminal `EngineCoreOutput(finish_reason=STOP)` to
     `outputs[client_index]` so the submitter's `async for` loop sees the
     request finish and the asyncio task exits cleanly.

A second pass through `_update_from_kv_xfer_finished` (next forward
step) sees the same `req_id` in `kv_connector_output.finished_sending`
— the LMCache adapter synthesizes this whenever an engine-finished
request has no tracked store/retrieve future. The existing
`finished_sending` branch then calls `_free_blocks`, which releases the
blocks back to the free pool. APC keeps the entry — that's how it
survives the request's death.

---

## 6. Phantom request state machine

```mermaid
stateDiagram-v2
    [*] --> WAITING: submitter.submit()
    WAITING --> WAITING_FOR_REMOTE_KVS: scheduler admits<br/>load_kv_async=True<br/>(LMCache hit)
    WAITING --> RUNNING: scheduler admits<br/>load_kv_async=False<br/>(LMCache miss)

    WAITING_FOR_REMOTE_KVS --> FINISHED_STOPPED: connector finished_recving<br/>→ _finalize_prefetch_only_request<br/>cache_blocks → APC entry created

    RUNNING --> FINISHED_STOPPED: normal prefill + sample 1 token

    FINISHED_STOPPED --> [*]: next step's finished_sending<br/>→ _free_blocks
    note right of WAITING_FOR_REMOTE_KVS
        Happy path:
        CPU→GPU retrieve, APC populated,
        no prefill ran. Cost: 1 retrieve.
    end note

    note right of RUNNING
        Fallback path:
        Full prefill of phantom prompt,
        STOREs to LMCache,
        samples 1 token. Wasteful.
    end note
```

---

## 7. Operational endpoints

### 7.1 `POST /v1/agents/chat/completions`

```jsonc
// Request body — superset of OpenAI ChatCompletionRequest
{
  "model": "Qwen/Qwen3-8B",
  "messages": [...],
  "agent_id": "agent1",                // required
  "prefetch_top_k": 20,                // optional, default 20
  "agent_cache_salt": null,            // optional, default "agent::<agent_id>"
  "record_in_registry": true,          // optional, default true
  // ...all standard ChatCompletionRequest fields...
}
```

Response is exactly a `ChatCompletionResponse` (or SSE stream of
`ChatCompletionStreamResponse` chunks) — no schema breakage.

### 7.2 `POST /v1/agents/reset_prefix_cache`

Query parameters:

- `reset_apc` (default `true`) — clear vLLM's GPU prefix tree.
- `reset_registry` (default `false`) — clear the in-process agent registry.
- `reset_connector` (default `false`) — propagate the reset to the
  connector (e.g., LMCache CPU L1). Use with care.

Returns:

```json
{
  "requested": {"reset_apc": true, "reset_registry": false, "reset_connector": false},
  "actions": ["apc_reset"],
  "apc": {"ok": true, "engine_returned": true, "connector_reset": false},
  "registry": {
    "before": {"num_agents": 3, "total_descriptors": 9, ...},
    "after":  {"num_agents": 3, "total_descriptors": 9, ...}
  },
  "duration_ms": 12.4
}
```

Logs three INFO lines per call:
- `agent_prefetch: resetting vLLM APC (reset_connector=...)`
- `agent_prefetch: APC reset complete (engine returned True)`
- `agent_prefetch: reset done in 12.40ms actions=['apc_reset']`

### 7.3 `GET /v1/agents/registry_stats`

```json
{
  "initialized": true,
  "overall": {
    "num_agents": 3, "total_descriptors": 9,
    "max_agents": 10000, "max_per_agent": 64, "default_top_k": 20
  },
  "per_agent_sizes": {"agent1": 3, "agent2": 2, "agent3": 4}
}
```

`initialized: false` means no agent request has hit the server yet.

---

## 8. Where the KV data physically lives

After running the workflow once, each prefix exists in up to three tiers:

| Tier | Where | Cap | Survives... |
|---|---|---|---|
| vLLM APC | GPU paged-KV blocks (in the prefix tree) | bounded by GPU mem | vLLM process lifetime; LRU eviction |
| LMCache L1 | CPU RAM of the `lmcache server` process | `--l1-size-gb 20` | LMCache restart |
| LMCache L2 (if configured) | disk / remote | n/a | n/a |

Only full **LMCache chunks** (16 tokens each) are stored. The agent
registry stores only the chunk-aligned tokens; the partial tail of the
prompt is excluded.

---

## 9. Performance observations (Qwen3-8B, ~22K-token preambles)

Measurements after `reset_prefix_cache?reset_apc=true` was issued
between runs (so APC is cold but LMCache + registry are warm):

| Call shape | Baseline cold | With prefetch |
|---|---|---|
| First call of an agent (N phantoms + real, all 22K-token retrieves) | ~970 ms | 1800-2800 ms ⚠️ |
| Second-onward call of an agent (phantoms hit APC, real hits APC) | ~970 ms | ~100-150 ms ✓ |
| Median across 9-call workflow | **~957 ms** | **~142 ms** |

The first-call regression is contention between the real call and N
concurrent 22K-token CPU→GPU retrieves. Subsequent calls benefit because
prior calls' phantoms have already populated APC. Three mitigations are
identified but not yet implemented:

- **Skip self-phantom**: don't fire a phantom for the prefix the real
  call is about to ask for.
- **Lower `prefetch_top_k`** on cold paths: `--prefetch-top-k 1` largely
  eliminates contention at the cost of warming fewer future prefixes.
- **Per-request priority**: submit phantoms at lower scheduler priority
  than real calls.

---

## 10. Known limitations / open issues

1. **Cache salt isolation isn't yet end-to-end.** The agent endpoint
   constructs `cache_salt = f"agent::{agent_id}"` but does not propagate
   it to the inner `ChatCompletionRequest` (no top-level `cache_salt`
   field is set). The submitter also stuffs cache_salt only into
   `kv_transfer_params`, which the connector's tracker doesn't read.
   Result: in practice both phantoms and real calls use
   `tracker.cache_salt = ""`. They match each other (which is what
   makes the demo work), but agents are not actually isolated from each
   other at the LMCache level. Fixable by setting
   `inner.cache_salt = cache_salt` before delegation and adding a
   `cache_salt` key to the submitter's prompt dict.

2. **Registry is in-process, not persisted.** Restarting vLLM wipes it.
   This is the "gotcha" that caused the early test confusion. A
   follow-up could back the registry with LMCache itself (reserved key
   prefix) or local sqlite.

3. **Hybrid KV cache manager unsupported.** Inherits the existing
   `LMCacheMPConnector` constraint. The connector's
   `reformat_block_ids` helper raises if multiple KV cache groups exist.

4. **First-call contention regression** described in §9 — the prefetch
   path makes per-agent first calls *slower* than baseline.

5. **`request_finished` returns `(True, ...)` for prefetch_only too.**
   This is correct given the LMCache adapter's finished_sending
   synthesis behavior, but it means the phantom's blocks are held for
   one extra forward step after `_finalize_prefetch_only_request` runs.
   Negligible in practice; documented as load-bearing in §5.6.

---

## 11. Cross-reference

- `plan/agent_prefetch_plan.md` — original design with §13 open
  questions, most of which are still open.
- `plan/commands.md` — the operational runbook (start services, test
  steps, log triage).
- `vllm/v1/agent_prefetch/__init__.py` — the public Python surface.
- `examples/online_serving/agent_prefetch_workflow.py` — the test
  client used for everything in §9.
