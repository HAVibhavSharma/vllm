# Agent-scoped Prefix Prefetch APIs

Two HTTP endpoints expose the per-agent prefix-cache flow added to
vLLM in this branch:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/agents/chat/completions` | Chat completion that **records** the prompt's prefix in a per-agent registry. **Does not warm APC.** |
| `POST` | `/v1/agents/prefetch` | Explicit cache-warming. Fans out phantom prefetches for the agent's registered prefixes so APC is warm on return. |

Two auxiliary endpoints are also available for benchmarking /
diagnostics:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/agents/reset_prefix_cache` | Reset GPU APC (optionally registry / LMCache). |
| `GET` | `/v1/agents/registry_stats` | Inspect per-agent registry sizes. |

Source: `vllm/entrypoints/openai/agent_chat/api_router.py`.
Request schemas: `vllm/entrypoints/openai/agent_chat/protocol.py`.

---

## 1. The mental model

The two main endpoints are designed to be used **together**, in a
specific order:

1. The first time an agent issues a chat, you call
   `/v1/agents/chat/completions`. The server runs the chat normally
   **and** records the chunk-aligned prefix of the prompt in an
   in-process registry keyed by `agent_id`. The recording step is
   best-effort and never blocks the response.

2. Before any subsequent chat for the same agent, you call
   `/v1/agents/prefetch` with the agent's id. The server looks up
   every prefix the registry has stored for this agent and fans out
   *phantom* prefill requests (each `max_tokens=1`) that drive the
   relevant blocks from LMCache CPU L1 into GPU APC. With `wait=true`
   the endpoint blocks until every phantom finishes, so APC is
   guaranteed warm on return.

3. You issue the real chat via `/v1/agents/chat/completions` again.
   APC hits on the prefix → low TTFT. The new prompt's prefix is
   recorded as well, so the next prefetch warms a superset.

That's the loop. The chat endpoint **never** triggers prefetches on
its own; the prefetch endpoint **never** consumes prompt text. The two
responsibilities are deliberately split.

```
┌────────────┐    record prefix     ┌───────────────────────┐
│   chat     ├──────────────────────►│  per-agent registry  │
│  endpoint  │                       │   (in-process)       │
└────────────┘                       └───────────┬───────────┘
                                                 │ top-k or all
                                                 ▼
                                     ┌───────────────────────┐
                                     │  prefetch endpoint    │
                                     │  fans out phantoms    │
                                     └───────────┬───────────┘
                                                 │ phantoms drive
                                                 ▼
                                     ┌───────────────────────┐
                                     │  LMCache CPU L1 → APC │
                                     └───────────────────────┘
```

---

## 2. `POST /v1/agents/chat/completions`

Drop-in superset of OpenAI's `/v1/chat/completions` with three extra
fields. Streams SSE the same way the upstream endpoint does.

### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| All `/v1/chat/completions` fields | — | varies | — | `model`, `messages`, `max_tokens`, `temperature`, `stream`, etc. |
| `agent_id` | string (1–128 chars) | **yes** | — | Scopes the prefix registry. Distinct agents do not share prefixes. |
| `agent_cache_salt` | string (≤256) or null | no | `"agent::<agent_id>"` | Override for the LMCache `cache_salt`. Two agents with the same explicit salt will share LMCache entries. |
| `record_in_registry` | bool | no | `true` | When `false`, the server skips tokenizing-for-registry and does **not** record. Use for one-shot calls you do not expect to repeat. |

### Behaviour

1. If `record_in_registry=true`, the server applies the chat template
   and tokenizes the rendered prompt so it has the exact
   `prompt_token_ids` the engine will consume.
2. Delegates to the standard chat-completion pipeline.
3. Chunk-aligns the prompt tokens (default chunk size is the engine's
   page size — see `DEFAULT_CHUNK_SIZE`), computes a prefix hash, and
   stores `PrefixDescriptor(token_ids, prefix_hash, cache_salt,
   last_used_ns)` under `agent_id` in the registry.
4. Returns the completion (streaming or non-streaming, matching the
   `stream` field in the body).

Recording failures are caught and logged; they never break the
response.

### Response

Identical to `/v1/chat/completions`:
- `stream: true` → `text/event-stream` (SSE), one `data: {...}` chunk
  per token batch, terminated by `data: [DONE]`.
- `stream: false` → standard JSON `ChatCompletionResponse`.

### Errors

| Status | Cause |
|---|---|
| 400 | Tokenization / chat-template rendering failed. |
| 404 | `model` does not match the served model id. |
| 501 | Server was not started with a chat-capable model. |
| 5xx | Engine-side errors propagate as in vanilla `/v1/chat/completions`. |

### Example — curl, non-streaming

```bash
curl -s -X POST http://localhost:8000/v1/agents/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "agent_id": "agent1",
    "messages": [
      {"role": "system", "content": "You are agent1 ..."},
      {"role": "user",   "content": "Summarise the latest report."}
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": false
  }' | python3 -m json.tool
```

### Example — Python (`requests`), streaming

```python
import json, requests

resp = requests.post(
    "http://localhost:8000/v1/agents/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "agent_id": "agent1",
        "messages": [
            {"role": "system", "content": LONG_PREAMBLE},
            {"role": "user",   "content": "What changed today?"},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    },
    stream=True,
    timeout=120,
)
for line in resp.iter_lines():
    if not line or not line.startswith(b"data: "):
        continue
    payload = line[6:]
    if payload == b"[DONE]":
        break
    chunk = json.loads(payload)
    for ch in chunk.get("choices", []):
        delta = ch.get("delta", {}).get("content", "")
        if delta:
            print(delta, end="", flush=True)
```

### Example — OpenAI Python SDK

The OpenAI client doesn't know about `agent_id` natively, but you can
pass it through `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct-AWQ",
    messages=[{"role": "system", "content": LONG_PREAMBLE},
              {"role": "user",   "content": "go"}],
    max_tokens=64, stream=True,
    extra_body={"agent_id": "agent1"},   # ← agent routing
    # also valid: extra_body={"agent_id": "agent1", "record_in_registry": False}
)
```

Note: you must hit the agent path explicitly. The OpenAI client posts
to `/v1/chat/completions` by default, which is **not** the agent
endpoint. If you want this to "just work" without a custom client, set
`client.base_url = "http://localhost:8000/v1/agents"` so it posts to
`/v1/agents/chat/completions`.

---

## 3. `POST /v1/agents/prefetch`

Warms APC for a single agent by replaying the agent's registered
prefixes through the engine as zero-output phantom requests.

### Request body

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `agent_id` | string (1–128) | **yes** | — | Selects which agent's prefixes to warm. |
| `prefetch_top_k` | int ≥ 0 or null | no | `null` (= all) | Cap on how many of the agent's most-recently-used prefixes to warm. `null` warms **every** prefix the registry has for this agent. `0` is a no-op. |
| `agent_cache_salt` | string (≤256) or null | no | `"agent::<agent_id>"` | Should match whatever salt was used at chat time, or LMCache will miss. |
| `wait` | bool | no | `true` | `true` → block until every phantom finishes (APC is warm on return). `false` → fire-and-forget; server returns as soon as phantoms are queued. |

### Behaviour

1. Looks up descriptors for `agent_id` in the registry (sorted MRU).
2. Truncates to `prefetch_top_k` if provided.
3. For each descriptor, calls `PhantomPrefetchSubmitter.submit(...)`.
   The submitter de-duplicates: if a phantom for the same
   `(agent_id, prefix_hash)` is already in flight, the new submission
   is dropped silently.
4. With `wait=true`, `await asyncio.gather(*tasks)` blocks until all
   phantoms finish. With `wait=false`, the endpoint returns
   immediately and the phantoms keep running in the background.

### Response

```json
{
  "agent_id": "agent1",
  "requested_top_k": "all",
  "available_prefixes": 4,
  "submitted": 4,
  "completed": 4,
  "waited": true,
  "duration_ms": 142.5
}
```

| Field | Meaning |
|---|---|
| `requested_top_k` | `"all"` if `prefetch_top_k` was null/omitted; otherwise the integer cap echoed back. |
| `available_prefixes` | Total prefixes currently registered for this agent. |
| `submitted` | Number of phantoms actually handed to the engine after dedup. |
| `completed` | Number that ran to completion. Equals `submitted` when `wait=true`; `0` when `wait=false`. |
| `waited` | Echoes the `wait` flag. |
| `duration_ms` | Wall-clock cost of the whole call. |

### Errors

| Status | Cause |
|---|---|
| 400 | Schema validation (e.g. negative `prefetch_top_k`). |
| 501 | Server has no chat handler. |

If `available_prefixes == 0` the endpoint still returns 200 with
`submitted=0, completed=0`. Empty registry is not an error — it just
means no chat has been recorded for this agent yet.

### Example — curl, wait for warm

```bash
curl -s -X POST http://localhost:8000/v1/agents/prefetch \
  -H 'Content-Type: application/json' \
  -d '{"agent_id": "agent1", "wait": true}' | python3 -m json.tool
```

### Example — curl, top-k cap, fire-and-forget

```bash
curl -s -X POST http://localhost:8000/v1/agents/prefetch \
  -H 'Content-Type: application/json' \
  -d '{"agent_id": "agent1", "prefetch_top_k": 5, "wait": false}' \
  | python3 -m json.tool
```

### Example — Python helper for a chat-prefetch cycle

```python
import requests

BASE = "http://localhost:8000"

def warm(agent_id: str, top_k: int | None = None, wait: bool = True):
    body = {"agent_id": agent_id, "wait": wait}
    if top_k is not None:
        body["prefetch_top_k"] = top_k
    r = requests.post(f"{BASE}/v1/agents/prefetch", json=body, timeout=120)
    r.raise_for_status()
    return r.json()

def chat(agent_id: str, system: str, user: str, max_tokens: int = 64):
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "agent_id": agent_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }
    r = requests.post(f"{BASE}/v1/agents/chat/completions",
                      json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

# Typical loop: warm, then chat.
warm("agent1")                            # APC is warm for this agent
print(chat("agent1", SYSTEM, "go"))       # TTFT should be low; prefix recorded
```

---

## 4. Recommended client patterns

### Pattern A — request-scoped warm (one chat per agent turn)

Right before each chat, warm the prefix. Simple and predictable.

```
prefetch(agent_id, wait=true)
chat(agent_id, ...)
```

Use when the agent is bursty and chats are infrequent — you don't want
APC entries getting evicted between turns by other traffic.

### Pattern B — speculative warm (parallel pipelines)

If you know multiple agents will fire soon (e.g. agent2 follows
agent1), kick off agent2's warm `wait=false` while agent1 is still
streaming, so it overlaps with generation.

```
prefetch(agent1, wait=true)
chat(agent1, ...)               # streaming starts
prefetch(agent2, wait=false)    # overlaps with agent1 streaming
chat(agent2, ...)
```

### Pattern C — batched warm, then drain

Before a known burst, warm every agent up front:

```
for agent in agents: prefetch(agent, wait=false)
# (optionally one prefetch with wait=true to act as a barrier)
for agent in agents: chat(agent, ...)
```

### When to set `record_in_registry=false`

When you're issuing a chat the agent will *never repeat the prefix
of* — e.g. ad-hoc admin queries, one-off prompts. Skipping the record
saves the tokenize-twice cost and keeps the registry tight. For 95%
of agent traffic, leave it on.

### Cache salt

`agent_cache_salt` is exposed for advanced sharing:
- Default (`"agent::<agent_id>"`) means each agent's prefixes are
  isolated in LMCache.
- Set the **same** explicit salt across multiple agents to make them
  share LMCache entries (e.g. several agents that genuinely share a
  static system preamble).
- Whatever salt you used at chat time **must** be the same one you
  pass at prefetch time, or LMCache misses on the warm.

---

## 5. Aux endpoints (diagnostics / benchmarking)

### `POST /v1/agents/reset_prefix_cache`

Query parameters (all optional):

| Param | Default | Effect |
|---|---|---|
| `reset_apc` | `true` | Reset vLLM's GPU paged-KV prefix cache. Requires no in-flight requests (otherwise silent no-op with warning log). |
| `reset_registry` | `false` | Drop the in-process agent prefix registry. Phantoms will then need to wait for chats to re-populate it. |
| `reset_connector` | `false` | Also clear LMCache CPU L1 via the KV connector. Throws away CPU-resident prefix data. |

Response includes before/after registry stats and which actions ran.

```bash
# Cold APC, keep registry + LMCache (typical bench reset):
curl -X POST http://localhost:8000/v1/agents/reset_prefix_cache

# Full nuke:
curl -X POST "http://localhost:8000/v1/agents/reset_prefix_cache?reset_apc=true&reset_registry=true&reset_connector=true"
```

### `GET /v1/agents/registry_stats`

Returns initialization state plus per-agent prefix counts. Useful to
verify "yes, my chat call recorded something" before benchmarking the
prefetch path.

```bash
curl -s http://localhost:8000/v1/agents/registry_stats | python3 -m json.tool
```

---

## 6. Common gotchas

1. **`model` mismatch → 404.** The `model` field in the chat body must
   exactly match the model id vLLM is serving. The prefetch endpoint
   does *not* validate `model`, but the chat endpoint does.

2. **No prefixes available on first call.** Right after server boot
   (or right after `reset_registry=true`), the registry is empty.
   Calling `/v1/agents/prefetch` returns `submitted: 0`. That's not an
   error — you need to issue one chat per `(agent_id, prefix)` to seed
   the registry.

3. **Salt drift.** If you pass `agent_cache_salt` only at prefetch
   time and not at chat time (or vice versa), the LMCache key won't
   match and the warm will miss. Either pass it in both places, or in
   neither.

4. **`record_in_registry=false` plus prefetch.** A chat that didn't
   record won't show up in the next prefetch. The registry is
   write-only via chats; prefetch only reads.

5. **In-flight chats block APC reset.** `reset_prefix_cache` is a
   silent no-op if any request is still holding blocks. The HTTP
   response is still 200; check the log for `"Failed to reset prefix
   cache because some blocks (N) are not freed yet"`.

6. **The registry is per-process, in-memory.** Restarting the server
   wipes it; multi-replica deployments don't share state. Plan around
   that if you scale out.
