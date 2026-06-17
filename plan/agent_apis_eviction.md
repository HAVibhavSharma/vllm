# Agent APIs — Eviction-Policy Update

This doc is a diff against [`agent_apis.md`](agent_apis.md) — the API
shape that existed when this experimental branch diverged. Read that
first; this file only describes **what changed** to surface the
agent-aware eviction policy through the HTTP layer, and how the new
field connects to the policy described in
[`agent_eviction_policy.md`](agent_eviction_policy.md).

Source:

- Router — `vllm/entrypoints/openai/agent_chat/api_router.py`
- Schemas — `vllm/entrypoints/openai/agent_chat/protocol.py`
- Policy — `vllm/v1/agent_prefetch/eviction.py`

---

## 1. What changed at a glance

| Change | Endpoint | Direction |
|---|---|---|
| One new optional field (`agent_probabilities`) on the chat body | `POST /v1/agents/chat/completions` | Inbound — flows into `kv_transfer_params` and reaches the engine `Request`. |
| Same field added to the prefetch body | `POST /v1/agents/prefetch` | Inbound — every phantom this call submits inherits it so its blocks are tagged and its vote is counted while the phantom is in flight. |
| New diagnostic endpoint | `GET /v1/agents/eviction_stats` | Outbound — counts of active votes, tagged blocks, tracked agents. |

Nothing in the baseline contract was removed. The chat and prefetch
endpoints still accept everything they did before; `agent_probabilities`
is optional with a server-side default of "abstain."

### A note on the history

Earlier drafts of this API also exposed `eviction_window`,
`eviction_threshold`, and `probability_ttl_seconds`. They have all been
removed — none of them actually shaped policy behaviour in the final
implementation. Window and TTL were never wired through, and the
threshold gate collapsed pathologically in round-robin workloads (see
[`agent_eviction_policy.md`](agent_eviction_policy.md#what-gets-passed-in)
for the full story). The only knob that survives is the forecast itself.

---

## 2. The one eviction field

`agent_probabilities` is the only policy-related field on either
endpoint. It appears on **both**
`POST /v1/agents/chat/completions` and `POST /v1/agents/prefetch`.

| Field | Type | Default | What it controls |
|---|---|---|---|
| `agent_probabilities` | `dict[str, float]` or null | `null` | Caller's forecast: for each agent, the probability it fires soon. The policy aggregates the **max** across all live requests' votes. A `null` (or omitted) value means "this request abstains" — it neither protects nor exposes any agent. The owning `agent_id` is *always* self-protected with an implicit 1.0 vote regardless of what's in the map. |
| `agent_id` | string 1–128 | — (chat: required; prefetch: required) | Already existed pre-divergence; restated here because it doubles as the owner tag the policy writes onto every cached block this request produces. |

### Self-protection is implicit

You never need to put `{your_agent_id: 1.0}` in `agent_probabilities`
yourself. The policy adds a 1.0 self-vote for the request's own
`agent_id` automatically and excludes all currently-active agents
from the eviction ranking. Forecasts only need to mention the **other**
agents you have information about.

### What the engine sees

The chat-side `to_chat_completion_request()` (in
`vllm/entrypoints/openai/agent_chat/protocol.py`) shoves `agent_id` and
`agent_probabilities` into the request's `kv_transfer_params` dict. The
scheduler reads them on the other side of the API/engine boundary and
calls `register_request` on the policy. So the API is the only place
this knob is exposed — everything downstream is internal plumbing.

### Vote lifetime

A vote lives exactly as long as the request that cast it. When the
scheduler marks the request finished, its row in the aggregator is
dropped. Block-ownership tags survive — that is how cached work from
a finished turn stays evictable (or reusable) on later turns.

---

## 3. `POST /v1/agents/chat/completions` (delta)

### Added request fields

Optional; baseline behaviour is unchanged when omitted.

```jsonc
{
  // ... all original fields ...
  "agent_id": "agent_e",
  "agent_probabilities": {
    "agent_f": 1.00,
    "agent_g": 0.83,
    "agent_a": 0.67,
    "agent_b": 0.50,
    "agent_c": 0.33,
    "agent_d": 0.17
  }
}
```

### Behaviour changes

- The recording / delegation pipeline is untouched. Prefix recording
  is still best-effort and never blocks the response.
- `agent_id` + `agent_probabilities` ride along in `kv_transfer_params`
  so the scheduler can register the vote when it admits the request.
- The vote dies with the request: as soon as the scheduler marks the
  request finished, `unregister_request` removes its row from the
  aggregator. Block tags survive.

### Example — chat with a forecast

```bash
curl -s -X POST http://localhost:8000/v1/agents/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "agent_id": "agent_e",
    "messages": [
      {"role": "system", "content": "You are agent_e ..."},
      {"role": "user",   "content": "Summarise the latest report."}
    ],
    "max_tokens": 64,
    "stream": false,
    "agent_probabilities": {
      "agent_f": 1.00, "agent_g": 0.83, "agent_a": 0.67,
      "agent_b": 0.50, "agent_c": 0.33, "agent_d": 0.17
    }
  }'
```

The forecast above is the round-robin example walked through in
[`agent_eviction_policy.md`](agent_eviction_policy.md#a-short-worked-example).
While this request runs, `agent_d`'s blocks sit at the head of the
eviction queue and `agent_f`'s sit at the tail.

---

## 4. `POST /v1/agents/prefetch` (delta)

### Added request field

Same field, same semantics. It applies to **every phantom** this
prefetch call submits.

```jsonc
{
  "agent_id": "agent_e",
  "prefetch_top_k": null,
  "wait": true,
  "agent_probabilities": { "agent_f": 1.00, "agent_g": 0.83, ... }
}
```

### Behaviour changes

- Each phantom registers its own vote with the policy when it starts
  and unregisters when it finishes — phantoms are normal requests
  from the policy's point of view.
- If `agent_probabilities` is omitted, every phantom falls back to a
  bare `{agent_id: 1.0}` self-vote. That is the minimum the policy
  needs to (a) flip the "opted out" gate so the phantom's freshly
  loaded blocks get tagged with `agent_id`, and (b) keep that agent
  protected while the phantom is in flight.
- All other prefetch semantics from the baseline (`prefetch_top_k`,
  `wait`, dedup against in-flight phantoms with the same
  `(agent_id, prefix_hash)`, the response envelope) are unchanged.

### When prefetch should mirror chat's forecast

If you call prefetch right before chat with the same `agent_id`, pass
the same `agent_probabilities` to both. Otherwise the phantom and the
chat will be voting on different futures, and the policy will take the
**max** across both — which is usually fine but can mask a forecast
update if the phantom's vote is more optimistic than the chat's new
one. Phantom votes end when the phantom finishes, so any divergence
is bounded by phantom lifetime.

---

## 5. New endpoint: `GET /v1/agents/eviction_stats`

Returns a snapshot of the eviction-policy singleton. Useful for
verifying that requests are arriving with `agent_probabilities`
attached and that blocks are being tagged.

### Response

```json
{
  "active_requests": 3,
  "tagged_blocks": 18421,
  "tracked_agents": 7
}
```

| Field | Meaning |
|---|---|
| `active_requests` | Number of live requests currently feeding votes into the aggregator. Includes phantom prefetches. |
| `tagged_blocks` | Number of cached GPU blocks the policy currently owns a tag for. Grows as caches fill, shrinks on `untag_block` / `clear_all_blocks`. |
| `tracked_agents` | Number of distinct agents with at least one tagged block. |

### Example

```bash
curl -s http://localhost:8000/v1/agents/eviction_stats | python3 -m json.tool
```

This endpoint is GET-only and side-effect free.

---

## 6. Unchanged endpoints

These are the same as in [`agent_apis.md`](agent_apis.md):

- `POST /v1/agents/reset_prefix_cache` — same query params, same
  semantics. Note that `clear_all_blocks` inside the policy is
  driven by APC reset so a full `reset_apc=true` call also wipes every
  block tag. Live votes from in-flight requests stay until those
  requests finish.
- `GET /v1/agents/registry_stats` — unchanged.

The pre-existing mental model (chat records → prefetch warms → chat
hits) still applies; eviction sits underneath it and changes **which
blocks survive** between turns, not who calls whom.

---

## 7. Common gotchas (eviction-specific)

1. **Forgetting `agent_probabilities` on prefetch.** A prefetch with
   no map still works — the phantom self-votes — but it provides no
   information about other agents. If you have a forecast, pass it
   to both chat and prefetch.

2. **Voting from outside the active set.** A request whose `agent_id`
   is **not** mentioned in its own `agent_probabilities` still gets
   the implicit 1.0 self-vote. Putting your own id in the map is
   harmless (max of 1.0 and your value); leaving it out is the
   normal case.

3. **Tags survive votes.** When a request (or phantom) finishes, its
   vote disappears but the blocks it produced remain tagged. They
   stay candidates for later eviction — at whatever ranking position
   their agent's aggregated probability gives them — until either the
   policy evicts them or APC is reset. This is intentional: it is how
   cache from a finished turn carries forward into the next turn for
   the same agent.

4. **The policy singleton is per-process.** Same caveat as the
   per-process registry. Multi-replica deployments do not share the
   policy state.
