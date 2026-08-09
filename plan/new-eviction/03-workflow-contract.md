# What the Engine Expects From the Workflow

**Status:** contract draft. Nothing implemented on either side.
**Written:** 2026-07-31, against branch `vllm-v2` @ `68c7acb14`.
**Audience:** whoever owns the LangGraph fork, the Prediction Engine and the
Trace Analyser. This is the only doc in the set aimed outward.
**Reads with:** [`00-design-checkpoint.md`](00-design-checkpoint.md),
[`01-redis-integration.md`](01-redis-integration.md) (transport and scoring
inputs), [`02-controller-scope.md`](02-controller-scope.md) (controller scope).
Where this doc and 01 §5/§8 disagree, **this doc is the interface** and 01 is the
rationale.

`[verified]` = read at the cited line, exists today. `[required]` = the engine
needs it and it does **not** exist yet. `[open]` = the workflow side must choose,
and the engine adapts.

---

## 0. Three surfaces

Two inbound, one outbound:

| Surface | Carries | Direction | Frequency |
|---|---|---|---|
| **A — the request** | *identity*: which job/node/call this generation belongs to | client → vLLM, in `sampling_params.extra_args` | every request |
| **B — Redis** | *forecast and history*: what will be needed next, and what a miss costs | Prediction Engine / Trace Analyser → Redis → vLLM | on change |
| **C — request records** | *timings*: what every LLM call actually cost | **vLLM → Trace Analyser**, JSONL on disk | every finished request |

Surface C closes the loop: **vLLM is the only thing that can measure LLM call
time, so the timings in `HISTORY` originate from the engine itself.** The Trace
Analyser aggregates them and publishes them back as a forecast input. It does
not instrument the workflow and does not time anything independently —
see [`04-trace-analyser.md`](04-trace-analyser.md).

This does not weaken the ingress rule: the engine still *reads* only Redis.
Surface C is an outbound file write on the front-end side, already implemented
as `FileStatLogger` (`vllm/v1/metrics/loggers.py:1272`) `[verified]`, and needs
three new fields (§1.1).

The engine does not call the Trace Analyser, does not accept probabilities on
the request, and does not write to Redis (01 §7, §6 below).

**A without B** = the engine tracks blocks per node but cannot rank them; it
behaves as LRU. **B without A** = the engine receives forecasts it cannot attach
to any block, and ignores them. Both surfaces are needed for any benefit.

---

## 1. Surface A — identity on every request

Set on the OpenAI-compatible chat request under `extra_args`:

```jsonc
{
  "model": "...",
  "messages": [...],
  "extra_args": {
    "job_id":         "run-8f21c3",      // required
    "langgraph_node": "research",        // required
    "call_type":      "tavily:summary",  // required  (NEW)
    "agent_id":       "langgraph:research"  // optional, stats only
  }
}
```

| Field | Status | What the engine does with it |
|---|---|---|
| `job_id` | `[verified]` read at `output_processor.py:248` | Redis keyspace key, and the first element of the block-index key. Scopes one workflow run. |
| `langgraph_node` | `[verified]` `output_processor.py:254` | **The join key with Redis `node_name`.** Second element of the index key. |
| `call_type` | **`[required]` — does not exist anywhere in the codebase today** | Third element of the index key; makes the join with `PROB`/`HISTORY` rows exact instead of aggregated (01 §5). |
| `agent_id` | `[verified]` `output_processor.py:251` | Per-request stats only. **Not** a join key — it is namespaced (`langgraph:research`) and would need string munging. |

These already cross the front-end → engine-core boundary; `scheduler.py:799-815`
reads `extra_args` engine-core side today. `[verified]` So no protocol change and
no new request field is needed — only that the workflow populates them.

### The three hard requirements

**1. `langgraph_node` must be byte-identical to Redis `node_name`.** The join is
a dict lookup, not a fuzzy match. `research` ≠ `Research` ≠ `langgraph:research`.
A mismatch is silent: the blocks are indexed, the forecast arrives, and they
never meet — the engine degrades to LRU with no error.

**2. `call_type` must be stable for what is physically the same prefix.** This
is the one open risk that can make the design *worse* than doing nothing (01 §5).
If the same cached prefix is labelled `summary` on one turn and `tavily:summary`
on the next, the index re-keys the same blocks every turn, each key holds a
fragment of the prefix, and no key ever accumulates enough history to be scored
usefully. **If the flattening is not deterministic, say so — the engine will key
at `(job_id, node)` and take `max` over call types instead.** That is a
one-line change on our side and an unbuildable design if we find out late.

**3. Every request in a job carries the same `job_id`,** including retries and
tool-triggered sub-calls. The index and the Redis rows are scoped by it.

### 1.1 Engine-side additions this requires `[required]`

Surface C carries the same identity fields back out, so `call_type` must be
threaded through the stats path, and two absolute timestamps must be added
because **`tool_execution_time` is a gap between requests and cannot be derived
from durations**. Specified in 04 §3; summarised here because it is engine work,
not workflow work:

| Change | Where |
|---|---|
| read `call_type` from `extra_args` | `output_processor.py:248-258`, beside the existing three |
| carry it through the stats record | `stats.py:222`, `:442`, `loggers.py:1249`, `:1317` |
| add `arrival_ts` / `finish_ts`, **wall clock** | same files — `arrival_time` exists at `stats.py:208` but is never emitted |

Clock hazard, worth stating once: `stats.py` mixes wall-clock and monotonic
timestamps (`:210`) `[verified]`. Only wall-clock values may be exported as
absolutes.

### Not expected on the request

`prob`, `time_to_next_call`, or any forecast. The previous attempt took
probabilities from the client per request and it failed for two structural
reasons: a request without them was invisible to the policy, and the votes died
when the request finished — leaving the policy blind in exactly the idle gaps
where it matters most (00 Part 2, weaknesses 1–2). The forecast now comes from
Redis, which outlives requests.

---

## 2. Surface B — the Redis keyspaces

Three keyspaces, all keyed by `job_id`. Field names below are what 01 §5
records; confirm or correct them.

### `INFO` — static, written once per job

| Field | Type | Engine use |
|---|---|---|
| `node_name` | string | Joins to `langgraph_node`. |
| `type` | `react` \| `non-react` | Structural prior. React nodes accumulate a growing prefix across turns; non-react do not, so their blocks are scored lower and dropped earlier (02 §9). |
| `list_call_types` | list[string] | The expected `call_type` values for this node. Lets the engine detect surface-A/surface-B label drift instead of silently mis-joining. |

Fetched once per job and cached; never re-read unless the job is new.

### `PROB` — dynamic, the forecast

| Field | Type | Engine use |
|---|---|---|
| `node_name` | string | join |
| `call_type` | string | join |
| `prob` | float | P(this `(node, call_type)` fires again **at all** this job). **Time-free** — must not encode "soon". The engine applies its own time discount to `time_to_next_call`; encoding timing here applies it twice (05 §5.2). |
| `time_to_next_call` | duration | *When*, given that it fires. Second importance term. It was also the TTL of the speculative protection floor; that decay is gone (12 §5.5) and the floor itself is now off by default (12 §6), so this field only feeds the time discount. |
| `update_ts` | timestamp | **Staleness gate.** Rows older than the cutoff are treated as unscored → neutral default → LRU behaviour. This is what makes a missed pub/sub message degrade instead of act on stale data. |

One row per `(node, call_type)`. A node with several call types has several rows,
each with its own `prob` and `time_to_next_call` — the engine does not want them
pre-aggregated.

### `HISTORY` — dynamic, the cost of a miss

| Field | Type | Engine use |
|---|---|---|
| `node_name`, `call_type` | string | join |
| `time_taken` | duration | LLM call duration → value of a hit. |
| `tool_execution_time` | duration | Tool duration → how long the prefix sits idle before it is needed again. |
| `prediction_headroom` | duration | How much lead time a prefetch has. |
| `token_input` | int | Prefix size in tokens → size in blocks. Lets the engine reason about *cost per block reclaimed*, not just value. |
| `token_output` | int | Growth per turn for react nodes. |
| `p_gpu`, `p_l1`, `p_cold` | float | Hit-class distribution. **`p_cold` is the direct test of the engine's "assume L1 resident" assumption** — it turns eviction cost from a constant into an expected value (04 §5.4). |
| `time_taken_gpu/_l1/_cold` | duration | `time_taken` segmented by hit class. A blended mean hides the quantity the policy wants. |

The last two rows are derived by the Trace Analyser from vLLM's own records
(surface C) — no workflow-side work, and nothing to instrument.

---

## 3. Transport requirements

### 3.1 The writers must `PUBLISH`, not only `SET` `[open]` — **the top blocker**

Pub/sub is the steady-state transport (01 §2). A writer that only `SET`s leaves
the engine frozen at whatever it read when it first connected: no error, no
staleness beyond the `update_ts` gate, just a policy that quietly never updates.

If changing the writers is not an option, **say so and change nothing** — Redis
**keyspace notifications** (`notify-keyspace-events`, `__keyspace@0__:*`) turn
plain `SET`s into events server-side. Same thread design on our side, different
channel names. This is a config change on the Redis instance, not a code change
in the workflow.

### 3.2 Channel naming `[open]`

Is the channel `PROB|<job_id>` itself, or one global channel with the job id
inside the payload? Decides `subscribe` vs `psubscribe` and where the filter
lives. Either works; we need to know which.

### 3.3 Message payload `[open]`

Does a published message carry the **full document**, or only "key X changed"
after which the engine must `GET`? Materially different code paths — the second
adds a round trip per message and needs its own failure handling.

### 3.4 Publish frequency `[open]`

Bounds how often the engine rebuilds its snapshot. The engine change-detects on
`update_ts` and skips unchanged snapshots, so a chatty writer is tolerable — but
**republish only on change**, not on a timer, if that is a choice.

### 3.5 Key existence is not required

`MGET` on a key that does not exist yet returns `None` → unscored → neutral
default, i.e. the block keeps its LRU position.
A job whose `PROB` row has not been written is not an error case; blocks for it
are simply ranked neutrally until the row appears.

---

## 4. Units and encoding — must be pinned `[open]`

Every one of these is a silent-corruption risk, not a crash risk. A `prob` sent
as `0-100` while the engine reads `0-1` inverts nothing and errors nowhere — it
just saturates every score and makes the policy uniformly wrong.

| Quantity | Engine assumption (change it if wrong) |
|---|---|
| `prob` | float in `[0, 1]` |
| `time_to_next_call` | **milliseconds** |
| `time_taken`, `tool_execution_time`, `prediction_headroom` | **milliseconds** |
| `update_ts` | epoch **milliseconds**, UTC |
| `token_input`, `token_output` | integer token counts |
| payload | UTF-8 JSON |

Also needed: is `PROB|<job_id>` **one JSON document covering all nodes**, or one
key per node?

---

## 5. What happens when the contract is not met

Every failure mode degrades toward today's LRU behaviour. None of them stops the
engine, and none of them is loud — which is exactly why the metrics in 01 §8 are
mandatory rather than nice-to-have.

| Failure | Engine behaviour | Detectable by |
|---|---|---|
| Redis unreachable at startup | policy inert, pure LRU | connection log, once |
| Redis dies mid-run | last snapshot serves until `update_ts` cutoff, then neutral → LRU | staleness counter |
| Writers `SET` but never `PUBLISH` | frozen at the seed snapshot; stale after the cutoff → LRU | staleness counter |
| Missed message (pub/sub is at-most-once) | stale row → LRU for that node; repaired on the next periodic reseed | reseed counter |
| Malformed JSON | that message dropped, thread survives, reconnects | decode-error counter |
| `langgraph_node` ≠ `node_name` | blocks indexed, forecast ignored, **fully silent** | per-node hit rate flat at baseline |
| `call_type` unstable across turns | index fragments; each key too sparse to score | index cardinality growth |
| `extra_args` absent entirely | request's blocks unscored → neutral default, LRU position kept | unscored-block ratio |
| `prob` in wrong units | policy uniformly wrong, no error | nothing — only an A/B against LRU catches it |

The last row is the argument for shipping behind the A/B flag from step 3 and
keeping the LRU arm one env var away.

---

## 6. What the engine does *not* expect

- **No calls into the Trace Analyser.** The engine *emits* records to it on
  surface C (a file write, fire-and-forget) and *reads* its output only via
  Redis. There is no request/response path in either direction. One ingress =
  one failure mode, one timeout policy, one staleness rule (02 §7).
- **No forecast on the request** (§1).
- **No protocol change** — `extra_args` already crosses into engine core.
- **No writes from the engine to Redis** in v1. If the per-node value table ever
  needs to survive a restart, that makes the engine a Redis *writer* and raises
  key-ownership questions (02 §8) — deliberately deferred.
- **No guarantee that a `job_id` is announced before its blocks exist.** Blocks
  can only exist after a request for that job was admitted, so the index is
  always seeded before a score is needed. Forecasts for unknown jobs are ignored,
  not an error (01 §4).

---

## 7. Answers needed, in priority order

Blocking build step 4 (the real Redis subscriber); steps 1–3.5 proceed without
them.

1. **Do the writers `PUBLISH`?** If not, can keyspace notifications be enabled?
   (§3.1 — everything else is moot if this is no.)
2. **Is `call_type` deterministic for the same physical prefix across turns?**
   (§1, requirement 2 — decides the index key shape.)
3. Channel naming: per-job or global? (§3.2)
4. Payload: full document or change notification? (§3.3)
5. Units for every field in §4.
6. `PROB|<job_id>` shape: one document for all nodes, or one key per node?
7. Redis endpoint, auth, and whether one instance is shared across jobs *and*
   across vLLM servers.
8. Publish frequency. (§3.4)
9. Field names in §2 — confirm or correct.
10. Target model: plain full attention, single KV cache group? (01 §7 asserts
    this and disables the policy otherwise.) Not a workflow question, but it
    gates the same build step.

Also needed from whoever owns the scoring: the **importance formula and its
weights**, plus the constants listed in 01 §8.

---

## 8. Minimum conformance check

Before the engine work starts, one end-to-end trace is worth more than any
schema review:

1. Run one job that exercises at least two nodes and two call types.
2. Capture, for every generation request, the `extra_args` actually sent.
3. Dump `INFO`, `PROB`, `HISTORY` for that `job_id` at three points in the run.
4. Capture the raw pub/sub stream for the job (`PSUBSCRIBE '*'` into a file).

That single artefact answers questions 1–6 and 9 at once, and it settles the
`call_type` determinism question — the one that cannot be recovered from late —
by direct observation rather than by assertion.
