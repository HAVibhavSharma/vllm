# Redis → KVCacheManager Integration

**Status:** design agreed, nothing implemented.
**Written:** 2026-07-30, against branch `vllm-v2` @ `68c7acb14`.
**Reads with:** [`00-design-checkpoint.md`](00-design-checkpoint.md) — where
eviction lives today and what the previous attempt did. This doc covers how the
importance signal gets from Redis into the engine and how it overrides LRU.
[`02-controller-scope.md`](02-controller-scope.md) expands the scope from an
eviction comparator to a full admission/prefetch/eviction controller and owns
the **canonical build order** (§10).
[`03-workflow-contract.md`](03-workflow-contract.md) turns §5 and §8 of this doc
into the interface the workflow side must implement against.

`[verified]` = read at the cited line. `[decided]` = settled in discussion.
`[open]` = still needs an answer.

---

## 1. The constraint that shapes everything

`KVCacheManager`, `BlockPool` and `Scheduler` all live in **`EngineCoreProc`**,
a separate OS process from the API server (`vllm/v1/engine/core.py:1064`
`run_engine_core`; busy loop `:1164` `run_busy_loop`). `new_step_starts()` fires
inside that loop. `[verified]`

> **The engine loop must never touch a socket.** A synchronous Redis `GET` in
> the tick puts network RTT on the critical path of every scheduler step.

Everything below follows from that one rule.

---

## 2. Chosen architecture: daemon subscriber thread + atomic snapshot swap

Redis client lives **inside the engine-core process**, on its own thread. Data
reaches the loop by reference swap, never by a call. `[decided]`

**Pub/sub, with a seed read** `[decided]`. Steady-state freshness comes from
`PUBLISH`; the thread blocks on `get_message()` rather than polling on a timer.

Pub/sub delivers only what is published **after** subscribe, so a one-shot
`MGET` seed is mandatory at two points — **on first subscribe** (otherwise a
server that starts mid-job knows nothing until the next publish) and **after
every reconnect** (the gap is lost). `MGET` does not disappear; it stops being
the steady-state path and becomes the cold-start path.

```
EngineCoreProc (its own process)
├─ redis-importance thread (daemon)                        ← NEW
│    on connect / reconnect:
│      psubscribe PROB|*, HISTORY|*
│      seed: one pipelined MGET over the active jobs' keys
│      (INFO| is static — fetched once per job, cached)
│    steady state:
│      get_message(timeout=…)  ← blocks, no timer
│    change-detect on update_ts / raw-bytes hash → skip if unchanged
│    decode JSON → build immutable ImportanceSnapshot
│    self._snapshot = new_snapshot         ← single attribute assignment
│
└─ busy loop (scheduler thread)
     scheduler.py:459 → KVCacheManager.new_step_starts()   (kv_cache_manager.py:539)
       └─ if period elapsed and snapshot changed:
            snap = self._snapshot          ← one local read, no lock
            score each (job_id, node) from snap + the local block index
            write the value table
            re-splice the worst K free blocks to the head  (§6)
```

### Precedent in the codebase `[verified]`

- `ZmqEventPublisher` runs exactly this shape: `daemon=True` thread named
  `"zmq-publisher"` created in `__init__` — `vllm/distributed/kv_events.py:331`.
- `EngineCoreProc` starts two daemon IO threads itself — `core.py:890`, `:902`.

### Why no lock is needed

If the snapshot is immutable (frozen dataclass / dict never mutated after
publish), a single attribute assignment is atomic under the GIL and the reader
takes one reference. The writer is the only mutator and publishes by
replacement.

This is what retires the previous attempt's `RLock` — it was guarding a race
that does not exist once the handoff is a reference swap.

### Failure isolation (mandatory)

Redis down, malformed JSON, a dropped subscription — none of it may propagate
into the loop. The thread catches everything, logs, reconnects with backoff and
**reseeds**; the snapshot simply goes stale in the meantime.

Two hazards specific to pub/sub:

- **At-most-once delivery.** A message published while the thread is
  disconnected, or dropped because Redis killed a slow consumer
  (`client-output-buffer-limit pubsub`), is gone forever. Mitigation: reseed on
  every reconnect, plus a low-frequency **periodic full reseed** as
  belt-and-braces. The `update_ts` staleness gate below is the backstop that
  makes a silently missed message degrade to LRU rather than to a wrong answer.
- **A stalled consumer gets disconnected**, so the reconnect path is a normal
  occurrence, not an exceptional one. It must be cheap and must reseed.

Use a `get_message()` timeout rather than an indefinite block, so shutdown and
the job-id queue stay responsive.

The scorer then needs a **staleness cutoff** against `update_ts` (already in the
`PROB` payload): older than X → fall back to plain LRU. The policy degrades to
today's behaviour, not to a dead engine.

---

## 3. Rejected alternative: push over the utility RPC

Keep Redis in the API-server process and push snapshots via
`core_client.py:812 call_utility(...)` → `EngineCoreProc._invoke_utility_method`
(`core.py:1323`). `[verified]` that this path exists.

**Rejected because:** it puts periodic traffic on the engine input queue shared
with request submission, makes cache policy depend on front-end liveness, and
under `--data-parallel-size > 1` has to fan out by hand (each engine core has
its own `BlockPool`). The thread approach gets DP right for free.

Revisit only if `redis` cannot be installed in the engine environment.

---

## 4. Identity and keys

All three identifiers already ride in `sampling_params.extra_args` and cross
into engine core. No protocol change. `[verified]`

| Key | Read at | Note |
|---|---|---|
| `job_id` | `output_processor.py:248` | Redis keyspace key |
| `langgraph_node` | `output_processor.py:254` | bare node name — **matches Redis `node_name`** |
| `agent_id` | `output_processor.py:251` | namespaced (`langgraph:supervisor`) — **do not use as the join key** |

`extra_args` is provably readable engine-core side: `scheduler.py:799-815` reads
`extra_args["manual_kv_prefilled_tokens"]` there today. `[verified]`

### Index key `[decided]`

```
(job_id, langgraph_node)  →  ordered list of block_ids
block_id                  →  set[(job_id, langgraph_node)]      (multi-owner)
```

- **`(job_id, node)`, not `node` alone** — two concurrent jobs running
  `research` have different `PROB`/`HISTORY` rows and different blocks;
  collapsing them averages unrelated forecasts.
- **`langgraph_node`, not `agent_id`** — direct join with Redis `node_name`, no
  string munging, no namespace mismatch silently splitting a node's blocks.
- **Multi-owner** — a shared system preamble belongs to every node whose prefix
  starts with it. Block score = `max` over owners. Single-owner is what produced
  the old branch's `RETAG` thrash (see 00, weakness 4).
- **Ordered list, not a set** — position within the node's prefix is required
  for tail-first eviction (§6). `cache_full_blocks` receives blocks in prefix
  order, so position is the enumeration index. Positions must tolerate gaps
  (§10).

### Node and job are known from the request `[decided]`

Given that, blocks for a job can only exist *after* a request for it was
admitted, so the index is always seeded before any score is needed. "Importance
arrived for a job I've never seen" is not a correctness case — just data to
ignore.

It also means the **seed** read never needs `SCAN`: keys are **constructed**
from the active-job set, so seeding cost scales with concurrent jobs rather than
with the whole keyspace. `MGET` on a key that doesn't exist yet returns `None`
→ unscored → neutral default (§6 rule 2).

For the subscription itself, **`psubscribe` on the pattern and filter against
the active-job set** on the thread side. Per-job `subscribe` is possible — the
loop can publish newly-seen `job_id`s via a `queue.Queue` with `put_nowait`,
which is a bounded non-blocking append and touches no socket — but it buys
nothing unless one Redis serves so many jobs that decoding foreign payloads
becomes real CPU. The job-id queue is needed regardless, to drive seeding for
jobs first seen after startup.

---

## 5. The three keyspaces as scoring inputs

Redis layout (keyed by `job_id`):

| Keyspace | Fields | Role |
|---|---|---|
| `INFO` (static) | `node_name`, `type: react/non-react`, `list_call_types` | Structural prior. React nodes accumulate a growing prefix across turns; non-react don't. The fork already encodes this — `agent_kind: non-react` wipes registry entries and forces fan-out to 1. |
| `PROB` (dynamic) | `node_name`, `call_type`, `update_ts`, `time_to_next_call`, `prob` | The forecast the old attempt took from the client, now server-pulled. `update_ts` is the staleness gate. |
| `HISTORY` (dynamic) | `node_name`, `call_type`, `time_taken`, `tool_execution_time`, `prediction_headroom`, `token_input`, `token_output` | Value of a hit in ms; size in blocks via `token_input`. |

`call_type` is a flattened leaf name — `summary` for a direct LLM call,
`tavily:summary` for an LLM call nested in a tool call.

### Granularity — resolved `[decided]`

One node has **several** `PROB`/`HISTORY` rows, one per call type, each with its
own `prob` and `time_to_next_call`.

**The workflow will send `call_type` in `extra_args`**, so the index keys at
`(job_id, node, call_type)` and the join with Redis is exact — no aggregation on
the way in, no information lost. `max` applies only where it is genuinely
needed: a block owned by several keys (shared preamble) takes the max of its
owners' scores, which subsumes max-over-call-types for free.

Remaining risk `[open]`: if the flattened `call_type` is **not deterministic**
for what is physically the same cached prefix (`summary` one turn,
`tavily:summary` the next), the index re-keys the same blocks every turn and
fragments. Needs confirming.

### L1 residency — measure, then price `[decided]`

The v1 scorer still **assumes L1 residency**, making eviction cost a constant
(~180 ms: GPU hit ≈200 ms vs LMCache reconstruct ≈380 ms). But the assumption is
no longer *accepted on faith* — it is measured, in three staged steps, and only
the first is mandatory.

**Step 1 — measure it. The data already exists.** `[verified]`
`FinishedRequestStats` carries `num_local_cached_tokens` and
`num_external_cached_tokens` **separately**, per request, already labelled with
`job_id` / `langgraph_node` (`stats.py:222`, emitted at `loggers.py:1317`).
That classifies every request:

| Class | Signature | Cost |
|---|---|---|
| GPU hit | `local > 0` | ≈200 ms |
| L1 reconstruct | `local == 0`, `external > 0` | ≈380 ms |
| **cold** | both `0` | **≈11 400 ms** |

The **cold fraction per `(node, call_type)` is the answer to this question.**
If it is ~0 the assumption holds and this item closes with no scoring change.
The Trace Analyser computes it as part of the hit-class segmentation it already
owns (04 §5.3) — this costs nothing extra.

**Step 2 — if the cold fraction is non-trivial, price it.** Publish `p_l1` and
`p_cold` per `(node, call_type)` in `HISTORY` and make the eviction-cost term an
expected value:

```
E[cost of evicting] = p_l1 · 180ms  +  p_cold · 11400ms
```

Note the asymmetry: at a 60× cost ratio, `p_cold = 2%` already doubles expected
cost. Precision is not the goal — **not treating 11.4 s as 180 ms** is.

**Step 3 — only if step 2 is still too coarse.** Track residency locally: the
engine knows when LMCache stored a prefix (`request_finished` returning
`delay_blocks=True`, `lmcache_mp_connector.py:1038`) `[verified]`, so it can hold
a per-prefix "saved at T" belief that decays. Approximate, local, no new
dependency.

**Rejected: querying the connector for residency.** A per-block lookup at tick
time is an RPC per block, and it introduces a **second ingress** into the process
that owns the scheduler loop — the thing §7 and 02 §7 exist to prevent. The
one-ingress rule is worth more than the precision.

**The metric that says which step you are on:** cold-miss rate per node. Flat at
zero → step 1 closed it. Non-zero → step 2 pays for itself immediately.

This measurement is worth having regardless of scoring: it is also the only way
to tell whether the policy beats LRU on the axis that matters, rather than on the
~180 ms axis the old benchmark actually measured (00 Part 2).

### Age `[decided]`

Age is a **term in the importance score**, not a separate GC pass — a silent job
decays out naturally. Still needs a hard drop at some age or the index
accumulates zero-score entries for the server's lifetime.

---

## 6. Enforcement: bounded re-splice on the tick `[decided]`

`get_new_blocks` is **not modified** — it stays a dumb `popleft_n`
(`block_pool.py:336`). The policy acts by re-ordering the queue it pops from,
on the tick, bounded to K blocks.

**The design choice, stated plainly:** a banded free queue (score-partitioned
lists, order by construction) was considered and rejected — not on cost, but on
blast radius. It *replaces* the structure every allocation and every free goes
through, so a banding bug is a lost or double-owned block rather than a bad
eviction, and flag-off no longer reproduces upstream behaviour. The re-splice
stays **additive**: one new method, one new tick body, and an off switch that
means literally nothing runs. §6.5 records the rejected design and the argument.

Two findings that previously blocked the re-splice are resolved here: the
missing head insertion (§6.1) and the O(F) scan (§6.4).

### 6.1 `appendleft` — the one primitive to add `[required]`

Today's ops `[verified]`:

| Op | Line | Lands |
|---|---|---|
| `popleft` / `popleft_n` | `:214` / `:251` | takes from **head** |
| `append` / `append_n` | `:304` / `:327` | puts at **tail** |
| `remove` | `:284`, O(1) | anywhere |

Nothing puts a block at the head, so "move the worst-K to the head" is not
expressible today. Add `appendleft` / `appendleft_n` — the mirror of `append`,
about five lines, using the `fake_free_list_head` / `fake_free_list_tail`
sentinels that already exist (`:200-212`). `[verified]`

This is **purely additive**: no existing method changes behaviour, and with the
policy off the new method is never called. That property is the whole reason
this design was chosen over banding.

The rejected alternative was to invert — move the *best*-K to the tail using the
existing `append_n`, needing no new primitive. It protects the top instead of
condemning the bottom, which leaves everything beneath the protected set in LRU
order exactly when pressure is deep enough to reach it. Not worth saving five
lines.

### Rule 1 — fresh blocks stay ahead of cached ones `[decided]`

A fresh block is exactly `block.block_hash is None`. It must be consumed before
any cached block is destroyed — using one costs nothing, destroying a cached
block costs its prefix.

`appendleft` creates a hazard the old write-up did not have: it inserts at the
**absolute** head, ahead of any fresh blocks sitting there. Splicing a cached
block in front of a fresh one inverts this rule and throws away a prefix while a
free non-cached block was available.

**Fix: an exact counter, not an estimate.** `BlockPool` maintains
`num_free_fresh` — incremented when a block with `block_hash is None` enters the
queue, decremented when one leaves. Both sites already inspect the block
(`free_blocks:408`, `get_new_blocks:336`, `touch:403`), so this is O(1)
bookkeeping with no scan. The tick then **skips the re-splice entirely while
`num_free_fresh > threshold`** — there is nothing to gain from ordering cached
blocks that will not be reached.

This is the honest version of the old branch's weakness 3 (00 Part 2). That
design *reverse-engineered* "am I about to evict something cached?" from
aggregate counters and admitted in-code that getting it wrong collapsed the
cache to a single prompt. This counts the exact quantity at the two places it
changes. Same purpose, no arithmetic.

Residual `[open]`: when the counter is at the threshold, up to K cached blocks
may still be spliced ahead of a few fresh ones. Bounded by K and only at the
boundary — acceptable, but worth a metric.

### Rule 2 — unscored blocks get a neutral default `[decided]`

Sort key is `(score, current LRU position)`. Unscored blocks (no Redis row yet,
or past the staleness cutoff) take a **configured neutral constant, never 0** —
scoring them 0 makes freshly built prefixes the top eviction candidates and
destroys them before first reuse, which is strictly worse than LRU.

Because the splice only *moves* the worst K, an unscored block is simply never
selected: it keeps its LRU position, which is exactly what a neutral score
should mean. No neutral-score arithmetic is needed at all.

### Rule 3 — within a node, tail-of-prefix first `[decided]`

A node's prefix is only worth anything as a contiguous run from position 0.
Freeing a block from the middle of a 1500-block prefix reclaims one block and
destroys the whole match. `free_blocks(ordered_blocks)` appends in
**caller-supplied order** (`block_pool.py:408`) and
`single_type_kv_cache_manager.py:315` already passes `reversed(req_blocks)`.
`[verified]` So intra-request order is already correct and free; the splice
orders one request's blocks against another's.

Within one tick, splice the selected K in **ascending score order** so the worst
block ends up nearest the head after successive `appendleft` calls. Get this
backwards and the policy evicts its best blocks first — a sign error with no
symptom other than an inverted hit rate, which is why the A/B arm and the
eviction-by-score counter (§8) are not optional.

### 6.4 Drive the splice from the index, not from the queue `[decided]`

The splice cannot be incremental — four things mutate the queue between ticks
`[verified]`:

| Mutation | Where | When |
|---|---|---|
| `popleft_n` | `block_pool.py:336` | every allocation |
| `append_n` | `:420` | every request free |
| `remove` | `:403` (`touch`) | **every prefix-cache hit** |
| `remove` + null replace | `single_type_kv_cache_manager.py:385-407` | every step, windowed/mamba |

The obvious implementation recomputes from `get_all_free_blocks()`
(`kv_cache_utils.py:352`) — but that is **O(F)**, ~24k iterations of interpreted
pointer-chasing, and it dominates the K bound by about two orders of magnitude.
K would bound only the relink, not the tick.

**Iterate the index instead.** The block index (build step 1) already holds every
scored block; "is it free" is `block.ref_cnt == 0`, checkable per block in O(1).
So:

```
tick:
  if num_free_fresh > threshold:  return        # Rule 1, O(1)
  candidates = worst K by score over indexed blocks with ref_cnt == 0
  for block in ascending score order:
      free_block_queue.remove(block)            # O(1)
      free_block_queue.appendleft(block)        # O(1)
```

Cost is O(tracked) for selection — a heap keeps it O(tracked log K) with no sort
— and O(K) for the relink. **`get_all_free_blocks()` is never called**, and
notably it has no caller anywhere outside `kv_cache_utils.py` today.

Unscored blocks are never visited, which is both the cheap path and the correct
one (Rule 2).

### Rule 4 — bounded and wall-clock paced `[decided]`

- **Bounded K** — move only the worst-scoring K blocks per tick (knob, ~256).
  Converges over several ticks; approximate order is fine for a heuristic.
- **Wall-clock cadence**, not step count — step duration varies too much.
- **Skip the tick** when the snapshot object is unchanged (`is` check) and no
  eviction happened since the last one.
- **Skip the tick** while fresh blocks are plentiful (Rule 1).

**Helpful inversion:** F is large only when the server is idle (most blocks
free) and small under pressure (most blocks referenced, hence not in the queue).
The expensive case is the one where a stall costs nothing — and iterating the
index rather than the queue removes even that.

### 6.5 Rejected: banded free queue

Partition the free list into B lists by score band; insert into the right band at
`free_blocks`, pop from the lowest non-empty band, repair score drift lazily at
pop via an epoch stamp. Order becomes a property of the structure, the tick never
touches the queue, and no head insertion is needed.

It is the better data structure and it was still rejected, for one reason:
**it replaces a structure on every request path rather than adding beside one.**

| | Re-splice + `appendleft` | Banded queue |
|---|---|---|
| Change shape | additive: one new method | replaces `FreeKVCacheBlockQueue` |
| Flag off | nothing runs; byte-identical to upstream | structure is live regardless |
| Worst bug | bad eviction order → slower, still correct | block in two bands or none → double-allocated KV or a starved pool |
| Upstream merges | rebases cleanly | conflicts whenever `kv_cache_utils.py` moves, resolved against our invariants |
| Tick cost | O(tracked log K) + O(K) | O(tracked), no queue contact |

The cost column favours banding; the risk column decides it. Revisit if the
splice is measured to be too expensive, or once the policy has earned enough
confidence that "flag off is upstream" stops being the safety property that
matters.

A middle option if that day comes: keep both behind the flag and choose the
structure at init, so flag-off remains byte-identical.

---

## 7. Hazards

### Hybrid / sliding-window models — scope out of v1 `[decided, needs confirmation]`

There is **exactly one** free queue, shared by every KV cache group:
`KVCacheCoordinator.__init__` creates one `BlockPool`
(`kv_cache_coordinator.py:50`) and hands the same instance to every
`single_type_manager` (`:66-71`); that pool builds one linked list over all
blocks (`block_pool.py:168`). Full-attention, sliding-window and mamba blocks
are interleaved in it. `[verified]`

Null blocks are a non-issue — exactly one (`block_id=0`), popped at init and
never returned (`block_pool.py:176-177`), and `free_blocks` already filters
`not block.is_null` (`:421`). `[verified]`

The real hazard: for hybrid models a cache hit at prefix position *p* requires a
hit in **every** group — `get_cached_block` returns `None` if any group misses
(`block_pool.py:184-189`). So evicting one group's block at *p* destroys the hit
at *p* for all groups while reclaiming a single block. `[verified]` Compounding
it, groups may have different block sizes
(`block_size % hash_block_size == 0`, `kv_cache_coordinator.py:430-434`), so
equal block counts ≠ equal value. `[verified]`

**Decision:** assert a single KV cache group at policy init; if
`len(kv_cache_groups) > 1`, log once and disable the policy (fall back to LRU).
Plain full-attention models build a `UnitaryKVCacheCoordinator` and the whole
class of problem is inert.

### Phantom prefetch attribution `[decided]`

Prefetches call `cache_blocks()` to register blocks in APC, so those blocks need
an owner. It comes from Redis — the next node from `PROB`, or inferred from
`HISTORY`.

Consequence: the owner is a **predicted** node, so the index gains entries whose
blocks no real request ever touched, and a wrong prediction *protects* useless
blocks at the predicted node's score. Index entries therefore need a
**speculative / confirmed** distinction, with speculative entries demoted if no
real request for that node arrives. The old branch hit this — commits *"fixed
the prefetch tagging issue"* and *"removed the double eviction of prefetch"*.

**Resolved in [`02-controller-scope.md`](02-controller-scope.md) §5** `[decided]`,
jointly with the pin problem — they are the same defect, an index entry created
by a prediction rather than an observation. In short:

- **Stamp** `speculative=True, created_at` at insert, discriminated by
  `request.kv_transfer_params["prefetch_only"]` (`scheduler.py:2321`), readable
  where `cache_full_blocks` already receives the `Request`. `[verified]`
- **Confirm** on the first *real* prefix hit — `BlockPool.touch()`
  (`block_pool.py:391`) is that signal, driven from
  `KVCacheManager.allocate_slots` (`kv_cache_manager.py:225`), the only layer
  holding both the `Request` and `new_computed_blocks`. `[verified]`
  Gated on `not prefetch_only`, or one phantom confirms another's guess.
- **Demote by decay, not by a second pass**: `score = max(base, floor(age))` for
  speculative entries, the floor decaying from above the normal range to *below*
  the neutral default over `TTL = time_to_next_call`. A falsified prediction
  becomes the preferred victim rather than merely losing protection.

### Position gaps `[open]`

`remove_skipped_blocks` (`single_type_kv_cache_manager.py:385-407`) replaces
mid-list blocks with the null block for windowed/mamba models, freeing them
while the owning request still runs. Recorded prefix positions must tolerate
gaps, and a run with a hole has a different tail than one without.

---

## 8. Blockers

**Waiting on input** — restated as an outward-facing contract with priorities and
a conformance check in [`03-workflow-contract.md`](03-workflow-contract.md) §7.
That doc is what to send the workflow owners; the list below is the same content
in engine terms.

- **Do the writers actually `PUBLISH`?** Pub/sub is now the steady-state
  transport, so a writer that only `SET`s leaves the engine frozen at its seed.
  If changing the writers isn't an option, Redis **keyspace notifications**
  (`notify-keyspace-events`, `__keyspace@0__:*`) turn plain `SET`s into events
  server-side — same thread design, different channel names.
- **Channel naming**: is the channel `PROB|<jobid>` itself, or a global channel
  with the job id inside the payload? Decides `subscribe` vs `psubscribe` and
  the filter.
- **Message payload**: does a message carry the **full document**, or only a
  "key X changed" notification the thread must then `GET`? Different code paths.
- Is the flattened `call_type` deterministic for the same physical prefix across
  turns? (§5) — if not, the index fragments
- Payload shape: `PROB|<jobid>` = one JSON list covering all nodes, or one key
  per node?
- Units: `prob` as `0-1` or `0-100`; `time_to_next_call` /
  `prediction_headroom` in ms
- Redis endpoint + auth; one instance shared across jobs and servers, or per job
- Publish frequency — bounds the snapshot rebuild rate
- Target model: plain full attention / single KV cache group? (§7 assert)
- Importance formula weights — shape is spec'd in
  [`08-scoring.md`](08-scoring.md); `τ` and the cost constants remain open and
  are resolved by replay (08 §7), not by asking
- Constants: `get_message` timeout, periodic-reseed interval, tick period,
  neutral default score, `update_ts` staleness cutoff, age-decay rate,
  hard-drop age, **splice bound K**, the `num_free_fresh` skip threshold (§6
  Rule 1), speculative floor height, its decay curve, and the TTL×k speculative
  hard drop (02 §5)

**Answered**
- `call_type` **is** on the request → key `(job, node, call_type)` (§5)
- Transport is **pub/sub**, with an `MGET` seed on subscribe and on every
  reconnect (§2)
- Speculative vs confirmed ownership, and protection of prefetched blocks —
  one decaying floor + confirm-on-touch (§7 above, 02 §5)

**Decisions to confirm**
- Assert single KV cache group; disable on hybrid (§7)
- Bounded-K re-splice driven from the index, with `appendleft` added (§6)

**Unresolved design**
- Cold prefixes mis-scored under "assume L1 resident" (≈11.4 s, not ≈180 ms) —
  **method decided, §5**: measure `p_cold` from data the engine already emits,
  then price it as an expected cost if it is non-trivial. Unresolved only until
  step 1 runs, which needs no new code on either side
- Hard drop for silent jobs so the index cannot grow unbounded
- Splice cost never measured. Cheap to settle with no GPU and no model: build a
  queue of F ≈ 24k, then time K=256 `remove` + `appendleft` pairs plus the
  index-side selection, against a scheduler step of ~10-50 ms. If it lands near
  the step budget, the banded queue in §6.5 is the fallback

**Engineering**
- `redis` is not a dependency, not installed, and there is no `.venv` in this
  checkout — needs adding to `requirements/` behind a guarded import so vLLM
  still boots without it
- One subscriber thread + index per engine-core process under data parallelism
- `reset_prefix_cache` (engine path and `/v1/agents/reset_prefix_cache`) must
  clear the index
- No Prometheus metrics anywhere in this fork — without eviction-by-score and
  per-node hit-rate counters a silently inverted policy is undetectable. The old
  branch used unconditional `print()` to stderr; that cannot ship.
  **Design in [`07-observability.md`](07-observability.md)**: always-on counters
  plus a flag-gated eviction decision log, joined against the request stats
  vLLM already writes. Primary signal is the score/outcome correlation — one
  scalar that detects an inverted policy (07 §5.1). Ships with step 3
- No tests; need fake-snapshot tests that run with no GPU and no Redis
- Benchmark needs an undersized cache, a stated baseline, and the
  noisy-forecast arm the old caveats doc called for but never ran
- Policy off by default behind a flag, so baseline LRU is one env var away —
  the role `VLLM_AGENT_EVICTION_FRESH_RATIO` played

---

## 9. Build order — moved

**Superseded by [`02-controller-scope.md`](02-controller-scope.md) §10**, which
is the canonical sequence. It keeps steps 1–3 from here unchanged and adds the
per-request value table, the prefetch/admission instructions, and the optional
demote path.

Kept here only as orientation: the foundation is **index → subscriber/snapshot →
re-splice enforcement**, each testable before the next exists, and all three are
unblocked.

---

## 10. File index (additions to 00's index)

| Path | Why |
|---|---|
| `vllm/v1/engine/core.py` | `run_engine_core:1064`, `run_busy_loop:1164`, daemon IO threads `:890`/`:902`, `_invoke_utility_method:1323` |
| `vllm/v1/engine/core_client.py` | `call_utility:812` (rejected path) |
| `vllm/distributed/kv_events.py` | `:331` daemon-thread precedent |
| `vllm/v1/core/kv_cache_coordinator.py` | one `BlockPool` `:50`, shared to managers `:66-71`, hybrid block sizes `:430-434` |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `remove_skipped_blocks:385-407`, `reversed(req_blocks):315` |
| `vllm/v1/engine/output_processor.py` | `job_id:248`, `agent_id:251`, `langgraph_node:254` |
