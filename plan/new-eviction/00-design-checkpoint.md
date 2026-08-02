# Node-Aware KV Eviction — Design Checkpoint

**Status:** design, nothing implemented. No code on `vllm-v2` yet.
**Written:** 2026-07-30, against branch `vllm-v2` @ `68c7acb14`.
**Purpose:** freeze what we know before the parameter set for the importance
function arrives.
**Companions:** [`01-redis-integration.md`](01-redis-integration.md) — how the
signal gets from Redis into the engine, and the enforcement rules.
[`02-controller-scope.md`](02-controller-scope.md) — the scope expansion to a
full admission/prefetch/eviction controller, and the **canonical build order**.
[`03-workflow-contract.md`](03-workflow-contract.md) — the outward-facing
interface: what the engine needs the workflow to send, and what happens when it
doesn't.
Where the docs disagree, **the higher number wins**; Part 5 below is reconciled
with both.

Claims below are tagged `[verified]` (read at the cited line), `[decided]`
(settled in discussion) or `[open]`. Nothing here is `[unverified]` — every
file:line in Parts 1, 2 and 4 was read. (There is no Part 3; the code survey
absorbed it.)

---

## Part 0 — The idea in one paragraph

vLLM computes KV-block importance **itself**, periodically, at the granularity
of a **LangGraph node within a job** (`research`, `supervisor`, …), from data
published to **Redis** by the workflow. It maintains a
`(job_id, node) -> ordered[block_id]` index that outlives the requests that
created those blocks, and on a timer (period configurable) re-scores every
tracked node. The resulting score drives which GPU blocks get evicted under
pressure — replacing pure LRU, which evicts by age.

The scoring lives in `KVCacheManager`. The enforcement lives in `BlockPool`.
Those are different files and that split matters (Part 1). The Redis client
lives in a daemon thread in the engine-core process and never appears on the
scheduler's critical path (01 §2).

---

## Part 1 — Where eviction actually is today

`KVCacheManager` is **not** where eviction happens; it is a façade forwarding to
`KVCacheCoordinator` / `BlockPool`. The policy is entirely in `BlockPool` plus
the free-list data structure.

### The two decision points

| # | What | Where | Role |
|---|---|---|---|
| 1 | **Victim selection** | `vllm/v1/core/block_pool.py:322` `get_new_blocks()` → `:336` `free_block_queue.popleft_n(n)` | This one line *is* the eviction policy. Every popped block then goes through `_maybe_evict_cached_block` (`:354`), which drops its hash from `cached_block_hash_to_block` and kills its prefix-cache identity. `[verified]` |
| 2 | **Priority assignment** | `block_pool.py:408` `free_blocks(ordered_blocks)` → `append_n` | Blocks re-enter at the **tail**, in caller-supplied order. `single_type_kv_cache_manager.py:315` passes `reversed(req_blocks)` so tail-of-prompt is evicted before head-of-prompt. Cheapest lever in the system: changes eviction order without touching allocation. It already gives the policy correct *intra-request* order for free; the tick's splice is what orders one request's blocks against another's (01 §6). `[verified]` |

### The data structure

`FreeKVCacheBlockQueue` (`vllm/v1/core/kv_cache_utils.py:162`) — intrusive
doubly-linked list threaded through `KVCacheBlock.prev_free_block` /
`next_free_block` (`:114`). `[verified]`

| Method | Line | Cost |
|---|---|---|
| `popleft` / `popleft_n` | `:214` / `:251` | O(1) / O(n) |
| `remove` (from the middle) | `:284` | **O(1)** |
| `append` / `append_n` | `:304` / `:327` | O(1) / O(n) |
| `get_all_free_blocks` | `:352` | O(#free) |

O(1) middle removal is what makes both `BlockPool.touch()` (`:391`, rescues a
block from the free list on a cache hit) and any custom "pull *this* block"
policy possible. `[verified]`

Note what is **absent**: there is no `appendleft` and no insert-at-position, so
nothing can be placed at the head of this list today. The policy **adds
`appendleft`** — the mirror of `append`, ~5 lines against the sentinels at
`:200-212`, purely additive — because moving the worst blocks to the head is the
whole enforcement mechanism (01 §6.1).

### Things that look like hooks but are not

- `BlockPool.evict_blocks(block_ids)` (`block_pool.py:424`) and
  `KVCacheManager.evict_blocks` (`kv_cache_manager.py:441`) exist for
  **KV-connector-reported invalid blocks** (corrupt async loads), called from
  `scheduler.py:2524`. Not a policy hook. `[verified]`
- `KVCacheManager` matters only because it is the last layer still holding the
  `Request` — i.e. where policy *inputs* live, not where decisions are made.
  `[verified]`

---

## Part 2 — The previous attempt (for reference; do not port as-is)

Lives on branch **`lmcache-vllm-v2`**, forked from `cd85eb2e8`.
Diff vs merge-base: **8 files, +847 / −28** under `vllm/`. `[verified]`

Not present on `vllm-v2`. The only trace on this branch is a stale
`vllm/v1/agent_prefetch/__pycache__/eviction.cpython-311.pyc` with no source.

### Files

| File | Δ |
|---|---|
| `vllm/v1/agent_prefetch/eviction.py` | +457 (new) |
| `vllm/v1/core/block_pool.py` | +186 |
| `vllm/v1/core/sched/scheduler.py` | +62 |
| `vllm/entrypoints/openai/agent_chat/protocol.py` | +62 |
| `vllm/entrypoints/openai/agent_chat/api_router.py` | +29 |
| `vllm/v1/agent_prefetch/submitter.py` | +36 |
| `vllm/v1/request.py` | +33 |
| `vllm/v1/agent_prefetch/__init__.py` | +10 |

Plus `plan/{agent_eviction_policy,lru_vs_agent_eviction,agent_apis_eviction}.md`,
`plan/new eviction - results.txt`, `examples/online_serving/agent_eviction/`,
`tests/v1/agent_prefetch/test_eviction.py`.

### Wiring

```
client sends agent_probabilities {agent_id: P(fires soon)}   ← the assumption
  → kv_transfer_params (also forwarded verbatim into phantom prefetches)
  → Request.agent_probabilities property            request.py:+109
  → Scheduler._register_agent_eviction()            scheduler.py:+1931 (admission)
  → AgentEvictionPolicy process-wide singleton      agent_prefetch/eviction.py
        _active        req_id  -> (agent_id, probs)   [IN-FLIGHT ONLY]
        _block_owner   block_id -> agent_id           [single owner]
        _agent_blocks  agent_id -> OrderedDict[block_id]  (tag order)
  ← block_pool.cache_full_blocks   -> tag_block()
  ← block_pool.get_new_blocks      -> _claim_low_probability_blocks()
                                        -> free_block_queue.remove(block)
  ← block_pool._maybe_evict_cached_block -> untag_block()
  ← block_pool.reset_prefix_cache  -> clear_all_blocks()
```

Ranking: exclude currently-active agents (each live request implicitly
self-votes `1.0` for its own `agent_id`), rank the rest by `max` probability
across live requests **ascending**, fully drain the lowest agent's bucket, walk
up until satisfied. No threshold gate.

Trigger gate: env `VLLM_AGENT_EVICTION_FRESH_RATIO`
(`0`/`off` = disabled → pure LRU baseline arm; `1.0` default; `>1` fires sooner).

### Measured result

`plan/new eviction - results.txt` — 7 agents (`agent_a`…`agent_g`) round-robin,
1 warm-up round + 3 scored rounds, ~23.9k prompt tokens per call, cold prefill
~11.4 s TTFT.

| Arm | GPU hit rate | mean TTFT | median | p95 | sources |
|---|---|---|---|---|---|
| treatment (policy on) | **71.4 %** (10/14) | 255.0 ms | 204.2 ms | 390.5 ms | GPU=10, LMC=4 |
| baseline (LRU) | 0.0 % (0/14) | 378.1 ms | 378.6 ms | 395.0 ms | LMC=14 |

≈33 % mean-TTFT reduction. **Read the baseline carefully:** it served every
call from LMCache L1, so this measures *GPU hit vs. CPU reconstruct*
(~200 ms vs ~380 ms), **not** vs. cold prefill. That sets the real units of the
cost function (Part 5, item 4).

### The five weaknesses

1. **Forecast is client-supplied and per-request.** A request with no
   `agent_probabilities` is invisible to the policy and silently falls through
   to LRU. → *Fixed by the new design.*
2. **Votes die with the request.** `_active` holds in-flight requests only, so
   in an idle gap the aggregator is empty and `evictable_blocks()` returns `[]`.
   The forecast is most valuable exactly when nothing is running.
   → *Fixed by the new design.*
3. **The trigger gate is guesswork.**
   `fresh_free_estimate = total_free − max(0, tagged − referenced)`
   reverse-engineers "am I about to evict something cached?" from aggregate
   counters; the in-code comment admits getting it wrong collapsed the cache to
   a single prompt. Ground truth is one frame later and free —
   `_maybe_evict_cached_block` knows per block whether the victim was cached.
   → *Fixed by decoupling scoring from the allocation path.*
4. **Single-owner block tags punish shared prefixes.** `_block_owner` is one
   agent per block, so a common system preamble is stolen back and forth (the
   `RETAG` log path) and the last writer owns it. The most valuable blocks get
   misattributed. → *Still open: Part 5, item 2.*
5. **Evicts blocks, but value is per contiguous run.** Draining a bucket in
   *tag* order is not prefix order, so freeing one block can punch a hole in
   the middle of a 1500-block prefix and destroy the whole match for one block
   of reclaim. A prefix is only worth anything as a contiguous run from
   position 0. → *Still open: Part 5, item 3.*

Also: unconditional debug `print()` to stderr throughout, process-global
mutable singleton with an `RLock` guarding a race that does not exist
(Part 4), no test coverage beyond `test_eviction.py`.

---

## Part 4 — Two mechanical facts that make the new design cheap

**1. The periodic tick site already exists.** `[verified]`

`KVCacheManager.new_step_starts()` (`kv_cache_manager.py:539`) is called once
per scheduler step from `scheduler.py:459`, and forwards through
`kv_cache_coordinator.py:270` to each `single_type_kv_cache_manager`
(`:441`, `:1065`).

It runs **inside the engine-core loop, on the thread that owns `BlockPool`** —
so no locking and no background thread. The period variable is a guard right
there: every N steps, a `time.monotonic()` delta, or both.

**2. Node identity needs no protocol change.** `[verified]`

`job_id`, `langgraph_node` and `agent_id` already ride in
`sampling_params.extra_args` and cross the front-end → engine-core boundary:
- `output_processor.py:248` (`job_id`), `:251` (`agent_id`), `:254`
  (`langgraph_node`) read them out of `extra_args` front-end side, for
  per-request stats.
- `scheduler.py:799-815` reads `extra_args["manual_kv_prefilled_tokens"]`
  **engine-core side** — proof that `extra_args` is available on the
  engine-side `Request`.

So the identifiers are read off `request.sampling_params.extra_args` inside
`KVCacheManager.allocate_slots` (`:225`) / `cache_blocks` (`:515`). No new
request field, and no repeat of the old branch's `kv_transfer_params`
smuggling.

**Index key `[decided]`: `(job_id, langgraph_node)`.** Reasons in 01 §4, in
short: Redis is keyed by `job_id`, so node name alone would average unrelated
forecasts from concurrent jobs; and `langgraph_node` is the bare name that joins
directly with the Redis `node_name` field, whereas `agent_id` is namespaced
(`langgraph:supervisor`) and would need string munging. Mixing the two silently
splits a node's block set.

### Resulting shape

- **New module** — the index + scorer (successor to `agent_prefetch/eviction.py`).
- **`KVCacheManager`** — read node id at allocate/cache time; call the scorer
  from `new_step_starts()` behind the period guard; run the bounded splice.
- **`BlockPool`** — two hooks to maintain the index: `cache_full_blocks`
  (`:211`) to add, `_maybe_evict_cached_block` (`:354`) to drop; plus
  `reset_prefix_cache` (`:443`) to clear. Also maintains `num_free_fresh`, an
  O(1) counter that gates the splice (01 §6, Rule 1).
- **`FreeKVCacheBlockQueue`** — gains `appendleft` / `appendleft_n`. Additive
  only; no existing method changes behaviour, and with the policy off the new
  method is never called (Part 5, item 1; 01 §6.1).

---

## Part 5 — Decisions log

Reconciled with 01. Enforcement detail lives in 01 §6; this is the summary.

| # | Question | Outcome |
|---|---|---|
| 0 | Protecting a prefix the policy just paid to fetch | `[decided]` Today LRU protects it by accident (freed blocks go to the tail); the re-splice deletes that accident, so a **decaying score floor on speculative entries ships with the splice itself**, in build step 3 — not later. Soft floor, not a hard `ref_cnt` pin: a hard pin turns a scoring bug into a leaked-block bug. Full mechanism in 02 §5. |
| 1 | How a score becomes an eviction order | `[decided]` **Re-order the free queue on the tick.** `get_new_blocks` untouched, stays a dumb `popleft_n`. Bounded to the worst-K blocks per tick, wall-clock paced. Requires adding **`appendleft`** (~5 lines, purely additive — nothing puts a block at the head today), and the candidate set is taken from the **index**, not from `get_all_free_blocks()`, whose O(F) scan would otherwise dominate the K bound. A banded free queue is the better structure and was **rejected on blast radius** — it replaces the free list on every request path, so flag-off no longer reproduces upstream and a bug costs a block rather than a hit (01 §6.5). On-demand consultation inside `get_new_blocks` remains rejected — it is where the old branch's fragile trigger gate came from. Proactive eviction to a watermark remains a possible follow-on. |
| 2 | Blocks owned by more than one node | `[decided]` `block_id -> set[(job, node)]`, block score = `max` over owners. Single-owner reproduces the old `RETAG` thrash. |
| 3 | Within a node, which block first | `[decided]` **Tail-of-prefix first**, preserving the contiguous-run-from-position-0 invariant that `free_blocks(reversed(req_blocks))` (`single_type_kv_cache_manager.py:315`) already encodes. Position is recorded from the `cache_full_blocks` enumeration order. |
| 4 | Should L1 residency discount importance | `[decided]` **Measure, then price — three staged steps, only the first mandatory.** (1) Classify every request as GPU hit / L1 reconstruct / cold from `num_local_cached_tokens` vs `num_external_cached_tokens`, which the engine already records per request `[verified]`; the **cold fraction per node is the answer**, and if it is ~0 this item closes with no scoring change. (2) If not, publish `p_l1` / `p_cold` in `HISTORY` and make eviction cost an expected value — at a 60× ratio, `p_cold = 2%` already doubles it. (3) Only if still too coarse, track a decaying local "saved to L1 at T" belief. Querying the connector per block is **rejected** — an RPC per block and a second ingress. v1 scoring still assumes residency, but the assumption is now instrumented rather than trusted. (01 §5) |
| 5 | A/B switch | `[decided]` Policy off by default behind a flag plus the period knob — the role `VLLM_AGENT_EVICTION_FRESH_RATIO` played. |
| 6 | Unscored blocks | `[decided]` Neutral configured constant, **never 0**; sort key `(score, current LRU position)`. Since the splice only *moves* the worst K, an unscored block is simply never selected and keeps its LRU position — which is what neutral should mean, with no arithmetic. Scoring them 0 destroys freshly built prefixes before first reuse — worse than LRU. |
| 7 | Job teardown | `[decided]` Age is a **term in the score**, so silent jobs decay out; plus a hard drop at some age so the index cannot grow unbounded. |
| 8 | Phantom prefetch owner | `[decided]` From Redis (next node in `PROB`, or inferred from `HISTORY`), stamped **speculative** at insert. Confirmed on the first real prefix hit (`BlockPool.touch()`, gated on `not prefetch_only`); until then its score is `max(base, floor(age))`, the floor decaying from above the normal range to *below* neutral over `TTL = time_to_next_call`. Same mechanism supplies the protection a prefetched prefix needs — 02 §5. |
| 9 | Hybrid / sliding-window models | `[decided]` Assert a single KV cache group; disable the policy otherwise. One free queue is shared by all groups and a per-group eviction destroys the hit for every group at that position — 01 §7. |
| 10 | `call_type` granularity | `[decided]` `call_type` **is** sent on the request → index keys `(job_id, node, call_type)`, exact join with Redis. `max` applies to multi-owner blocks (item 2), which subsumes max-over-call-types — 01 §5. |
| 11 | Redis transport | `[decided]` **Pub/sub** (`psubscribe` + filter) for steady state, with a pipelined `MGET` **seed on subscribe and on every reconnect** — pub/sub delivers only post-subscribe messages and is at-most-once. `INFO` cached once per job; snapshot republished only on change; `update_ts` staleness gate backstops a missed message — 01 §2. |

---

## Part 6 — What is still needed

The **parameter set for the importance function** — the function's *shape* is now
spec'd in [`08-scoring.md`](08-scoring.md); what remains is `τ`, the cost
constants, and worst-K vs threshold-gated selection, all of which the offline
replay harness (08 §7) resolves. Plus:

- Whether `call_type` is on the request (item 10 above).
- ~~Which signals are cheap enough to increment on the hot path vs. computed
  only at tick time.~~ **Answered:** hot path carries **counters only** — block
  count per key, prefix position, speculative flag, `num_free_fresh` — all O(1).
  Everything else is tick-time.
- The decay clock for the age term (wall time, steps, requests).
- Constants: neutral default score, `update_ts` staleness cutoff, age-decay
  rate, hard-drop age, tick period, splice bound K, the `num_free_fresh` skip
  threshold, speculative floor height + decay curve + TTL×k hard drop
  (item 0 above).

Full blocker list and build order: 01 §8–§9.

---

## File index

| Path | Why it matters |
|---|---|
| `vllm/v1/core/block_pool.py` | `get_new_blocks:322`, `_maybe_evict_cached_block:354`, `touch:391`, `free_blocks:408`, `cache_full_blocks:211`, `reset_prefix_cache:443` |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock:114`, `FreeKVCacheBlockQueue:162`, `remove:284`, sentinels `:200-212`. Gains `appendleft` — the only edit to an existing core structure, and it is additive (01 §6.1) |
| `vllm/v1/core/kv_cache_manager.py` | `allocate_slots:225`, `free:418`, `cache_blocks:515`, `new_step_starts:539` |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `free:303`, `reversed(req_blocks):315` |
| `vllm/v1/core/sched/scheduler.py` | `new_step_starts()` call `:459`; `extra_args` read `:799-815` |
| `vllm/v1/engine/output_processor.py` | `agent_id` / `langgraph_node` from `extra_args` `:251-258` |
| `plan/new-eviction/01-redis-integration.md` | Redis wiring, keyspace→scoring mapping, enforcement rules, blockers, build order |
| `plan/new-eviction/03-workflow-contract.md` | The workflow-facing contract: `extra_args` fields, Redis schema, units, degradation table, the answers needed |
| `plan/new-eviction/04-trace-analyser.md` | Implementation spec for the Trace Analyser — consumes vLLM's own request records, writes `HISTORY` |
| `plan/new-eviction/05-prediction-engine.md` | Implementation spec for the Prediction Engine — writes `INFO`/`PROB`, owns `call_type` |
| `plan/new-eviction/06-walkthrough.md` | One worked scenario through LRU, the previous attempt and the new policy, with diagrams. Start here to explain the design to someone |
| `plan/new-eviction/07-observability.md` | How to tell a working policy from a broken one — decision log, counters, the score/outcome correlation, what to hand over for analysis |
| `plan/new-eviction/08-scoring.md` | The v1 scoring function — value density in ms/block, term by term, invariants, the open constants, and the offline replay harness that validates it |
| `vllm/v1/metrics/loggers.py` | `_CSV_COLUMNS:1249`, `FileStatLogger:1272`, row build `:1317` — the timing source for 04 |
| `vllm/v1/metrics/stats.py` | `FinishedRequestStats:222`, `update_from_finished_request:442`, `arrival_time:208`, monotonic-clock note `:210` |
| `plan/l0_eviction.md` | Prior write-up of the L0 data structures + eviction walk-through |
| `plan/lru_vs_agent_eviction.md` (on `lmcache-vllm-v2`) | LRU limitations, demo plan, honest caveats §7 |
| `plan/new eviction - results.txt` (on `lmcache-vllm-v2`) | The measured numbers in Part 2 |
