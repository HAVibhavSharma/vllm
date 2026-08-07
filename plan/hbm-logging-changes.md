# HBM Block Accounting — What Changed, and Why

**What this is:** the `kv_hbm` log line, added to *both* the node-aware
eviction branch and the LRU baseline clone so the two runs can be compared
field by field.

**Status:** implemented 2026-08-07. Unit tests written but **not run** — this
machine has no working vLLM environment (see `AGENTS.md` §2); run them on the
box that has one.

---

## 1. The question this answers

Before this, the eviction policy had counters but almost nothing reached a
log. `EvictionCounters` tracked splices and evictions, `stats()` assembled
them, and `KVCacheManager.get_node_eviction_stats()` exposed them — to zero
callers. The only line that ever printed was the prefetch summary.

So four questions had no answer at runtime:

1. **How many blocks does HBM have, and how many are in use?**
2. **How many sit in the free queue as eviction candidates?**
3. **How many did the splice reshuffle?**
4. **Whose blocks were destroyed?** — the one that makes the other three
   actionable.

Question 4 is the reason this is not just `BlockPool.get_usage()`. "We evicted
12k blocks" is not something anyone can act on. "9k of them belonged to
`run-42:research`" says the forecast for one node is wrong, not that the cache
is too small.

Question 3 matters because a policy that reshuffled nothing and a policy that
reshuffled constantly look **identical in the hit rate** until the workload
changes. `splices=0` on a branch that is supposed to be splicing is a silent
failure — exactly the class of failure `07-observability.md` was written
about.

---

## 2. The line

One line, emitted by both branches, in `key=value` form:

```
kv_hbm variant=node_eviction total=24000 used=18342 free=5658 usage=76.4%
  queue=5658 splices=41 spliced_blocks=1312 evicted=8431 evicted_by_score=6120
  regret=0.083 index_keys=112 index_blocks=9130
  top_evicted=run-42:research=210,run-42:supervisor=180
```

(wrapped here; it is a single line in the log.)

| Field | Meaning | Baseline |
|---|---|---|
| `variant` | `node_eviction` or `baseline` | `baseline` |
| `total` | `BlockPool.num_gpu_blocks` — every block in HBM | same |
| `used` | `total - free` — held by running or cached-and-referenced requests | same |
| `free` | `get_num_free_blocks()` | same |
| `usage` | `used / total`, one decimal, percent | same |
| `queue` | `free_block_queue.num_free_blocks` — eviction candidates | same |
| `splices` | Ticks that reordered the queue | always `0` |
| `spliced_blocks` | Blocks moved to the head, cumulative | always `0` |
| `evicted` | Cached blocks that lost their prefix-cache identity | same |
| `evicted_by_score` | Of those, ones the policy had scored | always `0` |
| `regret` | Evictions whose key was requested again within the horizon | always `0.000` |
| `index_keys` | Distinct `job:node` keys with resident blocks | derived from the ownership map |
| `index_blocks` | Blocks currently claimed by some key | same |
| `top_evicted` | `job_id:node=<count>`, ranked, top 5 **this window** | same |

### Why policy-only fields are zero rather than omitted

LRU never reorders the free queue, so `splices` is meaningless on the
baseline. It is still printed as `0`. A field that disappears on one side
turns a mechanical field-by-field diff into manual reading, which is the thing
the shared format exists to avoid.

### `top_evicted` is per window, not cumulative

Counts reset every time a line is emitted. Two reasons, and the second is the
important one:

- The question being asked is "who is evicting **now**", not "who has ever
  evicted".
- It is what bounds the memory. A finished job stops being named instead of
  pinning its `job_id` in a dict for the lifetime of the server.

`<untracked>` collects blocks from requests that carry no `job_id` /
`langgraph_node` — benchmark traffic, warmup. It is counted rather than
dropped, so `top_evicted` always sums to `evicted` within a window.

### Rate limiting

Both branches gate the line **twice**: a wall-clock period (default 30s) *and*
a change fingerprint. An idle server logs nothing. An identical line every 30s
forever trains everyone to filter the line out, and the one time it matters is
the time it changed.

Occupancy is deliberately **not** in the fingerprint. It moves by a block on
every step, so including it would defeat the gate entirely.

One line fires at startup, before anything has happened. That is where `total`
comes from.

---

## 3. Changes — node-aware branch

Repo: `/Users/vibhavsharma/Projects/vllm` (branch `vllm-v2`)

### `vllm/v1/core/node_eviction/config.py`

- `hbm_summary_period_ms: float = 30_000.0` — period, `0` turns the line off.
- `hbm_summary_top_keys: int = 5` — how many keys `top_evicted` names.
- `validate()` rejects a negative `hbm_summary_top_keys`.

### `vllm/v1/core/node_eviction/controller.py`

- `_evictions_by_key: dict[str, int]`, `_last_hbm_log_monotonic`,
  `_last_hbm_fingerprint` in `__init__`.
- `on_block_evicted()` charges the eviction to every owner of the block. A
  block co-owned by several nodes is charged to **each**: the eviction cost
  all of them their hit. Blocks with no owner are charged to `<untracked>`.
- `hbm_summary()` — builds the line.
- `_maybe_log_hbm_summary(now)` — rate-limit + fingerprint gate, clears the
  per-window counts, and **returns the line it logged**. The return value is
  not decoration: `vllm`'s root logger sets `propagate=False`, so a test
  written against pytest's `caplog` sees nothing and passes vacuously.
- `maybe_tick()` calls it near the top, **above** the `ticks_skipped_fresh`
  and `ticks_skipped_unchanged` early returns. A tick that never reaches the
  splice is precisely the failure mode the line exists to expose; logging
  below the returns would go silent exactly when it is needed.
- `stats()` gained `hbm_total_blocks`, `hbm_free_blocks`, `hbm_used_blocks`,
  `hbm_usage`, `free_queue_len`.

### `tests/v1/core/node_eviction/test_controller.py`

`FakePool` gained `num_gpu_blocks` and `get_num_free_blocks()`. Seven tests:
pool split as an identity (`used + free == total`) rather than against
literals, attribution to `job:node`, `<untracked>` accounting, the rate limit,
the change gate, per-window reset, and the off switch.

---

## 4. Changes — LRU baseline

Repo: `/Users/vibhavsharma/Projects/baseline-vllm/vllm`

This branch has no `node_eviction` package, so the tracking is a new
self-contained module rather than a hook into an existing one.

### `vllm/v1/core/hbm_summary.py` (new)

`HBMSummaryLogger`, plus `node_label_for_request()` which reads the same
`sampling_params.extra_args` identity the policy branch reads
(`job_id` + `langgraph_node`) — no new plumbing, no protocol change.

State is one dict, `block_id -> "job:node"`. It is bounded by the pool: an
entry is added when a block is given a hash and removed when that hash is
stripped, which are the only two transitions that exist.

Cost: one dict write per newly cached block and one dict pop per eviction,
both on paths already iterating those blocks, plus a clock read per step.

Config is read straight from the environment rather than added to `envs.py` —
this is comparison instrumentation on a scratch clone, and keeping it off the
config surface keeps the diff against upstream small:

- `VLLM_HBM_SUMMARY_PERIOD_MS` — default `30000`, `0` disables.
- `VLLM_HBM_SUMMARY_TOP_KEYS` — default `5`.

### `vllm/v1/core/block_pool.py`

- Built in `__init__` when caching is on.
- `cache_full_blocks()` → `on_blocks_cached()`, claiming the new blocks.
- `_maybe_evict_cached_block()` → `on_block_evicted()`.
- `reset_prefix_cache()` → `on_reset_prefix_cache()`, which drops claims
  **without** counting evictions. An operator resetting the cache must not be
  charged to whichever job happened to be resident.

> **One non-obvious detail.** The evict hook sits *outside* the
> `popped is None` guard, which required restructuring that branch to keep the
> pop result. This mirrors where the policy branch releases its ownership
> claim. If the baseline only counted successful pops and the policy counted
> every call, `evicted=` would mean two different things and the comparison
> would be quietly wrong.

### `vllm/v1/core/kv_cache_manager.py`

- `new_step_starts()` calls `maybe_log()` — same site the policy branch ticks
  from, on the thread that owns `BlockPool`, so no locking.
- `get_hbm_summary_stats()` for a caller that wants the numbers structured
  rather than parsed back out of the line.

### `tests/v1/core/test_hbm_summary.py` (new)

Thirteen tests. The load-bearing one is
`test_line_has_the_same_fields_as_the_policy_branch`, which pins the field
list against a literal copied from the policy branch. If either side's format
drifts, that test fails instead of the comparison silently misaligning.

---

## 5. Running the comparison

```bash
# Policy branch — knobs live in NodeEvictionConfig / the JSON config file.
VLLM_NODE_EVICTION_POLICY=1 vllm serve ... 2>&1 | tee policy.log

# Baseline clone.
VLLM_HBM_SUMMARY_PERIOD_MS=30000 vllm serve ... 2>&1 | tee baseline.log

grep 'kv_hbm ' policy.log   > policy.kv_hbm
grep 'kv_hbm ' baseline.log > baseline.kv_hbm
```

Both files are `key=value` lines with identical field order, so they parse
with a two-line split and diff directly. To visualise them over time, see
[`hbm-timeline-visualisation.md`](hbm-timeline-visualisation.md).

---

## 6. What this does *not* capture

Worth stating so nobody reads more into the line than is there.

- **No per-block detail.** That is the decision log
  (`VLLM_NODE_EVICTION_DECISION_LOG`), one JSONL line per evicted cached
  block, flag-gated because it is unbounded under pressure.
- **No timestamps in the line itself.** Ordering comes from the log's own
  timestamp prefix. If you need the line to be self-contained for offline
  parsing, add a `t=` field to *both* branches — never one.
- **`used` includes the null block.** `BlockPool.get_usage()` subtracts it;
  this line does not, because `total`, `used` and `free` are meant to satisfy
  `used + free == total` exactly. Off by one against `get_usage()` is
  expected.
- **Cumulative vs. windowed is mixed on purpose.** `splices`, `evicted` and
  friends are cumulative counters; `top_evicted` is per window. A consumer
  that wants windowed rates for the cumulative fields takes differences
  between consecutive lines.
