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

So six questions had no answer at runtime:

1. **How many blocks does HBM have, and how many are in use?**
2. **How many sit in the free queue as eviction candidates?**
3. **How many did the splice reshuffle?**
4. **Whose blocks were destroyed?**
5. **Did the cache work?** — the hit rate.
6. **How much KV did we have to rebuild?** — the movement number.

Question 4 is why this is not just `BlockPool.get_usage()`. "We evicted 12k
blocks" is not something anyone can act on. "9k of them belonged to
`run-42:research`" says the forecast for one node is wrong, not that the cache
is too small.

Question 3 matters because a policy that reshuffled nothing and a policy that
reshuffled constantly look **identical in the hit rate** until the workload
changes. `splices=0` on a branch that is supposed to be splicing is a silent
failure — exactly the class of failure `07-observability.md` was written
about.

Question 6 is **the claim the policy is making**. See §2.2 — it is the one
number that is not satisfied by "buy more HBM".

---

## 2. The line

One line, emitted by both branches, in `key=value` form:

```
kv_hbm variant=node_eviction total=24000 used=18342 free=5658 usage=76.4%
  queue=5658 splices=41 spliced_blocks=1312 evicted=8431 evicted_by_score=6120
  regret=0.083 hit_rate=0.7412 hit_rate_win=0.8033 hit_tokens=9182304
  query_tokens=12388291 remat_blocks=4118 remat_mb=8236.0 remat_ratio=0.1902
  blocks_cached=21650 parsed_to_next_node_ms=2411.6
  parsed_to_next_node_p50_ms=2180.0 parsed_to_next_node_min_ms=612.0
  parsed_to_next_node_n=23 prefetch_gb=3.771 prefetch_blocks=2878
  prefetch_tokens=46048 prefetch_keys=9 prefetch_unsized=1
  index_keys=112 index_blocks=9130
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
| `hit_rate` | Prefix-cache hit rate **in tokens**, since boot | same |
| `hit_rate_win` | Same, this window only | same |
| `ttft_ms` | Engine-side time to first token, mean since boot | same |
| `ttft_win_ms` | Same, this window only | same |
| `ttft_p50_ms` | Median over the last 4096 samples | same |
| `ttft_p95_ms` | 95th percentile over the last 4096 samples | same |
| `ttft_n` | Samples taken, i.e. requests that produced a token | same |
| `hit_tokens` / `query_tokens` | The raw numerator and denominator | same |
| `phantom_hit_rate` | Hit rate of prefetch phantoms only, excluded from every rate above | always `0.0000` |
| `phantom_query_tokens` | Tokens looked up by phantoms — the prefill origination bought | always `0` |
| `remat_blocks` | **Movement.** Blocks cached again after being evicted | same |
| `remat_mb` | The same in MB of KV rebuilt | same |
| `remat_ratio` | `remat_blocks / blocks_cached` | same |
| `blocks_cached` | Every block ever given a hash | same |
| `parsed_to_next_node_ms` | **Prefetch budget.** Mean `time_to_next_call` over the forecast rows the Redis reader parsed | same |
| `parsed_to_next_node_p50_ms` | Median of the same | same |
| `parsed_to_next_node_min_ms` | The tightest gap any predicted call is working with | same |
| `parsed_to_next_node_n` | Rows that carried a gap at all — the population behind the three above | same |
| `prefetch_gb` | **Prefetch volume.** KV behind every predicted-and-absent prefix, priced at this model's page size | same |
| `prefetch_blocks` | The same in blocks | same |
| `prefetch_tokens` | The same in tokens (`blocks x block_size`) | same |
| `prefetch_keys` | Forecast keys past every admission gate and not resident | same |
| `prefetch_unsized` | Of those, ones with no observed prefix length — they price at zero, so this is how much of a floor `prefetch_gb` is | same |
| `index_keys` | Distinct `job:node` keys with resident blocks | derived from the ownership map |
| `index_blocks` | Blocks currently claimed by some key | same |
| `top_evicted` | `job_id:node=<count>`, ranked, top 5 **this window** | same |

### 2.1 Hit rate is in tokens, not requests

A 10-token request that hit fully and a 10 000-token request that missed
entirely are not "one hit and one miss". The second cost a thousand times
more. Both branches take `num_tokens` and `num_hits` straight from
`KVCacheManager.get_computed_blocks`.

Counted independently of `PrefixCacheStats`, deliberately. That object is
*drained on read* by whoever polls the metrics loggers, so sharing it would
make the line's hit rate depend on whether something else was scraping — and
on `log_stats` being on at all.

**And counted once per request, not once per scheduling attempt.** The
scheduler calls `get_computed_blocks` on every step a request spends in the
waiting queue — `sched/scheduler.py`, guarded only by
`num_computed_tokens == 0` — so counting per call yields a hit rate weighted
by queueing delay, which is set by the very eviction pressure the line exists
to measure. It biases against whichever arm queues more. On the 2026-08-09
run this produced **779,478,224 query tokens against ~440,000 real prompt
tokens**, a factor of ~1771, with one 33-second window contributing 87.9M.
`Request.cache_query_counted` now gates it, identically on both branches.

The `preempted` split is unaffected: it covers genuine re-queries after
preemption, where `num_computed_tokens > 0`, and those still deserve
exclusion. Note that the 2026-08-09 run had `query_tokens_fresh ==
query_tokens` exactly — the re-queries were *not* preemptions, which is why
that filter never caught them.

One consequence: `hit_rate_fresh` no longer reconciles with vLLM's own
`Prefix cache hit rate` line. Upstream's counter is left counting every call
so it stays comparable to stock vLLM; ours is the corrected one. Expect the
two to disagree, and by a wide margin under queueing.

`hit_rate_win` exists because a cumulative rate over a multi-hour run is
dominated by whatever the workload did in its first ten minutes. When the
question is "is it better *now*", the cumulative figure cannot answer it.

### 2.1.0 `ttft_*` — what the miss cost

Hit rate says how often the cache worked. It cannot say whether that
mattered: a policy can raise hit rate and still lose on latency if the blocks
it kept were cheap to rebuild and the ones it dropped were not. `remat_mb`
says how much KV was rebuilt; `ttft_*` says what it cost.

**Engine-side, deliberately.** `Request.first_token_ts - Request.arrival_time`,
both stamped inside engine core — `first_token_ts` in
`Request.append_output_token_ids`, the one point every path appends through.
The front end's TTFT additionally carries front-end queueing and
detokenization, neither of which an eviction policy can move, so including
them only dilutes the effect being measured. What is left is scheduler
queueing plus prefill, which is exactly what a cache miss pays for. Expect
this to read *lower* than the front end's number; they are not the same
quantity.

Sampled from `KVCacheManager.free`, the only layer still holding the
`Request` at teardown. That path also runs on **preemption**, and a preempted
request keeps its original `first_token_ts`, so the sample is gated on
`Request.ttft_recorded` — otherwise slow requests would be weighted by how
often they were preempted. A request that never produced a token contributes
nothing rather than a zero.

The mean is exact and cumulative; `ttft_p50_ms` and `ttft_p95_ms` come from a
bounded ring of the last 4096 samples, because retaining every sample for an
exact percentile is unbounded under load. Both percentiles are drawn from that
one ring, so the median-to-p95 spread describes a single population.

Read all three together. TTFT is heavy-tailed, so the mean can sit somewhere
no request actually was: nine 100 ms prefills and one 30 s cold miss average
to 3.09 s, which describes none of the ten. The median says what a typical
request saw, the p95 says what the worst 5% saw, and the mean says what the
work cost in aggregate. **A mean that moved without the median is a tail
effect** — quoting it as "requests got faster" is the specific mistake the
median was added to prevent. The converse is also a finding: median down with
the mean flat means the common case improved and one straggler ate the gain.

### 2.1.1 Phantoms are excluded, not discarded

A phantom prefetch queries the prefix cache like anything else, so until this
was fixed it landed in both numerator and denominator. That is worse than
noise, because the sign is not consistent: a phantom's first prefill is a
near-total miss that *deflates* the rate, while a phantom for an
already-resident prefix is a large hit no user ever experienced and it
*inflates* it. The baseline this line exists to be diffed against originates
no phantoms at all, so either direction shows up as a policy difference that
never happened.

`KVCacheManager.get_computed_blocks` now passes the `Request` through to
`on_cache_query`, which routes it with the same `_is_prefetch_only` test
`on_blocks_cached` and `on_prefix_hit` already use. Phantom tokens go to
`phantom_hit_rate` / `phantom_query_tokens` and touch nothing else.

Read `phantom_hit_rate` as a *cost* line, not a success line. High means the
residency test admitted a want for a prefix HBM still held and the phantom
bought nothing; low with a large `phantom_query_tokens` means real prefills
the policy chose to run — which with `--max-num-seqs 1` are serialized ahead
of real traffic (`new-eviction/12-first-run-findings.md` §5.6).

### 2.1.2 `reset_prefix_cache` zeroes the rates

It used to clear only the rematerialisation ring, so token counts survived a
cache wipe and went on describing a cache that no longer existed — visible as
a first post-reset run whose hit rate was really the previous run's. All
counters now reset together. This also matches upstream, where
`CachingMetrics.observe` resets its own aggregation on `stats.reset`
(`vllm/v1/metrics/stats.py:68`); without it the two figures were permanently
out of step after the first reset.

### 2.2 `remat_*` is the movement number — read this before quoting it

**A block counts as movement when its hash is cached again after having been
evicted.** The block existed, we threw it away, we rebuilt it. That is work
the eviction forced, and a better policy avoids it. `remat_mb` converts it to
MB using the summed `page_size_bytes` across KV cache groups.

Three things it deliberately is **not**:

- **Not the miss count.** Misses include every prefix the server has never
  seen. No eviction policy can do anything about those, and counting them
  would credit the policy for a cold start and bury the real signal under
  first-contact traffic. Rematerialisation is zero for a first-time prefix by
  construction.
- **Not hit rate.** Hit rate is necessary but not sufficient: a bigger cache
  raises it too, so on its own it cannot separate a better policy from more
  HBM. `remat_mb` is specifically the work the *eviction decision* caused.
- **Not counted twice for a popular prefix.** Once a hash is counted, it is
  dropped from the evicted ring, so a second count needs a second eviction.
  Without that, every subsequent cache of a hot prefix would read as movement
  and the number would grow without anything being rebuilt.

The claim this supports is: *"same hit rate, N GB less KV rebuilt"* — or
better, both improved. That is the sentence `remat_mb` exists to let you
write.

**Bounded by a ring**, default one full pool turnover
(`remat_window_blocks=0` → `num_gpu_blocks`). A block evicted longer ago than
that is not something this policy is still answerable for, and the ring is
also what caps the memory. Note the ring can hold a stale duplicate after a
hash is counted and discarded from the membership set; the effect is a
slightly shorter window for that one hash, so the error is bounded and can
only **under**-count movement — the right direction for a number used to
claim an improvement.

`remat_mb` reports `0.0` when the page size was never configured, rather than
guessing. A fabricated byte count is worse than an absent one when the whole
point is comparing two runs.

### 2.3 `parsed_to_next_node` and `prefetch_gb` are one reading, not two

`remat_*` says what the eviction decision cost after the fact. These two say
whether the *prefetch* decision could ever have paid — and neither half means
anything alone.

**`parsed_to_next_node_*` is the budget.** It summarises `time_to_next_call`
over the `PROB` rows the Redis reader parsed (`snapshot.py:_decode_one`), which
is the same interval the workload harness measures as
`t3_parsed_to_next_start_ns`: routing decision parsed → next node starts. Rows
that carry no gap yet publish `0` and are **excluded rather than averaged in** —
a mean dragged to zero by unmeasured rows reads as "there is no time to
prefetch", which is a conclusion about missing data, not about the workload.
`parsed_to_next_node_n` is the surviving population, so a small `n` is visible
next to the number it produced.

**`prefetch_gb` is the volume.** Every forecast key past the admission gates
(`_is_resident`, staleness, `prefetch_min_prob`, `prefetch_horizon_ms`) that
HBM does not hold, priced at the summed `page_size_bytes`. It is a **gauge, not
a total**: it falls when prefixes land as well as when the forecast moves on,
and `0.000` is the good state — everything predicted is already resident.

Read together they are a bandwidth question: `prefetch_gb / (min_ms / 1000)` is
the sustained L1→HBM rate needed to land the whole demand inside the tightest
gap. Above the link's rate, a prefetch miss is a *physics* result, not a
forecast error, and no amount of ranking work will fix it. The timeline page
prints that ratio as a banner for exactly this reason.

Two honesty caveats, both deliberate:

- **A key is priced at the longest prefix ever observed under it**, held in a
  bounded LRU outside the ownership index. The index cannot answer this: it
  deletes an entry when the last block is evicted (`index.py:133`), which is
  precisely the state a key is in when it becomes something to prefetch.
  Pricing on the surviving fraction would say a prefix gets cheaper to restore
  the more of it has been thrown away.
- **A key never observed resident prices at zero** and is counted in
  `prefetch_unsized`. `prefetch_gb` is therefore a *lower bound* whenever that
  field is non-zero, and the page refuses to quote it as a measurement past a
  third of the keys.

Both fields are emitted on the **baseline** arm too, with no want-list in
play. A field that exists on only one side cannot be diffed, and the baseline's
demand curve is the counterfactual the policy arm's claim is made against.

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
- `remat_window_blocks: int = 0` — evicted-hash ring size, `0` sizes it to
  the pool.
- `validate()` rejects negatives on both counts.

### `vllm/v1/core/node_eviction/metrics.py`

- `CacheMovementTracker` — hit rate (cumulative and windowed) and
  rematerialisation, with the bounded evicted-hash ring. Kept here rather
  than in the controller because it is observability, not policy: it makes
  no decisions and the controller only feeds it.

### `vllm/v1/core/node_eviction/controller.py`

- `_evictions_by_key: dict[str, int]`, `_last_hbm_log_monotonic`,
  `_last_hbm_fingerprint`, `self.movement` in `__init__`; new
  `block_size_bytes` constructor argument.
- `on_blocks_cached()` feeds the tracker **above** the identity check, so
  anonymous blocks are counted. Movement is a property of the cache, not of
  the workflow — a block rebuilt for an unnamed request cost exactly as much,
  and the baseline counts it. Skipping it here would make the two runs'
  `remat_blocks` incomparable, which defeats the point.
- `on_cache_query()` — new hook, called from `get_computed_blocks`.
- `_hash_of(block_id)` reads the hash off the block rather than taking it as
  an argument: `BlockPool` calls the evict hook *before* `reset_hash()`, so
  the hash is still there, and this keeps the hook signature stable.
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
  `hbm_usage`, `free_queue_len`, and everything in
  `CacheMovementTracker.as_dict()`.
- `_prefix_blocks: OrderedDict[NodeKey, int]` — the longest prefix ever seen
  per key, in blocks, capped at `PREFIX_SIZE_CACHE_MAX_KEYS`. Written from
  `on_blocks_cached` via `_note_prefix_size`. It has to live outside the
  ownership index because that index deletes an entry when its last block
  goes, which is exactly when the key becomes something to prefetch.
- `_update_forecast_gauges(snapshot)` — the `parsed_to_next_node_*` summary,
  gated on snapshot *identity* so a tick where nothing was published costs one
  comparison. Hoisted **above** `_maybe_log_hbm_summary` in `maybe_tick`: the
  line reports these gauges and the early returns below can skip a tick
  entirely, so computing them after would publish a budget belonging to a
  snapshot that had already been superseded.
- `_rebuild_want_list()` now runs its residency/admission walk whether or not
  origination is on, accumulating `_demand_keys` / `_demand_blocks` /
  `_demand_unsized`. Previously it returned immediately when `self._wants is
  None`, which would have left the baseline arm with no demand figure to diff
  against. `wants.is_tracked` moved *below* the gates so a want deduplicated
  against an in-flight one still counts as outstanding demand — the KV has to
  move either way.
- `_blocks_to_gb()` — same "report rather than guess" rule as `remat_mb`.
- `_demand_blocks` joins the `_maybe_log_hbm_summary` fingerprint, so a window
  where only the forecast moved still prints. The integer block count and not
  the float GB or the ttnc mean, which jitter on rounding and would defeat the
  gate.

### `vllm/v1/core/node_eviction/__init__.py`

`maybe_build_controller()` takes and forwards `block_size_bytes` and
`block_size_tokens`.

### `vllm/v1/core/kv_cache_manager.py`

- Passes `block_size_bytes` — **summed over KV cache groups**. One block id
  covers a page in every group, so the KV rebuilt when that id is evicted and
  re-cached is the sum, not any single group's page.
- Passes `block_size_tokens` — **not** summed, read off group 0. Every group
  pages the same token range, so one block id covers `block_size` tokens no
  matter how many groups exist; and the policy disables itself above one group
  anyway. Zero when there are no groups, for the same reason `remat_mb` is
  zero without a page size.
- `get_computed_blocks()` calls `node_eviction.on_cache_query()` alongside the
  existing `prefix_cache_stats.record()`.

### `tests/v1/core/node_eviction/test_controller.py`

`FakePool` gained `num_gpu_blocks` and `get_num_free_blocks()`. Eighteen
tests: pool split as an identity (`used + free == total`) rather than against
literals, attribution to `job:node`, `<untracked>` accounting, the rate limit,
the change gate, per-window reset, the off switch, token-weighted hit rate,
windowed vs. cumulative, first-time prefixes not counting as movement, the
second-eviction requirement, anonymous blocks still counting, reset not
counting, the ring bound, and the MB conversion with and without a page
size.

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
- `VLLM_HBM_REMAT_WINDOW_BLOCKS` — default `0`, i.e. size the ring to the
  pool.

The hit-rate and rematerialisation logic is a line-for-line mirror of
`CacheMovementTracker` on the policy branch. Duplicated rather than shared
because the two repos are separate clones — if you change one, change both,
and `test_line_has_the_same_fields_as_the_policy_branch` is what catches you
if you forget.

### `vllm/v1/core/block_pool.py`

- Built in `__init__` when caching is on.
- `cache_full_blocks()` → `on_blocks_cached()`, claiming the new blocks and
  testing each hash against the evicted ring.
- `_maybe_evict_cached_block()` → `on_block_evicted()`.
- `reset_prefix_cache()` → `on_reset_prefix_cache()`, which drops claims and
  clears the ring **without** counting evictions or movement. An operator
  wiping the cache did not make the policy redo work, and must not be charged
  to whichever job happened to be resident.

> **One non-obvious detail.** The evict hook sits *outside* the
> `popped is None` guard, which required restructuring that branch to keep the
> pop result. This mirrors where the policy branch releases its ownership
> claim. If the baseline only counted successful pops and the policy counted
> every call, `evicted=` would mean two different things and the comparison
> would be quietly wrong.

### `vllm/v1/core/kv_cache_manager.py`

- `configure(block_size_bytes=...)` after construction — `BlockPool` builds
  the logger but only `KVCacheManager` holds the `KVCacheConfig`. Summed over
  groups, same reasoning as the policy branch.
- `get_computed_blocks()` calls `on_cache_query()`.
- `new_step_starts()` calls `maybe_log()` — same site the policy branch ticks
  from, on the thread that owns `BlockPool`, so no locking.
- `get_hbm_summary_stats()` for a caller that wants the numbers structured
  rather than parsed back out of the line.

### `tests/v1/core/test_hbm_summary.py` (new)

Twenty-four tests, mirroring the policy branch's. The load-bearing one is
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

# Continuum clone (`/Users/vibhavsharma/Projects/vllm-continuum`) — same
# module, same env vars, `variant=continuum`.
VLLM_HBM_SUMMARY_PERIOD_MS=30000 vllm serve ... 2>&1 | tee continuum.log

grep 'kv_hbm ' policy.log    > policy.kv_hbm
grep 'kv_hbm ' baseline.log  > baseline.kv_hbm
grep 'kv_hbm ' continuum.log > continuum.kv_hbm
```

`plan/hbm-timeline.html` takes all **three** at once: it holds runs in slots
A, B and C rather than in fixed arms, and labels each column with the variant
its own lines name. Slot A is the reference — every Δ is measured against it,
so load `baseline.kv_hbm` there.

Both files are `key=value` lines with identical field order, so they parse
with a two-line split and diff directly. To visualise them over time, see
[`hbm-timeline-visualisation.md`](hbm-timeline-visualisation.md).

The headline comparison is the last line of each file:

| | baseline | policy | |
|---|---|---|---|
| `hit_rate` | did the cache work | | higher is better |
| `remat_mb` | **KV rebuilt** | | lower is better — this is the claim |
| `remat_ratio` | share of caching that was redoing | | lower is better |
| `evicted` | churn volume | | context for the above |
| `prefetch_gb` | KV the forecast still wants and HBM lacks | | lower is better — but see `prefetch_unsized` before quoting it |
| `parsed_to_next_node_min_ms` | tightest gap to move it in | | neutral; the workload sets it, and it is what makes `prefetch_gb` feasible or not |

A run where `hit_rate` is flat but `remat_mb` dropped is still a win: the same
requests were served with less KV rebuilt. A run where `remat_mb` dropped only
because `blocks_cached` dropped is **not** — check `remat_ratio`, which
normalises for exactly that.

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
- **Cumulative vs. windowed is mixed on purpose.** `splices`, `evicted`,
  `remat_blocks` and friends are cumulative counters; `top_evicted` and
  `hit_rate_win` are per window. A consumer that wants windowed rates for the
  cumulative fields takes differences between consecutive lines.
- **`remat_*` does not know where the KV came from.** A block rebuilt by a
  full prefill and one recovered from LMCache both count as one block. The
  cost difference between those is `delta_l1_ms` vs. `delta_cold_ms` in the
  scoring model, and separating them here would need the connector to report
  per-block provenance. If that matters, the decision log has `p_l1`.
- **No CPU/L1 offload traffic.** This counts KV re-entering *HBM* only. A
  design that swaps to host memory instead of dropping would show a lower
  `remat_mb` while moving just as many bytes over PCIe.
