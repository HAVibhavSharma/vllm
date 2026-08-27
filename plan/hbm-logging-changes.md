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
  query_tokens=12388291 external_hit_tokens=1634783 cold_tokens=1571204
  remat_blocks=4118 remat_mb=8236.0 remat_ratio=0.1902
  blocks_cached=21650 index_keys=112 index_blocks=9130
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
| `ttft_n` | TTFT samples this epoch, i.e. requests that produced a token. The latencies themselves are on their own `kv_hbm_ttft` lines (§2.1.0) | same |
| `ttft_win_n` | Same, this window only | same |
| `hit_tokens` / `query_tokens` | The raw numerator and denominator | same |
| `external_hit_tokens` | Of the missed tokens, those LMCache served over the connector (§2.1.1) | same |
| `cold_tokens` | Of the missed tokens, those LMCache could not serve either — the only ones that cost a real prefill (§2.2) | same |
| `phantom_hit_rate` | Hit rate of prefetch phantoms only, excluded from every rate above | always `0.0000` |
| `phantom_query_tokens` | Tokens looked up by phantoms — the prefill origination bought | always `0` |
| `remat_blocks` | **Movement.** Blocks cached again after being evicted | same |
| `remat_mb` | The same in MB of KV rebuilt | same |
| `remat_ratio` | `remat_blocks / blocks_cached` | same |
| `blocks_cached` | Every block ever given a hash | same |
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

### 2.1.1 `external_hit_tokens` / `cold_tokens` — splitting the misses

`query_tokens - hit_tokens` is not prefill volume, and reading it as one is
what makes the miss count look alarming. It counts every token HBM did not
hold, and HBM is not the only tier: a prefix LMCache holds is fetched over
the connector as an async load, not a forward pass. On the 2026-08-09 run
that was ~51% of them (`new-eviction/12-first-run-findings.md` §1), so half
the apparent misses were never prefills at all.

The two fields split that miss count in one subtraction:

```
missed   = max(num_tokens - num_local_hits, 0)
external = min(max(num_external_hits, 0), missed)   # -> external_hit_tokens
cold     = missed - external                        # -> cold_tokens
```

What survives as `cold_tokens` is the only quantity that reached the model:
tokens the workload had never sent, or had sent under a prefix that has since
diverged. Read the four fields as a cascade — `query_tokens` asked,
`hit_tokens` came from HBM, `external_hit_tokens` came from LMCache,
`cold_tokens` was prefilled — and by construction the last three sum to
`query_tokens`.

**Why the external half is reported and not derived.** `query - hit - cold`
looks like it already gives it, and does until the clamp fires; the moment a
connector reports a span overlapping the local hit, that arithmetic and the
counter disagree, and the arithmetic is the one that is wrong. Emitting it
also means a reader never has to know `hit_tokens` and the local half here
are gated on two different flags at two different points.

**Absolutes, not a second hit rate.** The external tier is not something
this eviction policy governs, so a rate would invite a comparison against
`hit_rate` that means nothing, and folding external hits into `hit_rate`
itself would destroy the baseline diff the line exists for — an external hit
*is* an HBM miss, just a cheap one. It would also double-count under
prefetch, since a phantom's LMCache load is committed into APC by
`_finalize_prefetch_only_request` and scored again as a local hit when the
real request lands.

**Counted in the scheduler, not the manager.** The two halves resolve at
different points: the local hit inside `KVCacheManager.get_computed_blocks`,
the external one only after `connector.get_num_new_matched_tokens` replies,
several branches later in `Scheduler.schedule`. Gated once per request on
`Request.cold_tokens_counted` — a second flag, because `get_computed_blocks`
sets `cache_query_counted` before the connector is consulted — for the same
queueing-delay reason as §2.1. Phantoms are excluded, like everywhere else.
Clamped at zero: the tiers are counted at different moments, so a connector
reporting a span that overlaps the local hit must not drive it negative — the
clamp charges that overlap to `external_hit_tokens`, which is the tier whose
report was optimistic.

With no connector configured, `external_hit_tokens == 0` and
`cold_tokens == query_tokens - hit_tokens` exactly, which is the correct
degenerate case.

### 2.1.0 `kv_hbm_ttft` — what the miss cost, one line per request

Hit rate says how often the cache worked. It cannot say whether that
mattered: a policy can raise hit rate and still lose on latency if the blocks
it kept were cheap to rebuild and the ones it dropped were not. `remat_mb`
says how much KV was rebuilt; this says what it cost.

**Per request, not aggregated.** Every request that produces a token gets its
own line:

```
kv_hbm_ttft variant=node_eviction epoch=2 req=chatcmpl-7f2 ttft_ms=412.7 \
  query_tokens=1536 hit_tokens=1024 external_hit_tokens=0 cold_tokens=512 \
  preempted=0 phantom=0
```

| Field | Meaning |
|---|---|
| `variant`, `epoch` | Which arm and which measurement epoch — the same two the `kv_hbm` line carries, so samples can be split on the warmup boundary after the fact |
| `req` | `Request.request_id` |
| `ttft_ms` | This request's engine-side TTFT |
| `query_tokens` | Prompt tokens looked up, i.e. this sample's denominator, captured at lookup time |
| `hit_tokens` | Of those, served from HBM |
| `external_hit_tokens` | Of those, served by the connector (§2.1.1) |
| `cold_tokens` | The rest — the tokens that actually cost a prefill |
| `preempted` | `Request.num_preemptions`; a nonzero one paid for scheduling pressure, not a cache miss |
| `phantom` | 1 for a prefetch phantom, excluded from every rate on the `kv_hbm` line but not from this one (§2.1.1) |

The breakdown travels with the latency because the latency alone cannot be
read: 412 ms is a fast cold prefill or a slow warm one depending entirely on
how much of the prompt had to be computed, and separating those two is the
whole question. A request that never reached `get_computed_blocks` has no
breakdown and reports `-1` rather than `0`, so it cannot be mistaken for a
genuine all-cold prefill.

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
nothing rather than a zero. A backwards wall clock would produce a negative
latency; that sample is dropped and no line is written, because a negative
latency in the log is worse than a gap in it.

**Why no mean, median or percentile on the `kv_hbm` line.** They used to be
there — `ttft_ms`, `ttft_win_ms`, `ttft_p50_ms`, `ttft_p95_ms`, the last two
over a 4096-sample ring — and they were removed in favour of the raw samples.
An aggregate computed in-process is fixed at write time to a window nobody
chose and to a population nobody selected. TTFT is heavy-tailed enough that
this routinely matters: nine 100 ms prefills and one 30 s cold miss average to
3.09 s, which describes none of the ten. With the samples in the log, the
mean, the median, the p95 and anything else can all be computed afterwards —
and computed over exactly the requests being asked about: warm only, one job
only, cold-token-count above some threshold, phantoms excluded. No summary
fixed at write time can be re-cut that way.

What stays on the `kv_hbm` line is `ttft_n` (and `ttft_win_n`), because every
rate on that line is unreadable without knowing how many requests produced it,
and because a window in which only latency moved still has to get past the
line's change gate.

**Cost.** One INFO line per request instead of one per window, which is real
under load. `NodeEvictionConfig.ttft_per_request_log` turns it off; `ttft_n`
does not depend on the switch.

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

### `vllm/v1/core/node_eviction/__init__.py`

`maybe_build_controller()` takes and forwards `block_size_bytes`.

### `vllm/v1/core/kv_cache_manager.py`

- Passes `block_size_bytes` — **summed over KV cache groups**. One block id
  covers a page in every group, so the KV rebuilt when that id is evicted and
  re-cached is the sum, not any single group's page.
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
