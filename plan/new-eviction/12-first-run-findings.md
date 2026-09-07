# First Real Run — What Broke, and the Four Fixes

> **Amended 2026-09-07.** Engine-side prefetch origination (step 6) has been
> removed: `wantlist.py`, `agent_prefetch/drain.py`, `drain_prefetch_wants`
> and `VLLM_NODE_EVICTION_PREFETCH_DRAIN` are gone, along with the
> `prefetch_*` want config and the `prefetch_wants_*` counters. The eviction
> policy itself — index, scoring, tick, splice, speculative stamping — is
> unchanged. Phantoms now arrive only from a client calling
> `POST /v1/agents/prefetch`, so the engine reports what landed
> (`speculative_*`, `speculative_waste`) but never what was asked for.
> Anything below describing a want-list is history.

**What this is:** the measured result of the first end-to-end run of the
node-aware policy against a real workload (2026-08-07, Qwen2.5-72B-AWQ,
LMCacheMPConnector, 228 requests replayed identically on both arms), and the
changes made in response.

**The headline:** the policy lost to the LRU baseline it replaced — but not
because the scoring model is wrong. The model works when it is fed a real
forecast; it almost never was. 98.6% of eviction decisions ran on extreme
values from the prediction engine rather than a graded probability, and the
splice churned the free queue hard enough to destroy LRU's recency order on
top of that.

**Correction, 2026-08-08.** An earlier pass of this analysis reported the
scores as *anti-correlated* with outcome (81.6% vs 65.7%). That was an
artifact: the "needed again" proxy counted same-millisecond duplicate
records, and half of all matches had a 0.0s gap. Requiring a >1s gap flips
the sign — the scores are **correctly correlated** (26.1% for the bottom
three deciles vs 49.3% for the top three). The numbers below are the
corrected ones.

---

## 1. What was measured

| | policy | baseline (LRU) |
|---|---|---|
| HBM prefix hit rate *(vLLM's own metric)* | **13.3%** | **18.8%** |
| KV rebuilt (`remat_mb`) | **125.8 MB** | **22.5 MB** |
| regret | 0.325 | — |
| evictions | 73,134 | 67,980 |
| blocks relocated by the splice | 2,054,400 | 0 |
| end-to-end latency (456 paired requests) | +1.4% | — |

Same trace on both arms: the first six windows match to four decimal places.
External (LMCache) hit rate was 51.1% vs 50.3% — unaffected, as expected.

### The decisive number

From the decision log, bucketing scored evictions by **forecast class** and
asking whether the same `(key, prefix_pos)` was evicted again more than 1s
later within the 120s regret window — which can only happen if it was
re-cached in between, i.e. the eviction was undone:

| forecast class | n | came back |
|---|---|---|
| A `prob=0.01, ttnc=1h` — `end_job()` floors | 42,125 | 24.3% |
| B `prob=1.0, ttnc=60s` — reach saturated | 16,150 | 54.7% |
| C other floored prob | 3,353 | 24.4% |
| D other saturated prob | 9,674 | 37.2% |
| **E real fractional forecast** | **1,033** | **0.3%** |

**When the forecast is real, the policy is nearly perfect — and it is real
1.4% of the time.** Classes A–D are not noise either: A/C mark genuinely dead
keys and B marks live ones, so the signal is a working *binary* classifier.
What it is missing is gradation, and gradation is what the ranking needs.

---

## 2. Why — the forecast never arrived, and the score degenerated

Terms recorded on the 72,335 scored evictions:

```
prob=0.01  ttnc=3600s   58.2%   ← predictor.floors(), published by end_job()
prob=1.0   ttnc=60s     22.3%   ← reach_probabilities() clamped at min(1.0, ...)
prob=0.01  ttnc=60s      4.6%
  …real fractional probabilities  ~4%
e_miss_ms=180 (delta_l1 default)  79.2%
```

Both extremes trace to real code paths in `kv-prediction-engine`, not to a
missing publisher:

- `prob=0.01, ttnc=3_600_000` is exactly `floor_probability` and
  `job_end_ttnc_ms` — `Predictor.floors()`, published by `end_job()`.
- `prob=1.0` is `reach_probabilities()` saturating. It computes
  `min(1, sum over h in 1..H of (M^h)[current][X])` with `max_hops=6`. On the
  9-node ODR graph the raw sums come out 0.5 / 1.0 / 1.5 / 2.0, so **4 of 9
  nodes clamp to 1.0 and 4 sit at the floor — one is fractional.** The
  accumulate-over-hops choice is right for react self-loops (a path product
  would decay the hottest node toward zero); the `min(1.0, ...)` is what
  destroys the gradation.

**98.6% of eviction decisions therefore ran on one of two extremes.**

That matters because of what `score = prob × decay × E_miss / blocks` does
when its numerator is constant. Verified across the largest homogeneous group
(29,363 evictions):

```
blocks=4846  score=0.00000307   score×blocks=0.014876
blocks=4845  score=0.00000307   score×blocks=0.014876
   distinct values of score×blocks in the group: 1
```

Exactly proportional to `1/blocks`. With the numerator pinned, prefix length
is the **only** discriminator left, so the policy reduced to *"evict from
whichever key owns the most blocks"* — which for prefix caching is close to
the worst available rule. `researcher_tools`, holding up to 7,500 of 9,558
blocks, always scored lowest, always filled every splice slot, and was
destroyed within seconds of each rebuild.

The `/blocks` term is not itself wrong (08 §2's tail-first argument holds). It
only becomes destructive when the numerator carries no information.

### The splice compounded it

```
splices          8,025    every one moved exactly 256.0 blocks — saturated all run
spliced_blocks 2,054,400
evicted           73,134  →  28 relocations per eviction
of those, spliced blocks that were actually evicted: 47,363  →  43 moves each
pool               9,558  →  the whole pool reordered 215×
```

`splice_max_blocks=256` at `tick_period_ms=250` moved 256 blocks to the head
1.6×/second while eviction consumed only ~9 per tick. Each tick pulled a fresh
worst-K to the front and pushed the previous one back; the queue never
settled. LRU's recency ordering was destroyed and nothing coherent replaced
it.

---

## 3. The fixes

### 3.1 `is_uninformative()` — added, then defaulted **off** (`scoring.py`)

Written to treat a sentinel row as the absent row it appears to be, gated by
`uninformative_prob_at_or_below` and `uninformative_ttnc_at_or_above_ms`.

**It is disabled by default, because the corrected measurement did not
support it.** The class it targets — `end_job()` floors — has a 24.3%
come-back rate, the *best* of the four degenerate classes. Those floors are
the engine correctly saying a finished job's blocks are free to go, and
gating them out would discard the one thing the forecast currently does well.

Kept, disabled, for the case it was written for: a source publishing a true
blackout (every key floored because it has no idea, not because jobs ended).
`validate()` rejects a prob gate above 1.0, which would make every key
unscored and the policy silently inert.

### 3.2 The splice no longer re-moves blocks it already moved (`controller.py`)

`_splice` skips any block already in `_spliced_scores` — already parked at the
head, untouched, nowhere better to go. Simulated over three ticks on a
three-block key: 9 relocations before, 3 after.

### 3.3 A cache hit clears the splice bookkeeping (`controller.py`)

Required for 3.2 to be correct. A touched block leaves the free queue, so
whatever position the splice gave it is gone; without dropping the record, a
block that was hit and later freed would be skipped forever. `on_prefix_hit`
now pops `_spliced_scores` / `_spliced_ranks` alongside `confirm_block`.

### 3.4 `splice_max_blocks` 256 → 32 (`node_eviction.json`)

Sized near the ~9 evictions/tick actually observed rather than 28× above it.

---

## 4. What is still open

- **`reach_probabilities()` saturates — this is the highest-value fix, and it
  is in `kv-prediction-engine`, not here.** `min(1, sum_h M^h)` at
  `max_hops=6` yields a binary signal on a 9-node graph. Either drop
  `max_hops` to 2–3 (config only; restores 3–4 fractional nodes of 9) or
  replace the quantity with a first-passage probability — make the target
  absorbing and read the absorbed mass, which is a true probability in [0,1]
  and cannot clamp. Simulated on the ODR graph the latter gives 0.375 / 0.5 /
  0.75 / 0.875 / 1.0 where the current formula gives 1.0 / 1.0 / 1.0 / 0.01.
  Class E says the payoff is large: 0.3% come-back against 24–55%.
- **`score_threshold` is still `None`** (unconditional worst-K, 08 §6).
- ~~**Prefetch never executed.**~~ Diagnosed and fixed — see §5.
- **The measurement regime is wrong for the question.** `--max-num-seqs 1`
  meant `max running: 1` throughout, and LMCache absorbed ~50% of traffic at
  `delta_l1=180ms`, so an HBM miss is cheap and end-to-end latency was a wash
  (+1.4%). Re-run with real concurrency before drawing conclusions either way.
- **`GPU KV cache usage` reads ~4%** while `index_blocks` is 9,520/9,558.
  vLLM's usage counts referenced blocks only; the prefix cache was full the
  whole time. Do not read low usage as "no eviction pressure".

---

## 5. Why prefetch never executed (2026-08-08 run)

Three independent reasons, all of which had to be false at once for a single
phantom to fire. From the 2026-08-08 log: `wants created=96 drained=96`,
`speculative keys=0 blocks=0`, 96 lines of `no registered prefixes yet`.

### 5.1 The registry was never written — the structural blocker

`AgentPrefixRegistry` was only written by `POST /v1/agents/chat/completions`
and `POST /v1/agents/prefetch`. The workload posted 228 requests to plain
`/v1/chat/completions` and zero to `/v1/agents/*`, so every want resolved to
zero descriptors and was dropped in `drain.py::_submit_want`.

The identity the policy keys on already rides in `vllm_xargs` on plain chat
requests — that is how engine core learns it. So the plain path always
carried what was needed and simply never recorded it.
`agent_prefetch/auto_register.py` now records the chunk-aligned prefix from
`OpenAIServingChat.create_chat_completion` whenever the request carries
`langgraph_node` and a registry exists. No client change.

The recorded prefix is hit-compatible by construction: the phantom passes
`cache_salt` in `kv_transfer_params` (an LMCache input), while vLLM's own APC
block hashing keys off `request.cache_salt` (`kv_cache_utils.py:645`), which
a phantom never sets. Same tokens ⇒ same block hashes ⇒ a later real request
hits them. Recording under the request's own salt rather than a synthetic
`agent::` one keeps that true on the LMCache side as well.

### 5.2 The residency test was all-or-nothing

`_rebuild_want_list` skipped any key with `index.get_entry(key) is not None`,
and `index.py:133` only deletes an entry when its **last** block goes. A key
whose prefix had been 99% evicted still read as resident. With `index_keys`
at 8–11 across a 9-node graph this gated out nearly everything — 96 wants in
74 minutes.

Replaced with `NodeEvictionController._is_resident`, a coverage ratio
`num_blocks / run_len` against `prefetch_min_coverage` (default 1.0: anything
short of fully resident is wanted). `run_len` derives from `max_position`,
which only grows, so partial eviction genuinely drives the ratio down.
Setting the threshold to 0 restores the old test for an A/B.

### 5.3 The prob and horizon gates split on a binary signal

`prefetch_min_prob=0.5` and `prefetch_horizon_ms=60s` were sized for a graded
forecast. Per §2 the forecast is 0.01 or 1.0 with ~4% between, and ttnc is
60s or 3600s — so both gates were selecting one arm of a broken binary rather
than ranking. Both now default off (0.0 = disabled) and stay as knobs for
when `reach_probabilities()` is fixed (§4). `prefetch_ignore_staleness`
defaults true, asymmetrically with the eviction half: a stale row there
destroys a block, here it costs at worst one redundant prefill.

### 5.4 Bounds re-sized for the new volume

`prefetch_max_outstanding` 8 → 64 and `prefetch_max_per_drain` 4 → 32 (both
were sized behind the gates), `prefetch_resubmit_backoff_ms` 60s → 10s, and a
new `prefetch_top_k_per_want` (default 1) caps the front-end fan-out — the
drainer previously used `get_all()`, every prefix ever recorded for the
agent, which was safe only while want volume was near zero. Registry growth
is now capped at 32 descriptors per agent, since ordinary chat traffic writes
one per turn.

### 5.5 The speculative floor no longer decays

The floor started at `speculative_floor_high` and decayed to
`speculative_floor_low = -1.0` over the prediction's own deadline, so a
falsified prediction became the *preferred* victim rather than merely losing
protection. Sound in principle; unusable against this forecast.

The deadline was `time_to_next_call_ms`, which is 60s or 3600s and nothing
between (§2). So the decay never tracked a horizon — it read one bit of the
broken forecast. On the 60s arm it was a countdown that put a prefetched
prefix *below every real score* at approximately the moment its predicted
call was due: the policy paid a prefill, protected the result for 60s, then
made it first-out just before the payoff. On the 3600s arm it protected for
an hour.

`scoring.speculative_floor(config)` is now a constant, and takes no clock —
asserted on the signature so the decay cannot be reintroduced silently.
`speculative_floor_low`, `speculative_default_ttl_ms` and
`speculative_hard_drop_ttl_multiple` are deleted. `BlockOwnershipIndex.gc`
lost its second clock with them: it ran on the same
`time_to_next_call_ms` TTL, so keeping it would have been the same expiry
under another name, deleting an entry the scorer was still holding at the
floor. Speculative and confirmed entries now age out together on
`index_hard_drop_age_ms`.

**What this trades away.** Nothing demotes a wrong prediction any more. An
unconfirmed entry holds the floor until the index GC drops it at 30 minutes,
and while it holds the floor the splice picks *real* blocks in preference to
it. Confirmation (`on_prefix_hit`) is the only release. It stays a soft pin —
a permanently high score only means "never volunteered by the splice"; the
block keeps its LRU position and ordinary `popleft` can still take it, so
this cannot strand a block the way a `ref_cnt` pin with a lost release would.
If the forecast is wrong often, the cost now shows up as real blocks evicted
to protect useless ones, and `waste=` is where it will be visible.

**That trade came due — see §6.** The floor is off by default as of
2026-08-09; the rest of this section describes the behaviour you get by
setting `speculative_floor_high` back above `delta_cold_ms`.

### 5.6 The staged region was two blocks deep

The splice never blocks an eviction — `get_new_blocks` stays an unmodified
`popleft_n`. It reorders the queue that is popped from, so **the policy
governs exactly the prefix of the free queue it has managed to stage**, and
everything past that is evicted in raw LRU order with the scores having no
say at all.

`splice_max_blocks=32` (§3.4) against ~30 evictions per tick — measured over
one 30s window of the 2026-08-08 log: 23 ticks, 734 blocks staged, 702
evictions — left a margin of two blocks. A request needing up to
`max_model_len / block_size` = 7,500 blocks pops through that in one
allocation and takes the remainder by age. Every conclusion drawn about the
scoring function was therefore drawn from a policy that was in charge of a
rounding error's worth of the queue.

`splice_max_blocks` is now a **catch-up rate limit rather than a fixed K**,
defaulted to 4096. This is safe now and was not before: the 2,054,400
relocations at K=256 came from the *absence* of the already-staged skip
(§3.2), which made every tick re-move the same blocks 43 times each. With the
skip, the candidate set holds only blocks that are free, scored and not yet
staged, so steady-state work is bounded by the newly-freed count (~30/tick)
and the cap binds only when there is a real deficit — which is exactly the
burst it exists to absorb.

Depth exposed a second-order bug worth naming: `appendleft_n` puts each
tick's batch *in front of* the previous one, so the head is a stack of
batches — newest first, each internally sorted, globally unsorted. At two
deep that is invisible; at thousands it inverts the policy, because a block
freed this tick lands ahead of a worthless one staged a minute ago. New
`splice_restage_period_ms` (default 5s) clears the staging record so one pass
re-sorts the whole region by current score, which also picks up forecast
changes since a block was staged. 0 restores the batch-stack behaviour.

Two new gauges on the `kv_hbm` line: **`staged`** — how deep the policy's
authority reaches into the free queue — and **`deficit`** — blocks it wanted
to rank and the cap could not take. A sustained non-zero `deficit` means
window 1 is still open.

**Untested at the time of writing.** The next run is the first that can
produce a non-zero `speculative blocks`, and the number to watch is `waste=`
on the prefetch line: with `--max-num-seqs 1` a phantom that misses in
LMCache is a full prefill serialized ahead of real traffic. Read it against
`staged`/`deficit` — a good forecast that governs 2 blocks of queue and a bad
forecast that governs 9,000 look identical in `hit_rate` alone.

**That last sentence stopped being true on 2026-08-10 — see §7.** A phantom
that misses in LMCache is no longer a prefill at all; it is finished without
being scheduled.

---

## 6. The speculative floor is off by default (2026-08-09)

`speculative_floor_high` now defaults to **0.0**, which means no floor at
all. §5.5 removed the floor's decay and named what that traded away —
"nothing demotes a wrong prediction any more" — and the next run showed the
bill.

### 6.1 What was observed

| Metric | baseline (LRU) | policy |
|---|---|---|
| Total prompt (prefill) tokens | ~275,700 | ~440,100 |

Regret was high on the same run. **Both numbers need qualifying before they
are used as evidence**, and neither qualification changes the decision:

- The prefill figure is `num_prompt_tokens`, which counts cached tokens too
  (`stats.py`, `PrefillStats.set`). It is a sum over *requests*, and the
  policy arm submits phantom prefetches as additional requests, so a large
  part of the delta is phantom prompt length rather than extra compute. The
  number that settles it is `num_computed_tokens` split by the
  `prefetch_only` column in the `FileStatLogger` CSV. That split had not been
  taken when this change was made.
- The regret figure is inflated by a **measurement loop that is still open**
  (§6.4).

The decision rests on the mechanism below rather than on either figure.

### 6.2 Why the floor was the wrong shape

Two properties compounded:

**Magnitude.** `1e9` against a score range bounded above by
`delta_cold_ms = 11_400` (invariant 6) is ~87,000x the top of the honest
range. That is not a tie-break in favour of predictions; it is a total order
with every prediction above every observation.

**Reach.** `_splice` takes `max` over a block's owning keys (01 §4), so a
block co-owned by a speculative key and a real one inherits the floor. A
phantom whose prefix starts with the shared system preamble therefore floors
that preamble for *every* node that shares it. The protected set is much
larger than the prefetched set, and grows with exactly the prefix sharing the
policy exists to exploit.

Together: the policy's evictions concentrate on real, confirmed blocks —
protection for predictions, paid for out of observations. With the forecast
still bimodal (~58% of rows at the `prob=0.01` floor, §2) and nothing
demoting a wrong prediction, the waste compounds instead of self-correcting.

### 6.3 What replaces it

Nothing. A speculative key is scored like any other key:

- It has a forecast row → normal `score = prob · decay · E_miss / blocks`.
  Guaranteed to exist for anything the want-list originated, since
  `_rebuild_want_list` iterates the snapshot's own rows.
- It has no row → absent from the value table, unscored, keeps its LRU
  position (Rule 2). Identical to how a *real* rowless key is treated. This
  is a change: the floor path used to manufacture a table row with a
  synthetic `score=0.0` base for these.

Worked example, both at 1 block: a phantom at `prob=0.01, ttnc=30s` scores
**0.9**; a real key at `prob=0.9, ttnc=1s` scores **156.8**. The phantom is
the first victim where before it was the last.

**Provenance is untouched.** `entry.speculative` is still stamped at insert,
`on_prefix_hit` still confirms, and `on_block_evicted` reads provenance from
the index rather than from the score breakdown — so `speculative_waste`
survives the floor's removal and is the number to watch, exactly as §5.5
said.

**A/B.** Set `speculative_floor_high` above `delta_cold_ms` (1e9 was the old
default) to restore the protected behaviour. The validator rejects a value
strictly between 0 and `delta_cold_ms`: such a floor protects some keys and
not others, which is a silent partial policy rather than a switch.

### 6.4 Still open — the regret counter measures its own prefetches

`on_prefix_hit` calls `observer.record_admission(key)` **above** the
`_is_prefetch_only` early return, so a phantom prefetch registers as demand
for its own key. The prefetcher only wants keys that are not resident — i.e.
keys we just evicted from — which closes a loop:

```
evict a block of K  →  coverage drops below prefetch_min_coverage
                    →  want K  →  phantom for K  →  partial prefix hit
                    →  record_admission(K)  →  that eviction is regret
```

Moving the `record_admission` call below the phantom gate is a one-line fix.
It is *not* in this change, so any regret figure measured before it lands is
an upper bound. `prefetch_min_coverage=1.0` makes the loop maximally tight:
the splice is deliberately tail-first (Rule 3), and shaving one block off a
900-block prefix is enough to drop coverage below 1.0.

### 6.5 What to watch next

A fresh prefetch is now the **largest denominator** in the density
`score = … / num_blocks`, so a just-landed 900-block prefix ranks low even
when the forecast that fetched it was strong. If prefetched prefixes start
being evicted before their predicted call, the fix is to score *unconfirmed*
entries with `num_blocks = 1` — the undivided `prob · decay · E_miss` form
that `_rebuild_want_list` already uses for the fetch decision, on the
reasoning that a big prefix is more worth holding, not less — rather than
reinstating a sentinel.

Two gates that would cut phantom volume at the source remain off and are
worth trying before anything else here: `prefetch_min_prob` (0.0) and
`prefetch_horizon_ms` (0.0), both disabled in §5.3 on the grounds that a
binary signal cannot *rank*. It cannot — but it can *admit*, and the arm it
admits (`prob=1.0`, ~22% of rows) is the one the forecast is confident about.
On the 3600s arm the discount is `30_000/3_630_000 ≈ 0.008`, so those wants
are warming prefixes due in an hour by evicting blocks due in a minute.

---

## 7. A phantom that misses LMCache is no longer prefilled (2026-08-10)

§5.6 named the cost and §6.1 measured it: prompt tokens roughly doubled
against the LRU baseline, and a large share of that is phantom traffic.
§6.1 also said the split by the `prefetch_only` column had not been taken —
this change removes the largest term without needing it.

### 7.1 What the fallback was

`get_num_new_matched_tokens` returns `(0, False)` when LMCache holds nothing
for the prefix. Nothing downstream distinguished a phantom from a real
request at that point, so the phantom was admitted like any other: full
prefill of the phantom prompt, STORE back to LMCache, sample one token,
finish. `lmcache_mp_connector.py` documented it as the "LMCache miss →
normal prefill path".

**Why that is the wrong trade.** A phantom is a *promotion* of KV from L1
into HBM, priced at `delta_l1_ms = 180`. If L1 does not have the prefix there
is nothing to promote, and what actually ran was `delta_cold_ms = 11_400` of
prefill on a prefix no client had asked for — 63x the cost the mechanism was
designed around, and at `--max-num-seqs 1` serialized *ahead* of real
traffic. The real request that eventually wants that prefix pays the same
prefill anyway, and pays it only if the prediction was right. The fallback
converted a wrong prediction from "no benefit" into "a full prefill of
latency charged to whoever was queued behind it".

### 7.2 The change

`Scheduler.schedule()`, immediately after `num_external_computed_tokens =
ext_tokens`: if the request is `prefetch_only` and `ext_tokens == 0`, pop it
from the queue and defer it to `self._prefetch_only_misses`.

Three details that are load-bearing:

- **Below the `ext_tokens is None` branch, not above it.** `None` means "the
  connector cannot say yet, ask again", not "miss". Aborting there would kill
  every phantom whose LMCache lookup had not resolved on its first poll —
  i.e. most of them.
- **Deferred, not finished in place.** `schedule()` has no `outputs` dict.
  `_finish_prefetch_only_misses()` drains the list from
  `update_from_output()`, the same deferral `failed_kv_load_req_ids` already
  uses.
- **Finished as `FINISHED_STOPPED` with `FinishReason.STOP`** — the identical
  shape `_finalize_prefetch_only_request` emits on the happy path. That
  matters twice over: `output_processor` records the row under
  `prefetch_only=True` like any other phantom, and the submitter's `async
  for` in `_run_one` only exits on a terminal output. Omitting it strands the
  generator and leaks the agent's `max_inflight_per_agent` slot (64), which
  would silently throttle prefetch for that agent for the rest of the run.

`finish_requests` → `_free_request` fires the connector's `request_finished`
hook, so the LMCache request tracker and session are torn down rather than
leaked. No blocks were ever allocated; `single_type_kv_cache_manager.free`
already handles that (`req_to_blocks.pop(request_id, [])`, commented "in case
a request is freed (aborted) before alloc").

### 7.3 What it trades away

**A prefix LMCache does not hold can no longer be warmed at all.** Prefetch
becomes strictly an L1→HBM promotion. Where the forecast is right *and* the
prefix is cold in LMCache, the real request now eats the prefill it would
previously have found already done — a TTFT regression on that subset,
bought against not paying for every wrong prediction.

This is the right side of the trade only while forecast precision is low.
If `speculative_waste` drops far enough that most phantoms are confirmed,
warming cold prefixes becomes worth paying for again and this should be
revisited — as an explicit opt-in, not a silent fallback.

### 7.4 Visibility

The path was silent, which is why it survived so long. Now:

- `Scheduler._prefetch_only_misses_total` — cumulative count.
- `warning_once` on the first occurrence, per-request `debug` after.

Read it against `speculative_confirmed`. A high miss count with low
confirmation means the forecast is naming prefixes LMCache never stored —
a prefetch-side problem, not an eviction-side one, and no amount of scoring
work will fix it.

`LMCACHE_MP_FULL_HIT_ONLY=1` composes: the connector then reports partial
prefix hits as 0 matched tokens, so partial hits abort too instead of
prefilling the remainder. No new knob was added for that.

### 7.5 Tests

`tests/v1/core/test_scheduler.py`, using the existing
`mock_kv(matched_tokens=…)` harness:

- `test_phantom_prefetch_missing_in_lmcache_is_not_prefilled`
- `test_phantom_prefetch_with_an_lmcache_hit_still_runs`
- `test_a_real_request_missing_in_lmcache_is_unaffected` — the abort keys on
  `prefetch_only`; an ordinary request that misses must still prefill.

These are the first tests of any kind on the scheduler's `prefetch_only`
path.
