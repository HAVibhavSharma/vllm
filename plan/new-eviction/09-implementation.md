# Implementation Status

**Written:** 2026-07-31, against branch `vllm-v2` @ `68c7acb14`.
**What this is:** the map from the design docs to the code that now exists.
Where this doc and 00–08 disagree about *intent*, they win; where they
disagree about *what the code does*, this one does.

**First run: 2026-08-03**, on `chisel-8` (Python 3.10, pytest 9.1.1).
`tests/v1/core/node_eviction/`: 104 collected, **92 passed, 12 failed**;
after fixing those, one more surfaced that the first batch had masked.
`tests/v1/metrics/test_request_stats_logger.py`: **2 passed, 1 failed**.

**Every failure was in test code. No production code was changed to make
them pass.** That matters for what the green suite is now evidence of: the
policy's logic was written correctly the first time, but four separate test
bugs — three of them arithmetically or structurally impossible assertions —
survived to this point because nothing had ever executed them.

- 11 x `TypeError: keywords must be strings` — the `fresh_rows` helper in
  `test_controller.py` took `**kwargs`, but every caller keys it by `NodeKey`
  tuples. Test-helper bug; the helper now takes the mapping positionally.
- 1 x `test_low_scoring_blocks_move_to_the_head`, which the `TypeError` above
  had been masking: its *precondition* asserted the free queue began
  `[1, 2, 3, 4, 5, 6]`. `index_prefix` only records ownership in the index —
  it does not touch the queue — so the queue was still in construction order
  with the unowned block 0 at the head. The precondition now asserts the
  property it meant (research ahead of supervisor) rather than an absolute
  list. The behaviour under test passed once reached.
- 1 x `test_ablation_changes_the_ranking` — asserted that ablating `decay`
  must change hit rate. On this fixture it cannot: under `oracle` every live
  key has `prob=1.0`, all three keys are the same size so `blocks` is
  constant, and `E_miss` is constant under the default hit-class assumption,
  so decay is the *only* varying term. Remove it and every live key scores
  identically (confirmed numerically: one value for live keys, that value
  x0.01 for dead ones), the splice ties everywhere, and Rule 3's tail-first
  tie-break selects the same blocks. The test prejudged the question the
  ablation harness exists to answer; split into
  `test_ablation_reaches_the_score` (mechanism is plumbed) and
  `test_a_uniform_workload_gives_decay_nothing_to_separate` (records the
  fixture's limit).
- 1 x `test_finished_request_stats_include_request_logger_fields` —
  `assert queued_time == 0.1` on a value that is `scheduled_ts - queued_ts`,
  i.e. `100.2 - 100.1 == 0.10000000000000853`. **Pre-existing**: the
  assertion dates to `00d4fc4c0`, before any of this work, and all four
  duration assertions in it are exact comparisons of float subtractions, so
  the test has never passed. Now `pytest.approx`. The `call_type`,
  `arrival_ts` and `finish_ts` assertions this changeset added all passed.

Note the ablation one is a real limitation of the replay fixture, not just a
bad assertion: **no ablation can separate terms on `walkthrough_trace()`**,
because it varies only one of them. A trace that can answer "which terms are
load-bearing" (09 §4 step 3) needs keys of differing size or differing
recurrence. Until then `--ablate` will report every term as inert.

The engine has still never been run end to end, and no test exercises a real
Redis, a real GPU, or a real workflow.

---

## 1. Build order (02 §10) against files

| Step | Status | Where |
|---|---|---|
| **1** Index | built | `vllm/v1/core/node_eviction/index.py` |
| **2** Poller + snapshot | built | `snapshot.py` (interface, fixture, JSON file source) |
| **3** REORDER enforcement | built | `kv_cache_utils.py` (`appendleft`, `appendleft_n`, `num_free_fresh`), `controller.py` (`_splice`) |
| **3** speculative floor | built | `scoring.py` (`speculative_floor`), `controller.py` (`_rebuild_value_table`) |
| **3** observability | built | `metrics.py`, wired from `block_pool.py` |
| **3.5** Value table + consult | built | `controller.get_value`, `KVCacheManager.get_node_value` |
| **4** Real Redis subscriber | built, **unexercised** | `redis_source.py` — the transport questions in 03 §7 are config knobs, not assumptions |
| **5** Real scoring | shape built, constants unvalidated | `scoring.py`, `config.py` |
| **6** Prefetch/admission instructions | **not built** | needs the §4 decision: want-list vs new message type vs connector API |
| **7** Demote-to-L1 | **not built** | data-gated on `p_cold` (02 §9) |

Also built, from 04 §3 (the engine-side gap that blocks the Trace Analyser):
`call_type`, `arrival_ts` and `finish_ts` now flow from `extra_args` through
`FinishedRequestStats` to the `FileStatLogger` CSV/JSONL.

The offline replay harness (08 §7) is `replay.py`, runnable as
`python -m vllm.v1.core.node_eviction.replay`.

---

## 2. Switching it on

```bash
# off (default): byte-identical to upstream, nothing in the package runs
VLLM_NODE_EVICTION_POLICY=0

# on, with no forecast: blocks are indexed and counted, nothing is ranked
VLLM_NODE_EVICTION_POLICY=1

# on, with Redis
VLLM_NODE_EVICTION_POLICY=1
VLLM_NODE_EVICTION_REDIS_URL=redis://localhost:6379/0

# tunables (all of them unvalidated defaults)
VLLM_NODE_EVICTION_CONFIG=/path/to/policy.json

# per-eviction decision log — unbounded under pressure, debugging only
VLLM_NODE_EVICTION_DECISION_LOG=/tmp/evictions.jsonl
```

`redis` is optional and not in `common.txt`:
`uv pip install -r requirements/node_eviction.txt`. With it missing the
policy logs once and ranks nothing.

---

## 3. Decisions made while implementing

These were not spelled out in 00–08 and had to be settled to write the code.

### 3.1 `appendleft_n` preserves order; it does not reverse

01 §6 Rule 3 says to splice "in ascending score order so the worst block ends
up nearest the head", and 06 §5 adds "successive `appendleft` calls reverse".
Those two statements are not compatible: repeated `appendleft` over an
ascending list leaves the *highest*-scoring of the K at the head, which is
the sign error both docs warn about.

Resolved by making `appendleft_n` a batch splice that **preserves** the
caller's order, so passing candidates ascending puts the worst block at the
head — the stated goal — in one O(K) operation. `appendleft` (singular) has
the ordinary single-block meaning. Both the docstring and
`test_appendleft_n_preserves_order` state the convention explicitly, because
getting it backwards has no symptom other than an inverted hit rate.

### 3.2 The speculative floor cannot demote a *scored* entry

02 §5 wants a falsified prediction to become the "preferred victim", but the
mechanism is `score = max(base, floor(age))` and `max` only ever raises. So:

- a speculative entry **with** a forecast row decays back to its honest base
  score (it stops being protected, but is not punished); and
- a speculative entry **without** one is driven by the floor alone, so it
  does go below every real score once the floor decays past zero.

Both are implemented as specified; the gap is in the spec, not the code. The
hard drop at `TTL x k` is what actually removes a long-falsified prediction.

### 3.3 The index is released even when the cache-map pop misses

`_maybe_evict_cached_block` has an early return when the block hash is not
found in `cached_block_hash_to_block`. The index drop is deliberately placed
*before* that return: every caller reaching it is about to hand the block to
another request or has been told the block is invalid, so keeping a claim on
it would score a stale key against contents that are about to be overwritten.

### 3.4 `num_free_fresh` lives in the queue, not the pool

01 §6 Rule 1 puts the counter in `BlockPool`. It is maintained in
`FreeKVCacheBlockQueue` instead, because that class is the only place every
insertion and removal passes through — `popleft`, `popleft_n`, `remove`,
`append`, `append_n`, `appendleft`, `appendleft_n`. Maintaining it a layer up
would mean re-deriving queue membership at each call site. `BlockPool`
exposes it as a property, so the design's read path is unchanged.

The one mutation that is *not* a queue operation is
`BlockPool.evict_blocks` stripping the hash from a block that is still
queued; `on_queued_block_hash_reset` covers it.

The counter is behind a `track_fresh` flag that only
`attach_node_eviction` sets, so with the policy off there is not even a
per-block branch in the bulk paths.

### 3.5 Payload shape is decoded permissively

03 §7 questions 3, 4 and 6 (channel naming, change-notification vs document,
one key per job vs per node) are unanswered. `decode_rows` accepts a list, a
`node_name`-keyed mapping, a single row, or a `{"job_id", "rows"}` wrapper,
and merges `PROB` and `HISTORY` fields into one row per key regardless of
arrival order. A row missing `job_id`, `node_name` or `call_type` is
**dropped rather than guessed at** — a mis-joined row is silent, and silence
is the failure this design is least able to detect.

Whether the writers `PUBLISH` at all (03 §3.1) is still the top blocker and
no amount of decoding leniency fixes it.

---

## 4. What to do next, in order

1. **Run the tests.** They have never executed. See the note at the top.
2. **Answer 03 §7 questions 1–6**, especially whether the writers `PUBLISH`
   and whether `call_type` is deterministic. The second one decides
   `use_call_type` and is unrecoverable if found out late.
3. **Capture a trace** with `VLLM_REQUEST_STATS_DIR` set and run the replay
   harness with `--compare --forecast oracle`. If the oracle arm does not
   beat LRU, the enforcement path is broken and no prediction work will help.
   Then run `--forecast history` for the realistic ceiling, and `--ablate` to
   find out which terms are load-bearing.
4. **Measure `p_cold`** per `(node, call_type)` from the same trace (04
   §5.4). It either closes the "assume L1 resident" assumption or changes the
   cost model by 60x, and it is the cheapest open question in the plan.
5. Only then tune `tau` and the splice bound.

---

## 5. Known gaps

Inside vLLM:

- **The replay fixture cannot separate scoring terms** (see the note at the
  top). `walkthrough_trace()` holds `prob`, `E_miss` and `blocks` constant,
  so `--ablate` cannot report anything but "inert" for every term. This
  blocks 09 §4 step 3 and is now the largest gap in the harness.
- **No end-to-end run.** The unit tests pass; the engine has never been
  started with the policy on, against a real Redis and a real workflow.
- **No Prometheus export.** Counters exist and are readable via
  `KVCacheManager.get_node_eviction_stats()`, but nothing publishes them.
  There are no Prometheus metrics anywhere in this fork (01 §8).
- **The value table has no consumer.** `get_node_value` is the O(1) read
  path from 02 §2 and nothing in the scheduler calls it yet; that is step 6.
- **Prefetch origination** (02 §4) is untouched, so the speculative path is
  only exercised by phantom requests the existing submitter already sends.
- **The timing memo** (02 §6, and named in step 5) is not built. The scorer
  reads `HISTORY` from Redis instead, which covers the same need for v1;
  the memo only matters if the engine should keep its own cross-job timing
  belief. Note 02 §6 and 04 §6 disagree about who owns that.
- **Splice cost never measured.** 01 §8 asks for a bench of K=256
  `remove` + `appendleft` against a ~24k-block queue, versus a 10–50ms
  scheduler step. Not done.
- **No benchmark.** 01 §8 wants an undersized cache, a stated baseline, and
  the noisy-forecast arm the old caveats doc called for but never ran.
- **Rule 1 boundary is uninstrumented.** 01 §6 leaves open that at the
  `num_free_fresh` threshold, up to K cached blocks can still be spliced
  ahead of a few fresh ones, and asks for a metric. There isn't one.

Outside vLLM — both components now exist as separate installable packages,
and unlike the engine-side work **their tests have been run**:

| Component | Spec | Where | Tests |
|---|---|---|---|
| Trace Analyser | 04 | `~/Projects/kv-trace-analyser` | 23 passing |
| Prediction Engine | 05 | `~/Projects/kv-prediction-engine` | 25 passing |

Both are framework-agnostic and depend on nothing but an optional `redis`.
The end-to-end contract was checked directly: a captured JSONL trace through
the analyser produces a `HISTORY` document the prediction engine consumes,
whose `PROB` output merges with it under vLLM's own decoder, and the
`extra_args` from `PredictionEngine.call_context()` joins to that merged row.

Two bugs the prediction-engine tests caught, both of which would have been
silent in production:

- The publish-on-change fingerprint included `update_ts`, so every forecast
  looked like a change and the engine would have rebuilt its snapshot on
  every workflow step.
- `time_to_next_call` was the cost of the shortest path, which gives a react
  self-loop and its exit node **the same** value — erasing exactly the
  separation this design exists to exploit (06 §3.3). Replaced with an
  expected hitting time in the Markov chain.

Still open on that side: LangGraph's adapter is untested against a real
compiled graph, and `prediction_headroom` is published as the raw capped gap
rather than the gap minus warming cost.
