# Observability — Detecting a Wrong Forecast

**Status:** spec, nothing implemented.
**Written:** 2026-07-31, against branch `vllm-v2` @ `68c7acb14`.
**Purpose:** answer three questions with data — *is the policy helping*, *when
it evicts wrongly why*, and *is it worse than doing nothing*.
**Reads with:** [`06-walkthrough.md`](06-walkthrough.md) §7, whose last caveat
this closes; [`01-redis-integration.md`](01-redis-integration.md) §8, where the
missing-metrics blocker is filed.

`[verified]` = read at the cited line. `[decided]` = settled. `[open]` = needs an
answer.

---

## 1. The principle

> **Ground truth is free, one frame later.** Nothing needs to be predicted or
> sampled. When a request arrives and hits or misses the prefix cache, the
> engine learns whether the block it evicted was actually needed.

This is the same observation that retired the old branch's trigger gate:
`_maybe_evict_cached_block` (`block_pool.py:354`) knows per block whether the
victim was cached, and `touch` (`block_pool.py:391`) knows when a prefix is
matched again (00 Part 2, weakness 3). `[verified]` Both are already on the
path; only the recording is missing.

Every metric below is a **join between a decision and its outcome.** Nothing
here requires new measurement — only that decisions be written down.

---

## 2. Why this is not optional

Almost every failure in this design is silent (03 §5). A policy that has its
sign inverted produces no error, no crash and no log line — just a hit rate that
looks like LRU's while consuming a Redis connection, a background thread and a
per-tick splice.

The previous attempt shipped unconditional `print()` to stderr and no counters,
and this fork has no Prometheus metrics anywhere (01 §8). So the failure mode is
not hypothetical: **there is currently no way to tell a working policy from a
broken one.**

---

## 3. Artifact 1 — request outcomes (exists) `[verified]`

`FileStatLogger` (`loggers.py:1272`) already writes per-finished-request JSONL
under `$VLLM_REQUEST_STATS_DIR`, carrying `job_id`, `langgraph_node`,
`prefill_time`, and the hit-class fields `num_local_cached_tokens` /
`num_external_cached_tokens`.

This is the **outcome** side of every join. It needs the three additions in
04 §3 (`call_type`, `arrival_ts`, `finish_ts`) and nothing else.

---

## 4. Artifact 2 — the eviction decision log `[required]`

One line per evicted **cached** block (fresh blocks carry no information), behind
a debug flag. This is the **decision** side.

```jsonc
{
  "ts": 1753980000123,
  "block_id": 84213,
  "keys": [["run-42", "research", "tavily:summary"]],  // multi-owner, 01 §4
  "score": 0.0141,
  "terms": {"prob": 0.85, "ttnc_ms": 60000, "decay": 0.33, "blocks": 1600},
  "speculative": false,
  "floor_applied": null,        // 02 §5, when speculative
  "prefix_pos": 1487,           // position within the node's prefix
  "run_len": 1600,              // to check tail-first, Rule 3
  "rank_in_splice": 12,
  "snapshot_age_ms": 340,       // staleness at decision time
  "num_free_fresh": 0,          // Rule 1 gate state
  "policy_on": true
}
```

Three fields carry more than they look:

- **`terms`** — logging the components, not just the total, is what makes a bad
  score *diagnosable* rather than merely visible. Without it every regression
  looks like "the formula is wrong".
- **`snapshot_age_ms`** — see §7.
- **`prefix_pos` / `run_len`** — the only way to verify Rule 3 is actually
  holding in production. A distribution of `prefix_pos` concentrated near
  `run_len` means tail-first is working; a uniform distribution means the policy
  is punching holes in the middle of prefixes and destroying whole matches for
  one block of reclaim.

---

## 5. Derived metrics

| Metric | Computation | Catches |
|---|---|---|
| **Score/outcome correlation** | mean score of blocks whose key was requested again within H, vs those not | **the sign error** |
| **Regret rate** | evictions whose key is requested within H ÷ all evictions | forecast wrong in the expensive direction |
| **Regret cost** | same, weighted by observed miss cost (`p_l1`/`p_cold`, 04 §5.4) | whether the regret actually *hurt* |
| **Calibration** | bucket by `prob`, compare to observed fire rate | `prob` systematically over/under-confident |
| **`ttnc` error** | predicted vs actual inter-request gap (04 §5.1 derives the gap) | the field doing most of the work (06 §3.3) |
| **Staleness attribution** | regret split by `snapshot_age_ms` | wrong forecast vs *late* forecast — §7 |
| **Speculative waste** | `evicted_before_confirm ÷ created` | prefetch predictions specifically (02 §5) |
| **Tail-first conformance** | `prefix_pos ÷ run_len` distribution | Rule 3 violations |
| **Unscored ratio** | blocks evicted with no score ÷ all | a broken join (03 §5) — silent otherwise |

### 5.1 The one number `[decided]`

**Score/outcome correlation.** If blocks that were needed again scored *lower*
than blocks that were not, the policy is anti-correlated — actively worse than
LRU, which is the failure 06 §7 names.

It is a single scalar, it needs no baseline run, and it should be checked before
anything else. A near-zero correlation says the formula carries no signal; a
negative one says ship the off switch today.

---

## 6. Always-on counters vs. the debug log `[decided]`

Per-eviction lines are unbounded under pressure, so they cannot be the
production mechanism.

| | Always on | Debug flag |
|---|---|---|
| What | counters only | full JSONL per eviction (§4) |
| Cost | O(1) increments | one line per evicted cached block |
| Answers | "is something wrong?" | "what exactly, and why?" |

**The always-on regret counter needs no offline join.** Keep a bounded ring
buffer of recent evictions keyed by `(job, node, call_type)` with their score and
timestamp. On request admission the key is already in hand, so a hit against that
buffer is an O(1) lookup and increments `evicted_then_needed`. That turns regret
from an analysis you have to run into a number the server reports.

Minimum always-on set:

```
evictions_by_score_total          # policy acted at all
evicted_then_needed_total         # regret
evicted_then_needed_cost_ms       # regret that mattered
speculative_created / _evicted_before_confirm
snapshot_stale_seconds            # transport health
unscored_eviction_ratio           # join health
policy_enabled                    # A/B arm, so dashboards can't lie
```

---

## 7. Separating a wrong forecast from a late one `[decided]`

A wrong forecast and a correct-but-stale forecast produce **identical
evictions** and need opposite fixes — one is the Prediction Engine's problem
(05), the other is transport (01 §2). Without `snapshot_age_ms` on the decision
line they are indistinguishable, and the natural instinct is to blame the
formula, which is the expensive place to look.

Split regret by `snapshot_age_ms`:

- flat across ages → the forecast is genuinely wrong → 05
- concentrated in old snapshots → delivery is late or messages are being missed
  → the pub/sub path, reseed interval, or the `update_ts` cutoff → 01 §2

This is one field and it halves the search space on every investigation.

---

## 8. What logs cannot answer

**Whether LRU would have done better on the same trace.** That is
counterfactual: logs show the mistakes the policy made, not whether it made
*more* than doing nothing.

Only the offline replay harness answers it — same captured trace, both policies,
compare hit rates. Which is also the cheapest formula-iteration loop available:
replay needs no GPU, no model and no Redis, so candidate scoring functions can be
compared in minutes rather than benchmark runs.

Recommended: build the replay harness alongside step 3, not after. It is the only
tool that can say "worse than LRU" out loud.

---

## 9. What to hand over for analysis

For a diagnostic session, two files covering the same window:

1. `$VLLM_REQUEST_STATS_DIR/finished_requests_engine*.jsonl` — outcomes
2. the eviction decision log — decisions

Plus, if available: a `PSUBSCRIBE '*'` capture of the Redis traffic for the same
window (03 §8), which makes the calibration and staleness analyses exact instead
of inferred.

Joined on `(job_id, langgraph_node, call_type)` and time, those produce every
metric in §5.

---

## 10. Build placement `[decided]`

The counters in §6 and the decision log in §4 ship **with step 3** (02 §10) —
the same step as the A/B flag and the speculative floor. Step 3 is the first
point where hit rate moves, which makes it the first point where it can move the
*wrong way*.

Shipping enforcement before observability means the first production signal of an
inverted policy would be a user reporting that things got slower.
