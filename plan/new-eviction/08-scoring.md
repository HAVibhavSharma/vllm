# The Scoring Function — v1 Spec

**Status:** shape `[decided]`, weights and constants `[open]`, **nothing
validated**. This is the one part of the design that cannot be closed by
reasoning — it needs a replay against real traces (§7).
**Written:** 2026-07-31, against branch `vllm-v2` @ `68c7acb14`.
**Reads with:** [`01-redis-integration.md`](01-redis-integration.md) §5 (the
inputs) and §6 (what consumes the output),
[`05-prediction-engine.md`](05-prediction-engine.md) §5.2 (what `prob` means),
[`04-trace-analyser.md`](04-trace-analyser.md) §5.4 (where `p_cold` comes from).

---

## 1. The frame

The decision is: **given fixed HBM, which blocks do we keep?** That is a
keep/drop problem over a capacity constraint, so the right quantity is a
**value density** —

> **expected milliseconds of latency saved, per block of HBM held.**

Not "how likely is this to be reused" (ignores cost and size), and not "how
expensive was this to build" (ignores whether it will ever be reused).

Units matter here: a score in ms/block is comparable across nodes, across jobs,
and against a wall-clock budget. A unitless "importance" is not, and it makes
every constant unfalsifiable.

---

## 2. The formula

```
score(key) = prob(key)
           × decay(time_to_next_call(key))
           × E_miss(key)
           ÷ blocks(key)

decay(t)   = τ / (τ + t)                       τ ≈ 30 s        [open]
E_miss     = p_l1 · Δ_l1  +  p_cold · Δ_cold   Δ_l1 ≈ 180 ms
                                               Δ_cold ≈ 11 400 ms
```

| Term | Meaning | Source |
|---|---|---|
| `prob` | P(fires again **at all** this job) — time-free | `PROB`, 05 §5.2 |
| `decay(ttnc)` | time discount — sooner is worth more | `PROB.time_to_next_call` |
| `E_miss` | expected cost if we drop it and it is needed | `HISTORY`, 04 §5.4 |
| `blocks` | prefix length in blocks | the index, free at tick time |

### Why each term

- **`prob`** — value only exists if the prefix is reused. Time-free by
  construction so the discount is applied exactly once (05 §5.2); encoding
  "soon" in both fields squares the preference for near-term nodes.
- **`decay`** — a block needed in 60 s occupies memory for 60 s that could
  serve other work. `τ/(τ+t)` is chosen over `exp(-t/τ)` for having a fatter
  tail: an 11.4 s prefix rebuild is still worth avoiding at t = 2 min, and an
  exponential writes that off too fast.
- **`E_miss`** — makes the cost of being wrong explicit. Under the current
  L1-resident assumption this is a constant and **cancels out of the ranking**
  (§3), re-entering automatically once `p_cold` is measured (01 §5).
- **`÷ blocks`** — freeing 1500 blocks and freeing 100 blocks are not the same
  action. This is what makes the score a density rather than a preference.

### Why dividing by `blocks` is legitimate

01 §6 warns that freeing a block from the middle of a 1500-block prefix destroys
the whole match. True for the *middle* — **not for the tail.** A prefix truncated
from the tail is still a valid match for its first N blocks, which is exactly why
Rule 3 evicts tail-first.

So retained value is roughly linear in blocks retained, and marginal block value
≈ `E_miss / blocks`. Without tail-first ordering this term would be a category
error; with it, it is the correct marginal.

---

## 3. What v1 actually computes

Under 01 §5's "assume L1 resident", `p_cold = 0` and `E_miss = Δ_l1`, a
constant. Constants do not change a ranking, so **v1 collapses to:**

```
score = prob × decay(time_to_next_call) ÷ blocks
```

Implement the full form anyway, with `E_miss` read from `HISTORY` and defaulted
to `Δ_l1`. Then measuring `p_cold` (04 §5.4) changes a data file, not the code.

---

## 4. Composition with the rest of the policy

Applied in this order:

```
1.  base        = score(key)                     as above
2.  block_score = max(base over owning keys)      multi-owner   01 §4
3.  if speculative and the floor is switched on:  02 §5, 12 §6
        block_score = max(block_score, floor)     off by default
4.  unscored blocks are never selected — they keep LRU position 01 §6 Rule 2
5.  splice moves the worst K, ascending           01 §6 Rule 4
```

**`max` over owners, not sum.** A shared system preamble owned by five nodes is
worth what its most valuable owner makes it worth; summing would make common
prefixes unevictable regardless of whether anything needs them.

**Unscored needs no numeric constant in v1.** Because the splice only *moves*
the worst K, an unscored block is never selected and simply keeps its LRU
position. The "neutral default constant" in the earlier drafts is not required
until something ranks scored against unscored globally.

---

## 5. Invariants — testable with no data at all

These are properties of the formula, checkable in unit tests before any trace
exists. Several encode failures the previous attempt actually shipped.

| # | Invariant | Why |
|---|---|---|
| 1 | strictly increasing in `prob` | — |
| 2 | strictly decreasing in `time_to_next_call` | — |
| 3 | strictly decreasing in `blocks` | it is a density |
| 4 | increasing in `p_cold` | expensive misses are worth avoiding |
| 5 | **never 0** for a scored block | 0 makes freshly built prefixes the top victims and destroys them before first reuse — strictly worse than LRU (00 Part 5 item 6) |
| 6 | bounded above by `Δ_cold` | so the speculative floor *can* be placed above the whole range (02 §5). Still the invariant the validator enforces, though the floor now defaults to off — 12 §6 |
| 7 | deterministic — no clock reads inside, no randomness | the same inputs must replay identically, or §7 proves nothing |
| 8 | a running node's blocks are never scored | `ref_cnt > 0` keeps them out of the free queue entirely |

Invariant 7 is the one that is easy to violate by accident: reading
`time.monotonic()` inside the score rather than passing a tick timestamp in makes
replay non-reproducible and the harness useless.

---

## 6. Open — the actual unknowns

| Item | Notes |
|---|---|
| `τ` | 30 s is a guess. It sets how far ahead the policy plans, and it interacts with tick period and typical tool duration. |
| Δ_l1, Δ_cold | ≈180 ms / ≈11.4 s from one old benchmark on one model and one prompt size. Should come from `HISTORY` segmented means (04 §5.4), not constants. |
| Whether `prob` carries signal | If a workflow has one dominant path, `p_reach` is near-uniform and `decay` does all the work (05 §5.2.1). |
| Splice selection | Unconditional worst-K, or only blocks below an absolute threshold? Unconditional always moves K blocks even when every candidate is valuable — which under a healthy cache is churn for nothing. **Unresolved.** |
| Age term | 00 Part 5 item 7 wants age as a score term so silent jobs decay out. Partly subsumed by the `update_ts` staleness gate; needs deciding whether an explicit term is still required. |

---

## 7. Validation — offline replay, before any GPU `[decided]`

**The formula cannot be validated by inspection, and it does not need a server
to validate.**

You already have the inputs: `FileStatLogger` JSONL gives every request with its
`job_id`, node, prefix size and hit class (04 §2), and the eviction decision log
gives every decision (07 §4).

The harness:

1. Replay a captured trace, simulating the block index and free queue in plain
   Python — no model, no GPU, no Redis.
2. At each allocation, apply a candidate scoring function and record which blocks
   would have been evicted.
3. Compare against what was actually reused later in the same trace.
4. Report hit rate, and the score/outcome correlation (07 §5.1).

What this buys:

- **A counterfactual LRU arm on the identical trace** — the only way to answer
  "is this worse than doing nothing", which no amount of production logging can
  (07 §8).
- **Formula iteration in minutes**, so `τ`, the term set, and threshold-vs-worst-K
  can be swept rather than argued about.
- **A falsification test for each term**: drop `prob`, drop `decay`, drop
  `÷ blocks`, and see which ablations actually move hit rate. It is entirely
  possible that one term carries everything and the rest are decoration — better
  to learn that from a sweep than to ship three terms and tune all of them.

**Build this alongside step 3.** It is cheaper than one benchmark run and it is
the only thing that can say the policy is worse than LRU out loud.

---

## 8. Anti-patterns

Each of these has a specific failure attached; none is hypothetical.

- **Scoring unscored blocks 0.** Destroys freshly built prefixes before first
  reuse. Worse than LRU.
- **Encoding "soon" in `prob`.** Applies the time discount twice (05 §5.2).
- **Summing over multi-owners.** Makes shared preambles permanently unevictable.
- **Reading the clock inside the score.** Breaks replay reproducibility (§5,
  invariant 7).
- **Tuning constants against the old benchmark.** Its baseline served every miss
  from LMCache L1, so it measures GPU-vs-L1 only (06 §7).
- **Shipping the formula before the counters.** An inverted policy is
  indistinguishable from a working one without the score/outcome correlation
  (07 §5.1).
