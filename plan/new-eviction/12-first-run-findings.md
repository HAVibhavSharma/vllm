# First Real Run — What Broke, and the Four Fixes

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
- **Prefetch never executed.** `speculative blocks=0` for the whole run, and
  51 log lines of `want for langgraph:supervisor has no registered prefixes
  yet -- nothing to warm`. Half the design is untested.
- **The measurement regime is wrong for the question.** `--max-num-seqs 1`
  meant `max running: 1` throughout, and LMCache absorbed ~50% of traffic at
  `delta_l1=180ms`, so an HBM miss is cheap and end-to-end latency was a wash
  (+1.4%). Re-run with real concurrency before drawing conclusions either way.
- **`GPU KV cache usage` reads ~4%** while `index_blocks` is 9,520/9,558.
  vLLM's usage counts referenced blocks only; the prefix cache was full the
  whole time. Do not read low usage as "no eviction pressure".
