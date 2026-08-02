# Trace Analyser — Implementation Spec

**Status:** spec, nothing implemented.
**Written:** 2026-07-31, against branch `vllm-v2` @ `68c7acb14`.
**Component:** external process. Consumes vLLM's per-request timing records,
aggregates them, writes the `HISTORY` keyspace to Redis.
**Interface:** [`03-workflow-contract.md`](03-workflow-contract.md) is the
contract this must satisfy. **Read it first** — this doc says how, 03 says what.
**Companion:** [`05-prediction-engine.md`](05-prediction-engine.md) writes
`INFO`/`PROB`; this one writes `HISTORY` only. They must not write each other's
keys.

`[verified]` = read at the cited line in this repo. `[required]` = engine-side
change needed before this component can be built. `[open]` = decide during
implementation.

---

## 1. Role, in one paragraph

vLLM measures; the Trace Analyser aggregates and publishes. Every timing this
component needs is produced by the engine — it does **not** instrument the
workflow, does not time anything itself, and does not talk to the Prediction
Engine. It reads a stream of finished-request records, maintains per-`(node,
call_type)` statistics that persist across jobs, and writes `HISTORY|<job_id>`
to Redis for the vLLM eviction policy to read back.

```
vLLM FileStatLogger ──JSONL──▶ Trace Analyser ──HISTORY──▶ Redis ──▶ vLLM policy
     (measures)                  (aggregates)                          (consumes)
```

The loop is deliberate: the engine's own measurements come back to it as a
forecast. Nothing else measures LLM time, because nothing else can.

---

## 2. Input — what the engine already emits `[verified]`

`FileStatLogger` (`vllm/v1/metrics/loggers.py:1272`) writes one CSV row and one
JSONL line per finished request to `$VLLM_REQUEST_STATS_DIR`, enabled by that
env var alone (`vllm/envs.py:722`). Columns at `loggers.py:1249`, populated at
`:1317`, sourced from `FinishedRequestStats` (`vllm/v1/metrics/stats.py:222`).

| Field | Type | Use here |
|---|---|---|
| `request_id` | str | dedupe key |
| `job_id` | str\|null | Redis key for the output |
| `agent_id` | str\|null | ignore (namespaced; not a join key — 03 §1) |
| `langgraph_node` | str\|null | join key |
| `e2e_latency` | float s | → `time_taken` |
| `num_prompt_tokens` | int | → `token_input` |
| `num_generation_tokens` | int | → `token_output` |
| `queued_time` | float s | queue-vs-compute split |
| `prefill_time` | float s | **the quantity a cache hit actually saves** |
| `inference_time`, `decode_time` | float s | generation cost, cache-independent |
| `num_cached_tokens` | int | hit/miss classification |
| `num_local_cached_tokens` | int | **GPU (APC) hit** |
| `num_external_cached_tokens` | int | **LMCache/connector hit** |
| `prefix_cache_hit_rate` | float | convenience |
| `finish_reason` | str | drop non-`stop` from statistics |

`num_local_cached_tokens` vs `num_external_cached_tokens` is the pair that makes
the L1-residency question answerable at all — see §5.4. It is the reason this
component can test 01 §5's central assumption without any new instrumentation
anywhere.

---

## 3. Engine-side gap — close this first `[required]`

Three fields are missing. None is hard; all are prerequisites.

### 3.1 `call_type` is not in the codebase at all

`job_id`, `agent_id` and `langgraph_node` are read from
`sampling_params.extra_args` at `output_processor.py:248/251/254` `[verified]`.
`call_type` is read nowhere. Add it alongside them, then carry it through
`FinishedRequestStats` (`stats.py:222`),
`update_from_finished_request` (`stats.py:442`), `_CSV_COLUMNS`
(`loggers.py:1249`) and the row dict (`loggers.py:1317`).

Without it every record aggregates at node granularity and the exact join
promised in 03 §1 does not exist.

### 3.2 No absolute timestamps `[required]` — the blocking one

`FinishedRequestStats` carries **durations only**. `arrival_time` exists on
`RequestStateStats` (`stats.py:208`) and is consumed to compute `e2e_latency`
(`stats.py:459`) but never emitted.

**`tool_execution_time` is a gap between two requests, and a gap cannot be
derived from durations.** Add:

- `arrival_ts` — absolute, wall clock
- `finish_ts` — absolute, wall clock (`arrival_ts + e2e_latency`)

### 3.3 Clock discipline — read this before touching stats.py `[verified]`

There are **two clocks** in that file. `arrival_time` is wall clock: it is
differenced against `iteration_timestamp = time.time()` (`stats.py:343`, `:363`,
`:459`). The engine-core timestamps — `queued_ts`, `scheduled_ts`,
`first_token_ts`, `last_token_ts` — are **monotonic**, flagged by the comment at
`stats.py:210`.

Monotonic values are meaningless across processes and across restarts. Export
**only** wall-clock absolutes; keep every monotonic value as a duration. Mixing
them produces gaps that look plausible and are arbitrary.

---

## 4. Transport from engine to analyser

**v1: tail the JSONL file.** `VLLM_REQUEST_STATS_DIR` is already the mechanism,
so this needs zero engine code beyond §3, and the analyser can be restarted,
rerun over history, and tested against a captured file.

Requirements on the reader:

- **Tail, don't re-read.** Checkpoint the byte offset per file; resume there.
- **Tolerate partial lines** — the writer is line-buffered but not atomic. A
  trailing incomplete line is normal; keep it in a buffer, don't discard it.
- **Handle rotation.** One file per engine per start:
  `finished_requests_engine{N}_{ts}.jsonl` (`loggers.py:1288`). Watch the
  directory, not one path.
- **Handle data parallelism.** One file *per engine index*. Records for one job
  can be split across engines. Merge by `job_id`, and sort by `arrival_ts`
  before computing any gap.

`StatLoggerBase` is a pluggable interface, so a push-based logger (straight to
Redis or a queue) is a drop-in replacement later. Don't build it first — the
file gives replayability, which matters much more while the scoring formula is
still unknown.

---

## 5. Derivations — what to compute

Per finished record, keyed `(langgraph_node, call_type)`:

| `HISTORY` field | Derivation | Notes |
|---|---|---|
| `time_taken` | `e2e_latency` | Total observed LLM call duration. |
| `token_input` | `num_prompt_tokens` | Prefix size → blocks. |
| `token_output` | `num_generation_tokens` | Per-turn growth for react nodes. |
| `tool_execution_time` | `next.arrival_ts − prev.finish_ts`, consecutive records **within one `job_id`**, ordered by `arrival_ts` | See §5.1. |
| `prediction_headroom` | the same gap, minus the measured time to warm a prefix | See §5.2. |

### 5.1 `tool_execution_time` is an inferred gap, not a measurement

vLLM cannot see tool execution — it sees only that request N finished and
request N+1 arrived. The gap is therefore **tool time + workflow overhead +
network + any human delay**, and it is the only signal available. Name the field
honestly in your own storage (`inter_request_gap`) even though it is published
as `tool_execution_time`.

Rules:

- Compute within one `job_id` only. Never across jobs.
- Order by `arrival_ts` after merging engine files (§4).
- **Drop the gap when `prev.finish_reason != stop`** — an aborted request's
  finish time means nothing.
- **Cap the gap.** An idle workflow, a crashed job, or a human-in-the-loop pause
  produces gaps of minutes. Discard anything above a configured ceiling rather
  than letting it into the mean; a single 10-minute gap makes a node look
  permanently unneeded, which is the exact failure that silently disables
  protection for a hot prefix.
- Gaps need ≥2 records in a job. A single-turn job contributes `time_taken` and
  token counts but no gap.

### 5.2 `prediction_headroom`

The lead time a prefetch has: how long after request N finishes before N+1
needs the prefix, minus what warming costs. v1: publish the capped gap and let
the consumer subtract. Refine once phantom-prefetch durations are recorded.

### 5.3 Statistics — do not use a plain mean `[decided]`

The measured distribution is bimodal by construction: ~200 ms on a GPU hit,
~380 ms on an L1 reconstruct, ~11.4 s on a cold prefill (00 Part 2). One cold
prefill drags a 20-sample mean by ~500 ms and the node looks expensive forever.

- Use an **EWMA** (α ≈ 0.2, tune) or a **trimmed median** over a bounded window.
- Keep `n` alongside every statistic and publish it. A mean over 2 samples must
  not be trusted like a mean over 200; the consumer needs to know which it has.
- **Segment by hit class** — see §5.4. Publishing one blended mean destroys the
  quantity the policy actually wants.

### 5.4 Hit-class segmentation — this answers the L1 question `[decided]`

Classify **every** request from the two fields the engine already emits
separately `[verified]`:

| Class | Signature | Expected cost |
|---|---|---|
| GPU hit | `num_local_cached_tokens > 0` | ≈200 ms |
| L1 reconstruct | `local == 0`, `num_external_cached_tokens > 0` | ≈380 ms |
| **cold** | both `0` | **≈11 400 ms** |

Publish per `(node_name, call_type)`:

| Field | Meaning |
|---|---|
| `p_gpu` | fraction served from GPU/APC |
| `p_l1` | fraction reconstructed from LMCache |
| `p_cold` | fraction paying full prefill |
| `time_taken_gpu` / `time_taken_l1` / `time_taken_cold` | segmented means |

**Why this is the highest-value thing this component produces.** 01 §5's scorer
assumes every evicted prefix is recoverable from L1 at ≈180 ms. `p_cold` is the
direct test of that assumption:

- `p_cold ≈ 0` → the assumption holds, and the open item in 01 §5 closes with no
  scoring change at all.
- `p_cold` non-trivial → the consumer switches to an expected cost,
  `p_l1 · 180ms + p_cold · 11400ms`. At a 60× ratio, **`p_cold = 2%` already
  doubles expected eviction cost**, so this is not a rounding correction.

Precision is not the goal — not treating 11.4 s as 180 ms is. Publish it from
day one even if the scorer ignores it at first; it is also the only way to tell
whether the policy beats LRU on the axis that matters, rather than on the
~180 ms axis the old benchmark measured (00 Part 2).

Do **not** try to answer this by querying the KV connector for residency. That
is an RPC per block and a second ingress into the engine process — rejected in
01 §5. This component answers it from data that already exists.

---

## 6. Aggregation scope — cross-job stats, per-job keys `[open]`

A tension to resolve explicitly, because the two docs it comes from disagree in
spirit:

- `HISTORY` is keyed **per `job_id`** (03 §2).
- Timing statistics are only useful **across jobs** — a new job has no history of
  its own, and 02 §6 already specifies the timing memo as persisting across jobs.

**Recommended:** maintain the aggregate keyed `(node_name, call_type)` across all
jobs, and write that aggregate into each active job's `HISTORY|<job_id>` row.
A new job therefore inherits useful statistics on its first turn instead of
being unscored for its whole first pass — which is exactly when eviction
pressure from a cold cache is highest.

Consequence to accept: two concurrent jobs running the same node see identical
`HISTORY`. That is correct — the timing of `research/summary` is a property of
the node, not of the run. Divergent per-job behaviour belongs in `PROB`, which
the Prediction Engine owns per job.

---

## 7. Redis write discipline

- **Only this component writes `HISTORY`.** `INFO`/`PROB` belong to the
  Prediction Engine (05). Shared-key writes make ownership unrecoverable.
- **`PUBLISH` after every `SET`.** The engine's steady-state transport is
  pub/sub; a writer that only `SET`s leaves the policy frozen at its seed
  (03 §3.1). This is the single highest-value requirement in the doc.
- **Publish on change only.** Re-publishing an unchanged aggregate every N
  seconds forces snapshot rebuilds for nothing. The engine change-detects, but
  don't rely on it.
- **Set `update_ts` on every write**, wall clock, matching the units pinned in
  03 §4. The staleness gate is what makes a missed message degrade to LRU rather
  than act on stale data — a row without a timestamp defeats it.
- **Idempotent writes.** Replaying the same JSONL must converge to the same
  state, or testing against a captured file proves nothing.
- **Never block on Redis.** Redis being down must not stall ingestion; buffer
  and retry with backoff.

---

## 8. Configuration

| Setting | Purpose |
|---|---|
| stats dir | matches `VLLM_REQUEST_STATS_DIR` |
| Redis endpoint / auth / db | shared with 05 |
| EWMA α or trim window | §5.3 |
| gap ceiling | §5.1 — the one that silently ruins statistics if wrong |
| min samples before publishing | avoid publishing an n=1 "mean" |
| checkpoint path | resume offset (§4) |
| publish debounce | floor on write frequency |

---

## 9. Tests

All of these run with no GPU, no vLLM, and no Redis (fake client):

1. **Replay** a captured JSONL → asserted `HISTORY` output. The primary test.
2. **Gap computation**: two records, one job → one gap. Interleaved jobs → no
   cross-job gaps. Non-`stop` finish → gap dropped. Over-ceiling → dropped.
3. **Single-turn job** → `time_taken` and tokens published, no gap.
4. **Partial trailing line** → buffered, then completed on the next read.
5. **Rotation** — a second engine file appears mid-run; both are consumed.
6. **Restart** — resume from checkpoint; no double counting (idempotence).
7. **Outlier**: one 11.4 s record among twenty 200 ms records must not move the
   published statistic more than a configured bound.
8. **Clock**: assert no monotonic value is ever published as an absolute (§3.3).

---

## 10. Build order

| Step | What | Blocked by |
|---|---|---|
| **1** | Engine change: add `call_type`, `arrival_ts`, `finish_ts` (§3) | nothing |
| **2** | JSONL tailer with checkpointing, rotation, partial lines (§4) | step 1 for real data; fixtures work now |
| **3** | Derivations + robust statistics (§5) | step 2 |
| **4** | Redis writer with publish discipline (§7) | Redis endpoint |
| **5** | Hit-class segmentation and `p_cold` (§5.4) — the direct test of the "assume L1 resident" assumption in 01 §5 | step 3 |

Steps 1–3 are unblocked and testable against fixtures today.

Step 5 is cheap and worth pulling forward: it needs no new input, and `p_cold`
either closes an open design item or changes the cost model materially. It is
the fastest question in the whole plan to answer.
