# Inter-question KV reuse: which question paid for the prefix

**Status:** as-built. Code: `vllm/v1/core/kv_provenance.py`,
hooks in `vllm/v1/core/hbm_summary.py`, `vllm/v1/core/block_pool.py`,
`vllm/v1/core/sched/scheduler.py`, `vllm/v1/engine/core.py`,
`vllm/entrypoints/openai/kv_metrics_router.py`.
Tests: `tests/v1/core/test_kv_provenance.py`.

## The question this answers

`hbm_summary.py` reports *how much* of the prefix cache was hit. It cannot
report *whose*: `on_cache_query` is handed two integers, and a 60% hit rate
looks identical whether every request is re-reading its own prefill or
question 12 is riding entirely on blocks question 7 paid for.

For a Deep Research workload the second case is the point. Questions share a
system prompt, tool schemas and often whole research turns, so the claim worth
measuring is "question 12 skipped 40k tokens of prefill because questions 3
and 7 had already computed them" — and no aggregate the server emits today
can distinguish that from question 12 talking to itself.

## Mechanism

One extra fact per block: **the first job that computed it.**

```
claim   job 7 prefills; its blocks become full        hash H -> job 7
lookup  job 12 arrives, the prefix cache hits on H    16 of job 12's cached
                                                      tokens came from job 7
```

A job is a question: `tests/run_evaluate_prefix.py::target` in the
`open_deep_research` repo mints a fresh `job_id` per graph invocation and
forwards it in `extra_body`, which vLLM's OpenAI layer folds into
`sampling_params.extra_args` — the same path `langgraph_node` and `agent_id`
already take. Nothing new is plumbed across the front-end → engine-core
boundary.

### Why the key is the bare block hash

Keyed on `Request.block_hashes` entries at `hash_block_size` granularity, not
on block ids and not on the group-qualified hash stored on a `KVCacheBlock`:

1. **Block ids are recycled.** An id names a slot in HBM, not content, so an
   id-keyed owner is overwritten by whoever allocates the slot next — and the
   provenance of an evicted-then-rematerialized prefix is lost exactly when
   it matters.
2. **External hits never touch a `KVCacheBlock`.** LMCache hits are hits all
   the same, and for cross-question reuse they are the *likely* case: HBM
   churns within one job while the CPU tier survives between them. The only
   handle on them is the request's own hash list.
3. **First-writer-wins is the honest attribution.** A prefix ten jobs share
   was paid for once. Crediting the tenth reader would report reuse that
   never happened.

### Where the hooks sit

| Hook | Site | Why there |
|---|---|---|
| `claim` | `BlockPool.cache_full_blocks` → `HBMSummaryLogger.on_blocks_cached` | The one place a block's content becomes cacheable, already iterating the blocks and their hashes. |
| `record_prefill` | `Scheduler.schedule`, beside `request.prefill_stats.set` | The only point where **both** tiers are known: local is `KVCacheManager`, external is the connector, and a cross-question hit is usually the second. |

`record_prefill` is guarded by `Request.reuse_provenance_counted` for the same
reason `cache_query_counted` exists: the enclosing block re-runs on every step
a request spends in the waiting queue, so counting per call would weight the
numbers by queueing delay.

### Three buckets, never merged

| Bucket | Meaning |
|---|---|
| `self` | The job re-reading its own earlier prefill. Real savings, but the *within*-job effect, not this one. |
| `cross` | Blocks another question paid for. **The number being measured.** |
| `unknown` | A hit on content this process never saw claimed — a connector store that outlived the server, or a claim aged out under `VLLM_KV_PROVENANCE_MAX_HASHES`. Not evidence of anything. |

Phantom (`/v1/agents/prefetch`) requests **claim** but do not **consume**: a
prefetch warms the cache on some job's behalf, so its blocks belong to that
job, but the hit it experiences is one no question ever waited on.

### Non-contiguous external runs

When the connector reports matched ranges via `get_external_runs`, those are
used in preference to `num_external_computed_tokens` — which is only the
*leading* run. A run overlapping the local hit is clipped, so a prefix both
tiers report cannot be counted twice. Both totals go into every record
(`external_leading_tokens`, `external_run_tokens`) so the difference stays
visible.

### What a measurement reset does *not* clear

`POST /v1/kv_metrics/reset` zeroes the reuse aggregates and **keeps the claim
map**. The warm phase is supposed to hit what the warmup left resident (and
what the connector still holds); forgetting who wrote those blocks would
re-label every one of those hits `unknown` and erase the effect the run exists
to measure.

## Turning it on

Off unless asked for — it costs a dict entry per distinct block hash and a
lookup per hit block on the scheduling path, and a baseline arm should be able
to run without it.

```bash
VLLM_HBM_SUMMARY_PERIOD_MS=30000 \
VLLM_KV_PROVENANCE=1 \
VLLM_KV_PROVENANCE_PATH=/abs/path/to/run/kv_reuse.jsonl \
vllm serve ...
```

| Env | Default | Meaning |
|---|---|---|
| `VLLM_KV_PROVENANCE` | `0` | `1` enables. Requires `VLLM_HBM_SUMMARY_PERIOD_MS != 0`, since the tracker hangs off `HBMSummaryLogger`. |
| `VLLM_KV_PROVENANCE_PATH` | unset | JSONL sink, one record per prefill. Unset = aggregates only. |
| `VLLM_KV_PROVENANCE_MAX_HASHES` | `1000000` | Claim-map cap, FIFO-trimmed oldest-first. |
| `VLLM_KV_PROVENANCE_TOP` | `8` | Pairs printed on the `kv_reuse` line. |

## Reading it

**Log line**, emitted next to `kv_hbm` on the same rate-limited, change-gated
period (its own line so `kv_hbm` stays byte-compatible with the policy
branch's for field-by-field diffing):

```
kv_reuse epoch=1 label=cold_done reqs=412 hit_tokens=1893376 local=1204224
  external=689152 self=612352 cross=1207296 unknown=73728 cross_frac=0.637
  src_jobs=9 claims=214883 trimmed=0 top=7->12:409600,3->12:212992,...
```

**`GET /v1/kv_metrics`** — the same numbers, structured, epoch untouched. Poll
it mid-run to watch cross-question reuse build up as questions accumulate.
Key fields: `reuse_cross_tokens`, `reuse_self_tokens`, `reuse_unknown_tokens`,
`reuse_by_source_job`, `reuse_by_consumer_job`, `reuse_top_pairs`.

**JSONL**, one record per prefill:

```json
{"job_id": "12", "node": "researcher", "prompt_tokens": 9184,
 "hit_tokens": 4096, "hit_tokens_local": 2048, "hit_tokens_external": 2048,
 "self_tokens": 1024, "cross_tokens": 3072, "unknown_tokens": 0,
 "sources": {"7": 2048, "3": 1024}}
```

Full misses are recorded too, with zeros. They are the denominator: "job 12
reused 40k tokens from job 7" means nothing without the prompts that reused
nothing.

**Rollup with question text** — in the `open_deep_research` repo, where the
`job_id → example_id → question` mapping lives in `query_summary.jsonl`:

```bash
python tests/analyze_kv_reuse.py --records /abs/path/to/run/kv_reuse.jsonl
```

`tests/kv_reuse_snapshot.py` there fetches `GET /v1/kv_metrics`;
`run_evaluate_prefix.py` calls it at the end of a run and drops the payload in
`kv_reuse_snapshot.json` beside the run's other metrics.
