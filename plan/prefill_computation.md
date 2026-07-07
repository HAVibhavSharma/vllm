# How Prefill Is Computed in vLLM (v1 Engine)

Prefill is the phase where a request's prompt tokens are run through the model to
populate the KV cache, before any new tokens are decoded. In the v1 engine there is
**no separate "prefill batch" vs "decode batch"** — every scheduling step produces a
single unified batch where each request contributes however many tokens the scheduler
assigned it (a prompt chunk, a single decode token, or anything in between). Prefill
is therefore an emergent property of token accounting, not a distinct code path.

The lifecycle, end to end:

```
Scheduler (decides how many prompt tokens to run this step)
    └── KVCacheManager (prefix-cache hit → skip already-computed tokens; allocate blocks)
        └── GPUModelRunner._prepare_inputs (flattens all requests into one token stream)
            └── Attention backend (varlen kernel handles mixed prefill/decode uniformly)
                └── Scheduler.update_from_output (advances num_computed_tokens; prefill
                    finishes when num_computed_tokens == num_prompt_tokens)
```

---

## 1. Scheduling: how much prefill runs each step

**File: `vllm/v1/core/sched/scheduler.py` — `schedule()`**

The central bookkeeping variable is `request.num_computed_tokens`. A request is "in
prefill" while `num_computed_tokens < num_prompt_tokens`. Each step, the scheduler
decides `num_new_tokens` per request:

- A global **token budget** starts at `max_num_scheduled_tokens`
  (`scheduler.py:445`) and is decremented as requests are scheduled
  (`scheduler.py:613`, `scheduler.py:1008`).
- **Chunked prefill** falls out naturally: `num_new_tokens = min(remaining_prompt,
  token_budget)` (`scheduler.py:502`, `scheduler.py:875`). A long prompt that doesn't
  fit in the budget is simply split across multiple steps — no special mode.
- `long_prefill_token_threshold` (`scheduler.py:500`, `scheduler.py:853`) caps how
  many prompt tokens a single request may claim per step, so one huge prefill can't
  starve decodes.
- Running requests are scheduled first (`scheduler.py:463`), then waiting requests
  are admitted from the queue while budget remains (`scheduler.py:679`).

The output is `SchedulerOutput` (see `vllm/v1/core/sched/output.py`) carrying
`num_scheduled_tokens` per request — the model runner never knows "prefill" vs
"decode", only token counts.

## 2. Prefix caching: skipping already-computed prefill

**Files: `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/kv_cache_utils.py`**

Before a new request's prefill is scheduled, the scheduler calls
`kv_cache_manager.get_computed_blocks(request)` (`scheduler.py:725`, implementation
at `kv_cache_manager.py:183`). The prompt is hashed into block-sized chunks
(block hashing lives in `kv_cache_utils.py`), and the longest prefix of blocks
already resident in the cache is reused. Those tokens set the initial
`num_computed_tokens`, so prefill computation starts *after* the cached prefix —
a full-prefix hit means the request skips almost all prefill work.

Block allocation for the tokens that *will* be computed this step happens via
`kv_cache_manager.allocate_slots(...)` in the same scheduling pass. If blocks can't
be allocated, the scheduler may preempt lower-priority running requests
(`scheduler.py:580` area).

For external KV sharing (e.g. the LMCache connector this branch works with), the KV
connector can additionally report externally-cached tokens, which also advance
`num_computed_tokens` before local compute.

## 3. Model runner: building the flattened prefill batch

**File: `vllm/v1/worker/gpu_model_runner.py` — `_prepare_inputs()` (line 1818)**

All scheduled tokens from all requests are concatenated into one flat 1-D token
tensor. Per-request boundaries are described by `query_start_loc` — cumulative token
offsets (`gpu_model_runner.py:1932-1938`). For example, a step with a 512-token
prefill chunk and three decoding requests yields `query_start_loc = [0, 512, 513,
514, 515]`.

Other relevant pieces built here:
- **Positions / slot mapping**: each token's position within its sequence and its
  destination slot in the paged KV cache (via `vllm/v1/worker/block_table.py`).
- **Logits indices** (`gpu_model_runner.py:2081`): sampling only happens at the last
  token of each request's chunk — and for a partial prefill chunk the output is
  simply discarded, since no token can be sampled until the full prompt is processed.

The persistent per-request state lives in `vllm/v1/worker/gpu_input_batch.py`
(`InputBatch`), updated incrementally from each `SchedulerOutput`.

## 4. Attention: executing prefill

**Files: `vllm/v1/attention/backends/` (e.g. `flash_attn.py`)**

The v1 backends use variable-length ("varlen") kernels — e.g.
`flash_attn_varlen_func` (`flash_attn.py:40`) — driven by `query_start_loc` and
`seq_lens`. A request whose query length > 1 is doing prefill (causal attention over
its chunk plus all previously cached KV); query length == 1 is a decode. Both run in
the **same kernel launch**. New K/V values are scattered into the paged KV cache
using the slot mapping.

`flash_attn.py` also implements **cascade attention** (`flash_attn.py:243`,
`flash_attn.py:476`): when many requests share a common prefix, the shared-prefix
attention is computed once and merged with per-request suffix attention.

Backend metadata (query starts, seq lens, block tables) is assembled per-step by
each backend's metadata builder; the registry is `vllm/v1/attention/backends/registry.py`.

## 5. Completion: when prefill ends

**File: `vllm/v1/core/sched/scheduler.py` — `update_from_output()`**

After execution, each request's `num_computed_tokens` advances by its scheduled
token count. Once it reaches `num_prompt_tokens`, the sampled token from that step is
the request's **first output token** — the request has transitioned to decode. The
condition at `scheduler.py:397` (`num_computed_tokens < max(num_prompt_tokens,
num_tokens - 1)`) is what distinguishes still-prefilling (and resumed/preempted)
requests from decoding ones.

---

## Quick reference: where to look

| Concern | File |
|---|---|
| Chunked prefill budgeting, admission, preemption | `vllm/v1/core/sched/scheduler.py` (`schedule()`) |
| Per-step scheduling output structures | `vllm/v1/core/sched/output.py` |
| Prefix-cache hit lookup, block allocation | `vllm/v1/core/kv_cache_manager.py` |
| Block hashing / prefix-cache utilities | `vllm/v1/core/kv_cache_utils.py` |
| Flattened batch construction, `query_start_loc` | `vllm/v1/worker/gpu_model_runner.py` (`_prepare_inputs`) |
| Persistent per-request batch state | `vllm/v1/worker/gpu_input_batch.py` |
| Prefill/decode attention execution, cascade attention | `vllm/v1/attention/backends/flash_attn.py` |
| Block table → slot mapping | `vllm/v1/worker/block_table.py` |
| Config knobs (`max_num_batched_tokens`, `long_prefill_token_threshold`, `enable_chunked_prefill`) | `vllm/config/scheduler.py` |
