# Exposing `num_computed_tokens` per request

This change surfaces, per request, the number of prompt tokens that were
**actually computed locally during prefill** (the real GPU compute work), as a
first-class stat. It complements the already-tracked `num_cached_tokens`.

```
num_computed_tokens = num_prompt_tokens - num_cached_tokens
```

- `num_cached_tokens` — prompt tokens skipped during prefill (local prefix-cache
  hits + external KV transfer). *Already logged before this change.*
- `num_computed_tokens` — prompt tokens that had to be run through the model.
  *New.*

Source of truth is `PrefillStats.num_computed_tokens` (see
`plan/prefill_computation.md`), captured at the moment prefill completes.

> Note on naming: the scheduler has a local variable also called
> `num_computed_tokens` (`scheduler.py::_mamba_block_aligned_split`,
> `request.num_computed_tokens + num_new_local_computed_tokens +
> num_external_computed_tokens`) that means the *cached/skipped* count. The stat
> added here is the **compute-work** value (`PrefillStats.num_computed_tokens`),
> which is the complementary quantity and the one that was not previously
> exposed.

---

## Where it now shows up in output

### 1. Per-request stats file logger (`FileStatLogger`)

Enabled by setting `VLLM_REQUEST_STATS_DIR`. Each finished request is written to
both `finished_requests_engine<N>_<ts>.csv` and the matching `.jsonl`.

A new `num_computed_tokens` column/field sits next to `num_cached_tokens`.

**CSV header (new column highlighted):**

```
request_id,job_id,agent_id,langgraph_node,input_text,output_text,finish_reason,
e2e_latency,num_prompt_tokens,num_generation_tokens,num_cached_tokens,
num_computed_tokens,prefix_cache_hit_rate,queued_time,prefill_time,
inference_time,decode_time,max_tokens_param
```

**JSONL row (excerpt):**

```json
{
  "request_id": "chatcmpl-abc123",
  "num_prompt_tokens": 1024,
  "num_cached_tokens": 768,
  "num_computed_tokens": 256,
  "prefix_cache_hit_rate": 0.75,
  ...
}
```

Here 768 of the 1024 prompt tokens were served from cache and only 256 were
actually prefilled on the GPU.

### 2. HTTP API — `/v1/agents/chat/completions` (and `/v1/chat/completions`)

The agent endpoint delegates to the standard chat-completion handler, so the
value flows through the normal `usage` object.

A new `computed_tokens` field is added to `prompt_tokens_details`, alongside the
existing `cached_tokens`.

**Gated by the server flag `--enable-prompt-tokens-details`** (same gate as
`cached_tokens`). Without it, `prompt_tokens_details` is omitted as before.

**Non-streaming response `usage`:**

```json
{
  "usage": {
    "prompt_tokens": 1024,
    "completion_tokens": 42,
    "total_tokens": 1066,
    "prompt_tokens_details": {
      "cached_tokens": 768,
      "computed_tokens": 256
    }
  }
}
```

**Streaming response** — same shape in the final usage chunk (sent when
`stream_options.include_usage` is true).

Invariant in the payload:

```
prompt_tokens_details.cached_tokens
  + prompt_tokens_details.computed_tokens
  == usage.prompt_tokens
```

---

## Backward compatibility

- **CSV/JSONL**: a new trailing-ish column is added. Consumers that key by column
  name are unaffected; positional CSV parsers should re-read the header.
- **API**: `computed_tokens` is additive and defaults to `null`. It only appears
  when `--enable-prompt-tokens-details` is set. Existing clients that ignore
  unknown fields are unaffected.
- `RequestOutput.num_computed_tokens` defaults to `None`, so external code
  constructing `RequestOutput` directly keeps working.

---

## Files changed

| File | Change |
|---|---|
| `vllm/v1/metrics/stats.py` | `FinishedRequestStats.num_computed_tokens`; `update_from_finished_request(...)` accepts + stores it |
| `vllm/v1/engine/output_processor.py` | Track `RequestState.num_computed_tokens` from `prefill_stats`; pass to finished stats and to `RequestOutput` |
| `vllm/outputs.py` | `RequestOutput.num_computed_tokens` field (+ docstring/repr) |
| `vllm/v1/metrics/loggers.py` | `FileStatLogger`: `num_computed_tokens` in `_CSV_COLUMNS` and the CSV/JSONL row |
| `vllm/entrypoints/openai/engine/protocol.py` | `PromptTokenUsageInfo.computed_tokens` |
| `vllm/entrypoints/openai/chat_completion/serving.py` | Populate `computed_tokens` in usage (streaming + non-streaming) |
