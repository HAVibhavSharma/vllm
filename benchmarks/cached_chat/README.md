# cached_chat — manual KV-cache reuse

`/v1/cached_chat/completions` is wire-compatible with `/v1/chat/completions`
and implements its own end-to-end KV-cache reuse pipeline. It does **not**
call into vLLM's automatic prefix cache or the KV connector framework. The
fingerprint store, capture trigger, and inject step are all our code; the
two extension points we use (worker-extension-cls and a small hook in
`gpu_model_runner`) exist precisely so user code can do this kind of
thing.

---

## 1. Required server flags

```
vllm serve <model> \
    --no-enable-prefix-caching \
    --worker-extension-cls \
        vllm.entrypoints.openai.cached_chat.worker_ext.ManualKVWorkerExtension
```

- `--no-enable-prefix-caching` — without this, the engine's automatic
  prefix cache shadows our manual reuse and you can't tell which path
  actually saved the prefill. Disable it.
- `--worker-extension-cls …ManualKVWorkerExtension` — installs our
  `manual_kv_drop` / `manual_kv_install` / `manual_kv_block_size` RPC
  methods on the worker. Without it, `/v1/cached_chat/completions` will
  log a warning on first request and fall through to plain chat.

Single GPU (TP=1, PP=1) only.

---

## 2. The two extra-args we route through `vllm_xargs`

The API server stuffs three keys into `request.vllm_xargs` (which becomes
`sampling_params.extra_args` on the engine side):

| Key                                | Type | Meaning                                                       |
| ---------------------------------- | ---- | ------------------------------------------------------------- |
| `manual_kv_handle`                 | str  | The handle to capture into / inject from.                      |
| `manual_kv_prefilled_tokens`       | int  | If > 0: scheduler treats this many tokens as already computed. |
| `manual_kv_capture_after_prefill`  | bool | If true: worker captures prefix KV once prefill completes.     |

A *miss* sends `(handle, capture=True)`. A *hit* sends `(handle,
prefilled_tokens=N)`. They're independent flags; capture and inject can
in principle co-occur but in this pipeline we only set one per request.

---

## 3. End-to-end lifecycle

### 3.1 First request for a given prompt (MISS, capture)

```
client                 API server                     engine                 worker
  |                       |                              |                      |
  |-- POST cached_chat -->|                              |                      |
  |                       | fp = sha256(messages, …)     |                      |
  |                       | entry = store.reserve(fp)    |                      |
  |                       |   # generates fresh handle   |                      |
  |                       | xargs[manual_kv_handle] = h  |                      |
  |                       | xargs[capture_after_prefill] |                      |
  |                       |   = True                     |                      |
  |                       |--------- generate() -------->|                      |
  |                       |                              |-- schedule -------->|
  |                       |                              |  Request added,      |
  |                       |                              |  num_computed=0      |
  |                       |                              |---execute_model---->|
  |                       |                              |  _update_states:    |
  |                       |                              |    new_req sees     |
  |                       |                              |    capture flag →   |
  |                       |                              |    record           |
  |                       |                              |    pending_capture  |
  |                       |                              |    [req_id] = (h,L) |
  |                       |                              |  forward (prefill+  |
  |                       |                              |    1st decode token)|
  |                       |                              |  next iteration:    |
  |                       |                              |  _update_states:    |
  |                       |                              |    sees             |
  |                       |                              |    num_computed >=  |
  |                       |                              |    L  → capture     |
  |                       |                              |    blocks 0..L/B    |
  |                       |                              |    into pinned host |
  |                       |                              |  generate rest      |
  |                       |<------- response ------------|                      |
  |                       | store.populate(fp, captured) |                      |
  |<------- 200 ----------|                              |                      |
```

### 3.2 Second request for the same prompt (HIT, inject)

```
client                 API server                     engine                 worker
  |                       |                              |                      |
  |-- POST cached_chat -->|                              |                      |
  |                       | fp = sha256(messages, …)     |                      |
  |                       | entry = store.get(fp)        |                      |
  |                       |   populated, prefix_len=L    |                      |
  |                       | xargs[manual_kv_handle] = h  |                      |
  |                       | xargs[prefilled_tokens] = L  |                      |
  |                       |--------- generate() -------->|                      |
  |                       |                              | scheduler sees       |
  |                       |                              |   prefilled_tokens=L |
  |                       |                              |   in extra_args →    |
  |                       |                              |   num_external_      |
  |                       |                              |   computed_tokens=L  |
  |                       |                              |   allocate_slots     |
  |                       |                              |   reserves blocks    |
  |                       |                              |   for L tokens but   |
  |                       |                              |   doesn't prefill.   |
  |                       |                              |---execute_model---->|
  |                       |                              | _update_states:     |
  |                       |                              |   new_req sees      |
  |                       |                              |   handle + prefilled|
  |                       |                              |   → copy blob       |
  |                       |                              |   pinned host →     |
  |                       |                              |   GPU into newly    |
  |                       |                              |   allocated blocks  |
  |                       |                              | forward             |
  |                       |                              |   (only the suffix  |
  |                       |                              |    is prefilled,    |
  |                       |                              |    rest is decode)  |
  |                       |<------- response ------------|                      |
  |<------- 200 ----------|                              |                      |
```

---

## 4. The actual code points

### 4.1 New files

```
vllm/entrypoints/openai/cached_chat/
    __init__.py
    protocol.py        # request/response aliases of ChatCompletion*
    store.py           # ManualKVStore (API-side) — fingerprint -> handle
    serving.py         # OpenAIServingCachedChat — the orchestration above
    api_router.py      # POST /v1/cached_chat/completions
    worker_ext.py      # ManualKVWorkerExtension + _ManualKVRunnerState
                       #   - process_pending(): inject + capture-when-ready
                       #   - manual_kv_drop()   : RPC for eviction
                       #   - manual_kv_install(): RPC for block_size + lazy init

benchmarks/cached_chat/
    test_cached_chat.py
    README.md          # this file
```

### 4.2 Surgical engine edits

Two small edits, both gated on a new request flag so they're no-ops when
the manual pipeline isn't in use.

`vllm/v1/core/sched/scheduler.py` — inside the `if request.num_computed_tokens
== 0:` branch, after the local prefix cache and the optional KVConnector
match, we honor `sampling_params.extra_args["manual_kv_prefilled_tokens"]`
by treating it as `num_external_computed_tokens`. The existing
`allocate_slots` path then reserves blocks for those tokens without
issuing prefill.

`vllm/v1/worker/gpu_model_runner.py` — at the end of `_update_states`,
right after `self.input_batch.refresh_metadata()`, we call
`self._manual_kv_state.process_pending(scheduler_output)` if a state has
been attached. The state is lazily created the first time the worker
extension's RPC methods touch it.

That's the whole engine surface area — six logical lines of code that
read a known sentinel from the request and delegate the work.

---

## 5. Running the test

```
python benchmarks/cached_chat/test_cached_chat.py \
    --base-url http://<host>:8000 \
    --model <served-model-name> \
    --repeats 3 --warmup --strict --strict-speedup 1.5
```

Expected output on a healthy run:

```
[call 1] expect=MISS (capture)     latency=0.41s prompt_tokens=92 completion_tokens=64
[call 2] expect=HIT  (inject, #1)  latency=0.16s prompt_tokens=92 completion_tokens=64
[call 3] expect=HIT  (inject, #2)  latency=0.15s prompt_tokens=92 completion_tokens=64
STRICT OK: call 1 / call 2 = 2.56x
```

If the speedup ratio is too low, the most likely culprits are (in order):

1. `--no-enable-prefix-caching` was not set, so the engine reused KV
   automatically and call-1 was already cheap.
2. `--worker-extension-cls …ManualKVWorkerExtension` was not set, so the
   capture/inject hooks never installed and the manual pipeline silently
   fell through.
3. The prompt is shorter than one block (default `block_size=16` for
   most attention backends; see `--block-size`). With nothing captured,
   nothing can be reused. Use a longer prompt.

Server logs are the source of truth. Look for:

```
[manual-kv] capture OK handle=… req_id=… blocks=N tokens=N*B layers=L
[manual-kv] inject OK  handle=… req_id=… blocks=N tokens=N*B
[cached_chat] HIT  fp=… handle=… prefix_len=N hits=K
[cached_chat] MISS fp=… handle=… — capture-after-prefill armed
[cached_chat] POPULATED fp=… prefix_len=N (prompt_tokens=P, block_size=B)
```

---

## 6. Limitations / TODO

- TP=1 only. A TP>1 deployment would shard each layer across ranks; the
  current pipeline assumes one rank's view of `kv_caches` is the whole
  tensor.
- PP=1 only. Pipeline parallelism would need per-stage capture/inject.
- Standard attention only (no Mamba / SSM cache shapes).
- Eviction is best-effort: the API-side LRU calls `manual_kv_drop` but
  doesn't await it; in steady state it's fine but a burst of misses can
  briefly leak pinned memory on the worker.
- Coarse fingerprinting: we hash the rendered request body, not the
  tokenised prompt at block boundaries. Two requests that share only
  the first K blocks won't currently share KV — they'd get different
  fingerprints. Switching to a per-block-token hash is a follow-up.
- Streaming requests work, but the populate-store-on-success step keys
  off `ChatCompletionResponse`; for the streaming path the entry stays
  reserved-but-unpopulated until the next non-streaming call resolves
  the same fingerprint, which is a small footgun. (Capture itself still
  happens on the worker either way; the API-side bookkeeping just lags.)
