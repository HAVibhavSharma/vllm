# 13 — A separate queue for phantom prefetches

## Why

A phantom prefetch (`/v1/agents/prefetch`, `prefetch_only=True`) warms a
prefix that a real request is predicted to need next. Until now it entered the
scheduler exactly like a real request:

- **Queue order.** It sat in the same waiting queue. Under
  `--scheduling-policy priority` it was submitted with the default priority 0,
  which outranks every replayed real call except the first.
- **Running slots.** A seed that prefills counted toward `max_num_seqs`.
- **Memory.** A loading phantom takes all its blocks up front and holds them
  until LMCache's copy finishes. While it is loading it is parked in the
  waiting queue as `WAITING_FOR_REMOTE_KVS`, not in `running`. When memory
  ran out, two things could happen to a real request:
  - a new one waited, because its allocation failed (`schedule`, the
    `new_blocks is None` break);
  - a running one needing its next block caused the preemption of a *real*
    running request. Preemption only picks from `running`, so the phantom
    could never be the one chosen.
- **Compute.** A seed's prefill shares the step's token budget, and a real
  prefill beside it ran at about half speed.

The goal is that a phantom never makes a real request wait, never takes its
slot, budget or memory, and never causes it to be preempted, while still
warming as early as that allows.

## The design

Two independent switches, so each change can be measured on its own:

| Switch | What it turns on |
|---|---|
| `VLLM_PHANTOM_QUEUE=1` | The phantom queue: its own queue, admission rules, slot cap, ordering and expiry (parts 1–5). |
| `VLLM_NODE_EVICTION_BASIC=1` | The *basic* node-aware eviction mode: warmed blocks are protected from eviction until there is contention (part 6). Requires the phantom queue. |

### 1. Its own queue

Phantoms go into `Scheduler.phantom_waiting`, never into `waiting`. The
scheduler looks at that queue only at the start of a step, after deciding
the real requests are not affected. An admitted phantom is moved into
`waiting` for that one step. There it goes through the ordinary admission
path: cache lookup, LMCache lookup, allocation, the async load. Any phantom
still unadmitted at the end of the step (a seed deferred for want of an idle
step) goes back to `phantom_waiting`. So the real queues hold a phantom only
while it is actually loading or prefilling.

Because phantoms never sit in the real queue, the priority policy orders
real requests only. The submitter also gives phantoms the lowest priority
whenever the engine runs the priority policy, so the old shared-queue path
(phantom queue off) no longer lets them jump ahead either.

### 2. When a phantom may start (rule 1)

A phantom is admitted at the start of a step only if all of the following
hold:

1. **No real request is waiting.** Nothing real is in `waiting` or
   `skipped_waiting` except requests already admitted and loading their own
   KV (`WAITING_FOR_REMOTE_KVS`).
2. **The phantom slot is free.** At most `VLLM_PHANTOM_QUEUE_MAX_LOADING`
   phantoms hold blocks at once (default 1). That counts:
   - phantoms loading (`WAITING_FOR_REMOTE_KVS`);
   - seeds prefilling (`running`);
   - finished phantoms whose blocks still wait for LMCache to confirm its
     save.

   Phantoms never count toward real requests' `max_num_seqs`, and none is
   admitted while `running` is full.
3. **Memory stays spare for live requests.** If `need` is an upper bound on
   the phantom's blocks (its whole prompt, ignoring any cache hit), the
   phantom is admitted only when
   `free_blocks − need ≥ VLLM_PHANTOM_QUEUE_RESERVE_BLOCKS × live_requests`.
   - `live_requests` counts real running requests plus real requests loading
     their own KV.
   - The reserve defaults to 4 blocks, about what one request generates while
     a phantom loads.
   - It never reads `max_tokens`: in replay that comes from the trace, and
     the rule must behave the same in replay and live.

A phantom that fails a check stays queued, and the reason is counted.

### 3. Seeds compute only in idle steps

A seed is a phantom allowed to prefill on an LMCache miss. With the phantom
queue on, seeds always follow `VLLM_PREFETCH_PREFILL_IDLE_ONLY`: they
prefill only in steps where no real request needs prefill, and pause
mid-prefill when one arrives. `VLLM_PREFETCH_PREFILL_STEP_TOKENS` still caps
their per-step chunk.

### 4. Order

Newest first. The most recent prediction is the likeliest to match the next
real request, and the oldest is the likeliest to be stale.

A phantom that was admitted but did not get its blocks goes back to the
*end* of the line. That covers a seed deferred for want of an idle step, or
an LMCache lookup still pending. Otherwise a seed deferred step after step
would be the newest every time, and no other phantom would get a turn. Its
expiry still counts from when it first arrived.

### 5. Dropping stale phantoms

A queued phantom is finished without running, as a miss, in two cases:

- **Expiry:** it has waited `VLLM_PHANTOM_QUEUE_TIMEOUT_S` (default 30 s).
- **Arrival:** a real request arrives whose prompt starts with the phantom's
  tokens, under the same cache salt. That request will load or compute the
  prefix itself, so the warm is no longer worth anything.

Only queued phantoms are dropped. One already loading is left to finish.

### 6. Basic eviction: protect warmed blocks until contention (rule 2)

When a phantom finishes, its cached blocks are freed as usual, but each is
marked *warmed* until `VLLM_NODE_EVICTION_WARM_PROTECT_S` from then (default
10 s).

- **Eviction order.** Allocation takes ordinary free blocks first and skips
  warmed ones. It takes a warmed block, oldest first, only when no ordinary
  free block is left. Under contention the real request wins, and the
  prefetch is the one lost.
- **End of protection.** A real request hitting a warmed block (a prefix hit,
  `touch`) ends its protection. From then on it follows that request's normal
  lifecycle, and becomes an ordinary LRU block when the request frees it.
- **Wrong predictions.** A warmed block nobody uses becomes ordinary after
  the 10 s.

This is a mode of the node-eviction controller (`node_eviction="basic"` in
pitlane), exclusive with observe mode and with the scored splice. Its
`kv_hbm` lines say `variant=basic`.

## What is counted

`SchedulerStats.phantom_queue` carries cumulative counters, and the
scheduler timeline (`scheduler_engine*.jsonl`) writes them on every sample:

- `queued` (current depth), `admitted`, `expired`, `dropped_on_arrival`;
- `held_real_waiting`, `held_slot`, `held_memory`: steps a phantom was
  eligible but held back, by reason;
- `real_blocked_with_phantom` and `real_preempted_with_phantom`: a real
  request's allocation failed, or a real request was preempted, while a
  phantom held blocks. The target is 0.
- With basic eviction: `warm_protected`, `warm_used`, `warm_expired`,
  `warm_taken_under_contention`.

## How it is shown not to hurt

The pitlane ladder, all with replay priority and cached search:

| Arm | Adds |
|---|---|
| `ours_full_lru_idle_prio` | today's prefetch, phantoms in the shared queue at the lowest priority |
| `ours_full_lru_pq_prio` | the phantom queue |
| `ours_full_basic_prio` | basic eviction on top |

Each is compared against `ours_none_cached_prio` on the same questions and
reps. Real-request TTFT and queue time must be no worse, and the counters
above must show no real request blocked or preempted by a phantom.

## Not covered here

- **The LMCache server's GPU kernels.** They run in another process, which
  outside MPS takes turns with vLLM on the GPU. They are measured separately
  with `PITLANE_NSYS=1` (`pitlane/nsys_overlap.py`). That cost applies to
  every store and load, not only phantoms.
- **Disaggregated (P/D) arms.** Untested with the phantom queue.
