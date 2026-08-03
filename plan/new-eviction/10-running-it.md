# Running It — Deployment and Bring-Up

**What this is:** how to stand up the three components and the Redis between
them so a LangGraph workflow runs against the node-aware eviction policy.

**Read this first.** The unit tests pass as of 2026-08-03 (09 §1), but the
engine has never been *run* with the policy on — no end-to-end pass against a
real Redis, a real GPU and a real workflow. Treat the below as a first
bring-up, not a deployment of something known to work.

---

## 1. The topology

Four processes. Only three of them are servers.

```
┌─────────────────────┐         ┌──────────────────────────────┐
│  Workflow process   │         │  vLLM engine                 │
│  (LangGraph app)    │         │  VLLM_NODE_EVICTION_POLICY=1 │
│                     │  HTTP   │                              │
│  + kv-prediction-   │────────>│  subscriber thread           │
│    engine (library) │ vllm_x  │  ├─ psubscribe PROB|*        │
│                     │  args   │  └─ psubscribe HISTORY|*     │
└──────────┬──────────┘         └───────────┬──────────────────┘
           │                                │
           │ SET/PUBLISH                    │ writes JSONL
           │ INFO|<job>                     │ VLLM_REQUEST_STATS_DIR
           │ PROB|<job>                     ▼
           │                    ┌──────────────────────────────┐
           ▼                    │  finished_requests_engine*.  │
    ┌─────────────┐             │  jsonl                       │
    │    Redis    │<────────────┴───────────┬──────────────────┘
    │             │  SET/PUBLISH            │ tails
    │             │  HISTORY|<job>          │
    └─────────────┘             ┌───────────┴──────────────────┐
                                │  kv-trace-analyser (daemon)  │
                                └──────────────────────────────┘
```

- **kv-prediction-engine is a library, not a service.** It runs inside the
  workflow process, because it needs the workflow's state transitions. For
  LangGraph it is wrapped by `KVForecastSession` in the langgraph fork; for
  Open Deep Research the whole path is already wired — see §7.4.
- **kv-trace-analyser is a daemon** and it tails a file vLLM writes. That is
  the one hard co-location constraint — see §2.
- The engine never writes to Redis and never reads a socket on the request
  path. The subscriber is a daemon thread; the scheduler only ever reads an
  in-process snapshot.

### Keyspace ownership

| Key | Written by | Read by |
|---|---|---|
| `INFO\|<job_id>` | prediction engine | **nobody** — see §8 |
| `PROB\|<job_id>` | prediction engine | vLLM |
| `HISTORY\|<job_id>` | trace analyser | vLLM, prediction engine |

Both writers `SET` the key and then `PUBLISH` on a channel of the same name.
**You do not need `notify-keyspace-events`** — that open question in 03 §7 is
answered by the implementation choosing explicit publishes.

---

## 2. Hosting constraints

Two things constrain where these can run.

1. **The analyser must see vLLM's stats directory.** It tails
   `finished_requests_engine*.jsonl`. Same host, or a shared volume, or a
   sidecar container with the directory mounted read-only. There is no
   network transport for the trace in v1 — that is deliberate (04 §4), because
   the file is what makes the replay harness work.
2. **Everything else needs only Redis.** The workflow process and the engine
   never talk except over HTTP.

A single-box layout is fine and is what to start with: Redis, vLLM, and the
analyser on the GPU host; the workflow wherever it already runs.

Redis needs no persistence, no cluster, no auth for a first bring-up. Sizing
is trivial — a few KB per active job.

---

## 3. Step 0 — run the tests

105 tests in `tests/v1/core/node_eviction/`; all passing as of 2026-08-03 on
Python 3.10. Re-run after install, on the GPU box rather than a dev machine.

```bash
cd /path/to/vllm
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in
uv pip install -r requirements/node_eviction.txt

.venv/bin/python -m pytest tests/v1/core/node_eviction/ -v
.venv/bin/python -m pytest tests/v1/metrics/test_request_stats_logger.py -v
```

The external packages' tests have been run, but re-run them anyway since you
are installing fresh:

```bash
cd ~/Projects/kv-trace-analyser    && uv pip install -e '.[dev]' && pytest -q
cd ~/Projects/kv-prediction-engine && uv pip install -e '.[dev]' && pytest -q
```

Expect 23 and 25 passing respectively.

---

## 4. Step 1 — Redis

```bash
docker run -d --name kv-forecast-redis -p 6379:6379 redis:7-alpine
redis-cli ping   # PONG
```

That is the whole configuration.

---

## 5. Step 2 — the vLLM engine

`redis` is not in `common.txt`. Without it the policy still indexes blocks but
ranks nothing, and logs once — so install it, or you will be running a very
elaborate LRU.

```bash
uv pip install -r requirements/node_eviction.txt
```

Launch:

```bash
export VLLM_NODE_EVICTION_POLICY=1
export VLLM_NODE_EVICTION_REDIS_URL=redis://localhost:6379/0

# Required for the analyser — this is the trace transport.
export VLLM_REQUEST_STATS_DIR=/disk2/vibhav/vllm-logs/stats

# Optional, debugging only. Unbounded under pressure.
export VLLM_NODE_EVICTION_DECISION_LOG=/disk2/vibhav/vllm-logs/debug/evictions.jsonl

# Optional, all tunables. See §9.
export VLLM_NODE_EVICTION_CONFIG=/home/vibhav/Build/KVCOMM-VLLM/vllm/node_eviction.json

# Optional, OFF by default. Turns on prefetch origination (step 6): engine
# core publishes a want-list, the API-server process drains it and submits
# phantoms. Leave this OFF to measure the eviction ranking on its own —
# reordering is free, origination buys prefill work. See §10.
# export VLLM_NODE_EVICTION_PREFETCH_DRAIN=1
# export VLLM_NODE_EVICTION_PREFETCH_DRAIN_INTERVAL_S=1.0


LMCACHE_MP_FULL_HIT_ONLY=0 VLLM_USE_DEEP_GEMM=0 \
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 120000 \
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
  --gpu-memory-utilization 0.95 \
  --block-size 16 \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --enforce-eager \
  --seed 0 \
  --enable-prompt-tokens-details


Do **not** pass `--disable-log-stats`; the `FileStatLogger` will not run and
the analyser will have nothing to tail.
```

### Verify

```bash
# Sanity: send one request with identity attached, then check the stats file
# grew and carries the three new fields.
tail -1 $VLLM_REQUEST_STATS_DIR/finished_requests_engine*.jsonl \
  | python -c 'import json,sys; r=json.load(sys.stdin); print({k:r[k] for k in ("job_id","langgraph_node","call_type","arrival_ts","finish_ts")})'
```

If `call_type` is `null`, the workflow is not attaching identity to its
requests — go to §7.3.

The A/B baseline is `VLLM_NODE_EVICTION_POLICY=0`, which is byte-identical to
upstream: the controller is never built and every hook is a null check.

---

## 6. Step 3 — the trace analyser

```bash
cd ~/Projects/kv-trace-analyser
uv pip install -e '.[redis]'

kv-trace-analyser run \
  --stats-dir /disk2/vibhav/vllm-logs/stats \
  --redis-url redis://localhost:6379/0 \
  --checkpoint /disk2/vibhav/kv-trace-analyser/offsets.json
```

`--checkpoint` matters: without it a restart re-reads every file from byte 0.
Records are deduplicated by `request_id` so this is not a correctness bug, but
it wastes a startup.

Two other commands: `once` does a single pass and prints a report; `report`
does a pass with Redis disabled, which is how you inspect a captured trace
without publishing anything.

### Verify

```bash
kv-trace-analyser report --stats-dir /var/log/vllm/stats --json
redis-cli --scan --pattern 'HISTORY|*'
redis-cli GET 'HISTORY|<job_id>' | python -m json.tool
```

You are looking for `p_cold` in those rows. That number is the single most
valuable thing this component produces (04 §5.4) and it decides whether the
scorer's cost model is right or wrong by 60x.

Nothing publishes until a `(node, call_type)` has `min_samples` (default 2)
observations, and updates are debounced to 1/s per job.

---

## 7. Step 4 — the workflow

For the Open Deep Research benchmark this is **already wired**; see §7.4 for
what changed and where. For any other workflow, §7.1–7.3 are the contract.

```bash
cd ~/Projects/kv-prediction-engine
uv pip install -e '.[redis]'
export KV_FORECAST_REDIS_URL=redis://localhost:6379/0
export KV_PREDICTION_TRANSITION_STORE=/var/lib/kv-prediction/transitions.json
```

`KV_PREDICTION_TRANSITION_STORE` must be a **persistent path**. Transition
counts are a property of the workflow, not the run — they are what a fresh job
inherits instead of starting from uniform successors.

### 7.1 Declare the graph

`react_nodes` and `call_sites` cannot be inferred — LangGraph does not know
which nodes accumulate a prefix, and `call_type` **must** come from declared
structure rather than runtime content (05 §4). A label derived from message
count, tool output, or an LLM-call index drifts, re-keys the same blocks every
turn, and is worse than no policy at all.

```python
from kv_prediction_engine.adapters import graph_from_spec

graph = graph_from_spec({
    "supervisor": {"type": "non-react", "calls": ["plan"],
                   "next": ["research"]},
    "research":   {"type": "react",
                   "calls": ["summary", ("summary", "tavily")],
                   "next": ["supervisor"]},
})
```

> **`graph_from_langgraph` is usually the wrong tool, and fails silently.**
> It reads `compiled.get_graph()`, which sees only top-level nodes. Passing
> `xray=True` is not a fix: `Graph.extend(subgraph, prefix=name)` emits
> `f"{prefix}:{id}"`, so a subgraph node arrives as
> `research_supervisor:supervisor` while the runtime
> `metadata["langgraph_node"]` — the value actually sent to vLLM — is the bare
> `supervisor`. Every row then mis-joins with no error. Worse, a subgraph
> invoked *imperatively* rather than registered with `add_node` is invisible at
> any xray depth. Check both before trusting it; prefer a hand-written spec
> using the names the runtime emits, and reconcile it at startup.

### 7.2 Drive the lifecycle

```python
engine.start_job(job_id)              # writes INFO + initial PROB
engine.enter_node(job_id, "research") # republishes PROB
engine.start_tool(job_id, "research", "tavily")
engine.exit_node(job_id, "research")
engine.end_job(job_id)                # floors every key, saves transitions
```

`end_job` is what frees a finished job's blocks promptly rather than waiting
for the age term to decay them out. Put it in a `finally` — a job that died
still has to release its blocks.

Under LangGraph, `astream(stream_mode="tasks", subgraphs=True)` already emits
exactly these transitions, so the lifecycle needs no hook inside the Pregel
loop. Drive it from that stream rather than from the transition-prediction
tracker: the tracker is gated on a `transition_prediction` config, which is
the same switch that turns the prefix-prefetch worker on — and an eviction
experiment usually wants prefetch *off* (§7.4).

### 7.3 Attach identity to every LLM request

Three fields must ride on every request: `job_id`, `langgraph_node`,
`call_type`. Both of these reach `sampling_params.extra_args`:

```python
# (a) explicit — vllm_xargs is the documented field
extra_body={"vllm_xargs": {"job_id": ..., "langgraph_node": ..., "call_type": ...}}

# (b) top-level — what the ODR integration actually uses
extra_body={"job_id": ..., "langgraph_node": ..., "call_type": ...}
```

(b) works because `OpenAIBaseModel` sets `extra="allow"`, so unknown top-level
body fields land in `model_extra`, which `chat_completion/protocol.py` merges
into `extra_args`. Either is fine; (b) is convenient when a LangChain
integration already owns `extra_body`.

A request that arrives with no `call_type` is indexed under the empty label
and will not join `PROB`/`HISTORY` rows that carry one. Send it, or set
`use_call_type: false` in the vLLM config — **the two sides must agree.**

### 7.4 What is wired for Open Deep Research

| Where | Change |
|---|---|
| `langgraph-dev` `pregel/_vllm_agent.py` | `derive_call_type(node, tool)` — the label, beside the existing `derive_agent_id` |
| `langgraph-dev` `pregel/_kv_forecast.py` | *new* — `KVForecastSession`, a generic driver owning the `PredictionEngine`; topology arrives as a spec, so it knows nothing about ODR |
| `open_deep_research/llm_request_metadata.py` | sets `payload["call_type"]` beside the `job_id` / `langgraph_node` it already sent |
| `tests/run_evaluate_node_eviction.py` | env setup, `ODR_GRAPH_SPEC`, lifecycle on the existing task stream, startup reconcile, shutdown flush |

Run it:

```bash
export KV_FORECAST_REDIS_URL=redis://localhost:6379/0
python tests/run_evaluate_node_eviction.py --max-queries 6
```

Notes specific to this integration:

- **Prefetch is disabled by clearing `LANGGRAPH_VLLM_AGENT_ENABLE` and
  `LANGGRAPH_VLLM_AGENT_BASE_URL`** — *not* by `LANGGRAPH_ABLATION_MODE`,
  which nothing in langgraph or ODR reads and which only labels the LangSmith
  experiment. The real switch is `vllm_agent_enabled()`, true when
  `LANGGRAPH_VLLM_AGENT_ENABLE == "1"` **or** `LANGGRAPH_VLLM_AGENT_BASE_URL`
  is set, so both must be cleared, and after `load_dotenv`. Set
  `KV_EVICTION_DISABLE_PREFETCH=0` to leave prefetch on. The harness prints
  the real state at startup — trust that line, not the mode label.
- **`VLLM_REQUEST_STATS_DIR` is set by the harness** and defaults under its
  own metrics directory. Point the analyser (§6) at the value it prints.
- **The graph spec is hand-written** (9 nodes), for the reason in §7.1: ODR
  nests `supervisor`/`supervisor_tools` in a registered subgraph and invokes
  `researcher`/`researcher_tools`/`compress_research` imperatively, so no
  reflection route names them the way the runtime does. `research_supervisor`
  is declared with no call sites — it makes no LLM call but is a real
  transition state.
- **Drift is caught twice**: top-level nodes at startup, everything else from
  the task stream via `unknown_nodes`, printed at the end of the run. An
  undeclared node is not fatal — it gets no `PROB` row, so its blocks stay
  unscored in LRU order — but it silently drops out of the experiment.

Verify the spec builds before a long run:

```bash
python -c "
from kv_prediction_engine.adapters import graph_from_spec
import sys; sys.path.insert(0,'tests')
from run_evaluate_node_eviction import ODR_GRAPH_SPEC as S
g = graph_from_spec(S, name='open_deep_research')
print('nodes', len(g.nodes), 'keys', len(g.keys()))
"
```

---

## 8. Bring-up order and what to check

Start in this order and verify each before the next.

| # | Start | Check |
|---|---|---|
| 1 | Redis | `redis-cli ping` |
| 2 | vLLM | JSONL file appears; `call_type` non-null after a workflow request |
| 3 | Analyser | `HISTORY\|*` keys appear after ≥2 requests per node |
| 4 | Workflow | `PROB\|*` and `INFO\|*` appear |

The single most informative check is step 2's `call_type`. All three identity
fields travel the same path, so if `call_type` is null in the JSONL then the
`extra_body` plumbing is broken and **the policy is inert no matter what else
is running** — every block indexes under a null key and nothing is ranked:

```bash
tail -1 $VLLM_REQUEST_STATS_DIR/finished_requests_engine*.jsonl \
  | python -c 'import json,sys; r=json.load(sys.stdin); print({k:r[k] for k in ("job_id","langgraph_node","call_type")})'
```

Then confirm the forecast side is publishing, and that its labels match:

```bash
redis-cli --scan --pattern 'PROB|*'
redis-cli GET "PROB|<job_id>" | python -m json.tool | grep -E 'node_name|call_type'
```

Those `call_type` values must be **byte-identical** to the ones in the JSONL
above. A mismatch is the silent failure this whole contract exists to prevent:
both sides keep working, nothing errors, and the join simply returns nothing.

Then confirm the engine is actually consuming the forecast:

```python
# From anywhere with the KVCacheManager, e.g. a debug endpoint:
stats = kv_cache_manager.get_node_eviction_stats()
```

The fields that tell you it is alive: `snapshot_revision` climbing,
`index_keys` non-zero, `value_table_size` non-zero, `splices_total` climbing.
If `ticks_skipped_fresh` is the only thing moving, the cache is not under
pressure and the policy is correctly doing nothing — you need a smaller
`--gpu-memory-utilization` or a bigger workload to see it act.

For correctness rather than liveness, `regret_rate` and
`score_outcome_correlation` are the two that say whether the ranking is right
way up. A negative correlation means the splice is inverted and you are
evicting your best blocks.

**Cold start is real.** The first run of a workflow has no `HISTORY` and no
transition counts, so `time_to_next_call` falls back to
`default_llm_time_ms + default_tool_time_ms` and is marked low confidence,
and reach probabilities start uniform. Do a warm-up run before judging
anything — and keep `KV_PREDICTION_TRANSITION_STORE` on a path that survives
it, or every run is a cold start.

---

## 9. Tuning

All tunables live in one JSON file, so the replay harness can sweep them
without a rebuild:

```json
{
  "tau_ms": 30000,
  "splice_max_blocks": 256,
  "tick_period_ms": 250,
  "fresh_skip_threshold": 0,
  "staleness_cutoff_ms": 120000,
  "delta_l1_ms": 380,
  "delta_cold_ms": 11400,
  "speculative_default_ttl_ms": 30000,
  "use_call_type": true
}
```

`use_call_type` is the one entry here that is **not** a tuning knob — it is
half of a contract. `true` requires the workflow to send a `call_type`;
`false` keys at `(job_id, node)` and ignores it. The ODR integration sends one,
so it stays `true`. Changing it on one side only makes every key miss.

**Every other one is an unvalidated default.** Do not tune them by
intuition — the offline harness replays a captured trace deterministically:

```bash
python -m vllm.v1.core.node_eviction.replay --help
python -m vllm.v1.core.node_eviction.replay \
    --trace $VLLM_REQUEST_STATS_DIR/finished_requests_engine0_*.jsonl \
    --compare --forecast oracle
```

**If the oracle arm does not beat LRU, stop.** The enforcement path is broken
and no amount of prediction work will help. Only after that passes are
`--forecast history` (the realistic ceiling) and `--ablate` meaningful.

`--ablate` needs a **real captured trace**, not the built-in fixture. The
synthetic `walkthrough_trace()` holds `prob`, `E_miss` and block count
constant across its keys, so decay is the only term that varies and every
ablation reports as inert (09 §1). A trace with keys of differing prefix size
and differing recurrence is what makes the sweep able to answer anything.

`use_call_type: false` is the fallback if labels turn out not to be
deterministic across turns — it keys at `(job_id, node)` instead. It is a
one-line config change now and a rebuild later, which is why 05 §4 wants the
question answered before the engine work, not after.

---

## 10. What this does and does not do at this state

**Does:** reorders the free-block queue so low-value prefixes are evicted
first, protects prefetched prefixes with a decaying floor, and degrades to
plain LRU when Redis is down, the forecast is stale, or a key is unscored.

**Does not:**

- **Originate prefetches — unless `VLLM_NODE_EVICTION_PREFETCH_DRAIN=1`.**
  Step 6 now exists (09 §5): engine core publishes a want-list, and a
  background task in the API-server process drains it and submits phantoms
  through the existing submitter. It is **off by default**, and an eviction
  experiment usually wants it off (§7.4) so the ranking is measured on its
  own.

  With it off, the speculative path has exactly one producer:
  `/v1/agents/prefetch`, whose only caller is langgraph's
  `BackgroundVLLMAgentWorker` → `warm_agent_prefixes()`, gated on
  `vllm_agent_enabled()`. **Those phantoms carry no `job_id`, so they are
  invisible to the policy** — `node_key_for_request` returns None and nothing
  is stamped speculative. So with prefetch left off, the decaying floor,
  `speculative_floor_high` / `_low`, `speculative_default_ttl_ms` and
  `speculative_hard_drop_ttl_multiple` are all inert and
  `index_speculative_keys` / `speculative_waste` stay 0. The drainer's
  phantoms *do* carry identity, which is what makes the floor reachable at
  all — turning the drain on is the only way to exercise it.
- **Consult the value table per request.** `get_node_value` is implemented and
  has zero callers, so the O(1) read path from 02 §2 is dead code today. Note
  step 6 did **not** need it: the want-list reads the forecast directly, and
  the diagram's "evict others in order of least value except this" is served
  by the splice plus the speculative floor (09 §5.4).
- **Demote to L1 instead of destroying.** Step 7 is unbuilt and gated on
  measuring `p_cold` first.
- **Export Prometheus metrics.** Counters exist behind
  `get_node_eviction_stats()`; nothing publishes them.
- **Consume `INFO`.** `redis_info_pattern` is declared in `config.py` and
  never subscribed to. `list_call_types` was specified as the label-drift
  detector for the highest-risk element in the design (05 §3), and that
  detector is not wired up — so `call_type` drift will currently be silent.
  Until it is, `kv-trace-analyser report --json` after a run is the manual
  substitute: eyeball whether the emitted labels are the set you declared.
