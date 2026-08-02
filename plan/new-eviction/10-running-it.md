# Running It — Deployment and Bring-Up

**What this is:** how to stand up the three components and the Redis between
them so a LangGraph workflow runs against the node-aware eviction policy.

**Read this first.** Nothing in the vLLM changeset has ever been executed
(09 §1). The bring-up below is written against the code as it exists, but
step 0 is not optional and the whole thing should be treated as a first
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
  workflow process, because it needs the workflow's state transitions.
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

Blocking. 102 tests exist in `tests/v1/core/node_eviction/` and none has ever
been executed. Do this on the GPU box, not a dev machine.

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
export VLLM_REQUEST_STATS_DIR=/var/log/vllm/stats

# Optional, debugging only. Unbounded under pressure.
export VLLM_NODE_EVICTION_DECISION_LOG=/var/log/vllm/evictions.jsonl

# Optional, all tunables. See §9.
export VLLM_NODE_EVICTION_CONFIG=/etc/vllm/node_eviction.json

mkdir -p "$VLLM_REQUEST_STATS_DIR"
vllm serve <model> --host 0.0.0.0 --port 8000
```

Do **not** pass `--disable-log-stats`; the `FileStatLogger` will not run and
the analyser will have nothing to tail.

### Verify

```bash
# Sanity: send one request with identity attached, then check the stats file
# grew and carries the three new fields.
tail -1 $VLLM_REQUEST_STATS_DIR/finished_requests_engine*.jsonl \
  | python -c 'import json,sys; r=json.load(sys.stdin); print({k:r[k] for k in ("job_id","langgraph_node","call_type","arrival_ts","finish_ts")})'
```

If `call_type` is `null`, the workflow is not sending `vllm_xargs` — go to §7.

The A/B baseline is `VLLM_NODE_EVICTION_POLICY=0`, which is byte-identical to
upstream: the controller is never built and every hook is a null check.

---

## 6. Step 3 — the trace analyser

```bash
cd ~/Projects/kv-trace-analyser
uv pip install -e '.[redis]'

kv-trace-analyser run \
  --stats-dir /var/log/vllm/stats \
  --redis-url redis://localhost:6379/0 \
  --checkpoint /var/lib/kv-trace-analyser/offsets.json
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

This is the only part that needs code you write.

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

Either from a compiled LangGraph:

```python
from kv_prediction_engine import PredictionEngine, PredictionConfig
from kv_prediction_engine.adapters import graph_from_langgraph
from kv_prediction_engine.publisher import build_client

graph = graph_from_langgraph(
    compiled_graph,
    react_nodes={"research"},
    call_sites={
        "supervisor": ["plan"],
        "research":   ["summary", ("summary", "tavily")],
    },
)
```

or, with no framework:

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

`react_nodes` and `call_sites` cannot be inferred — LangGraph does not know
which nodes accumulate a prefix, and `call_type` **must** come from declared
structure rather than runtime content (05 §4). A label derived from message
count or tool output drifts, re-keys the same blocks every turn, and is worse
than no policy at all.

### 7.2 Drive the lifecycle

```python
config = PredictionConfig.from_env()
engine = PredictionEngine(graph, config, client=build_client(config.redis_url))

engine.start_job(job_id)              # writes INFO + initial PROB
...
engine.enter_node(job_id, "research") # republishes PROB
engine.start_tool(job_id, "research", "tavily")
engine.exit_node(job_id, "research")
...
engine.end_job(job_id)                # floors every key, saves transitions
```

`end_job` is what frees a finished job's blocks promptly rather than waiting
for the age term to decay them out. Put it in a `finally`.

### 7.3 Attach identity to every LLM request

```python
extra = engine.call_context(job_id, node="research", call="summary", tool="tavily")
# -> {"job_id": ..., "langgraph_node": "research", "call_type": "tavily:summary"}

client.chat.completions.create(
    model=...,
    messages=...,
    extra_body={"vllm_xargs": extra},
)
```

`vllm_xargs` is what becomes `sampling_params.extra_args` inside the engine,
which is where both the index key and the trace fields are read from.

`call_context` raises `UnknownCallSite` for an undeclared call site. That is
deliberate: the alternative is a label that reaches vLLM with no matching
`PROB` row and joins to nothing, silently.

---

## 8. Bring-up order and what to check

Start in this order and verify each before the next.

| # | Start | Check |
|---|---|---|
| 1 | Redis | `redis-cli ping` |
| 2 | vLLM | JSONL file appears; `call_type` non-null after a workflow request |
| 3 | Analyser | `HISTORY\|*` keys appear after ≥2 requests per node |
| 4 | Workflow | `PROB\|*` and `INFO\|*` appear |

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
anything.

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

**Every one of these is an unvalidated default.** Do not tune them by
intuition — the offline harness replays a captured trace deterministically:

```bash
python -m vllm.v1.core.node_eviction.replay --help
python -m vllm.v1.core.node_eviction.replay \
    --trace $VLLM_REQUEST_STATS_DIR/finished_requests_engine0_*.jsonl \
    --compare --forecast oracle
```

**If the oracle arm does not beat LRU, stop.** The enforcement path is broken
and no amount of prediction work will help. Only after that passes are
`--forecast history` (the realistic ceiling) and `--ablate` (which terms are
load-bearing) meaningful.

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

- **Originate prefetches.** Step 6 is unbuilt, so the speculative path is only
  exercised by phantom requests the existing submitter already sends.
- **Consult the value table per request.** `get_node_value` is implemented and
  has zero callers, so the O(1) read path from 02 §2 is dead code today.
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
