# Prediction Engine — Implementation Spec

**Status:** spec, nothing implemented.
**Written:** 2026-07-31, against branch `vllm-v2` @ `68c7acb14`.
**Component:** lives with the LangGraph fork. Writes `INFO` (once per job) and
`PROB` (on state change) to Redis.
**Interface:** [`03-workflow-contract.md`](03-workflow-contract.md) is the
contract. **Read it first.**
**Companion:** [`04-trace-analyser.md`](04-trace-analyser.md) writes `HISTORY`
and owns all timing statistics. This component **consumes** those statistics and
must not recompute them.

`[verified]` = read at the cited line in this repo. `[required]` = must be true
for the design to work. `[open]` = decide during implementation.

---

## 1. Role, in one paragraph

This is the only component that knows the *shape of the workflow* — which node
is executing, what can follow it, and how far through the graph the job is.
vLLM cannot infer any of that, and the Trace Analyser sees only finished
requests after the fact. Its job is to answer, continuously and per job:
**"which node fires next, and how soon?"** — and to publish that as `PROB`.

```
LangGraph execution ──▶ Prediction Engine ──INFO/PROB──▶ Redis ──▶ vLLM policy
                                ▲
                        HISTORY │ (timings, from 04)
```

It is also the component that **emits `call_type` on the request** (§4), which
makes it the one place where the whole design can silently fail.

---

## 2. What it replaces, and why that matters

The previous attempt took `agent_probabilities` from the client on each request
(00 Part 2). It failed structurally, and both failures are requirements on this
component:

1. **A request without probabilities was invisible.** The policy silently fell
   through to LRU with no signal that it had. → Publish for **every** node in
   the job, not only the one currently running.
2. **Votes died with the request.** In-flight requests were the only source, so
   during an idle gap the policy was blind — precisely when a forecast is most
   valuable, because that is when eviction happens with nothing to protect the
   prefix. → `PROB` must stay valid **between** requests, describing what will
   happen next rather than what is happening now.

If this component only publishes while a request is in flight, it reproduces the
old design's failure exactly, through a different transport.

---

## 3. `INFO` — static, once per job

Written at job start, before the first generation request. Never rewritten
unless the graph changes.

| Field | Source | Notes |
|---|---|---|
| `node_name` | graph topology | **Must be byte-identical to the `langgraph_node` sent on the request** (03 §1). Same string constant, not two literals that happen to match today. |
| `type` | `react` / `non-react` | React nodes accumulate a growing prefix across turns; non-react hold one static prefix. Drives whether the policy protects a growing prefix or a fixed one (02 §9). |
| `list_call_types` | graph topology + tool bindings | The complete set of `call_type` values this node can emit. Lets the engine detect label drift instead of silently mis-joining. |

Emit one row per node in the graph, including nodes not yet reached. Cost is
negligible and it lets the policy score a prefetch target before the job has
ever visited it.

---

## 4. `call_type` — the single highest-risk element `[required]`

The engine keys its block index at `(job_id, node, call_type)` and joins to
`PROB`/`HISTORY` on it. The label is a flattened leaf name: `summary` for a
direct LLM call, `tavily:summary` for an LLM call nested inside a tool call.

> **The same physical prefix must always produce the same `call_type`.**

If a prefix is labelled `summary` on one turn and `tavily:summary` on the next,
the index re-keys the same blocks every turn, every key holds a fragment, and no
key accumulates enough history to be scored. **That is worse than no policy at
all** — it consumes memory and produces noise.

Requirements:

- **Derive the label from graph structure** — node identity plus tool binding —
  **never from runtime content.** A label derived from tool output, message
  count, or anything that varies per turn will drift.
- The same call site must produce the same label across turns, across jobs, and
  across process restarts.
- It must appear both in `extra_args` on the request (03 §1) and in the
  `PROB`/`HISTORY` rows, spelled identically.
- `list_call_types` in `INFO` must be the exact set of labels actually emitted.

**If a deterministic flattening is not achievable, say so before the engine work
starts.** The fallback — key at `(job_id, node)` and take `max` over call types —
is a one-line change now and a rebuild later. This is the question in 03 §7 item
2, and the conformance capture in 03 §8 answers it by observation.

---

## 5. `PROB` — dynamic, the forecast

One row per `(node_name, call_type)`, published on every workflow state change.

| Field | Meaning | Units (03 §4) |
|---|---|---|
| `prob` | P(this node+call_type fires again **at all**) — time-free, see §5.2 | float `[0, 1]` |
| `time_to_next_call` | expected time until it fires | ms |
| `update_ts` | when this row was computed | epoch ms, wall clock |

### 5.1 Publish for every node, always

Not just the current one, and not just its immediate successors. The policy
ranks blocks against each other, so a node with no row is indistinguishable from
a node that will never run again — both take the neutral default. Silence is not
"low probability"; it is "no information", and the engine deliberately treats it
as neutral rather than zero (01 §6, rule 2).

Nodes that genuinely will not run again should be published with a **low
probability**, which is the signal that actually frees their blocks.

### 5.2 What `prob` means — settle this before computing anything `[decided]`

Two readings are possible and they are not interchangeable:

| | Reading | Sums to 1? |
|---|---|---|
| (a) | P(this is the **next** node to fire) | yes, over nodes |
| (b) | P(this node fires **within horizon H**) | no |

**Neither. Use the time-free version:**

```
prob              = P(this (node, call_type) fires again at all, this job)
time_to_next_call = when, GIVEN that it fires
```

Reason: the engine's scorer already applies a time discount,
`decay(time_to_next_call)` (00 Part 6). If `prob` also encodes timing — reading
(b), or (a) implicitly — the discount is applied twice and near-term nodes get
squared preference. One field, one job:

| Field | Answers | Must NOT encode |
|---|---|---|
| `prob` | *whether* | when |
| `time_to_next_call` | *when* | whether |

Publishing (a) has a second failure: normalising over 20 nodes puts every value
near 0.05, and the engine ranks *blocks against each other*, so a distribution
that is flat by construction carries almost no signal.

### 5.2.1 Computing it — v1 proposal `[open]`

A Markov chain over graph nodes. Topology comes from LangGraph; edge weights
come from completed runs.

1. **Transition matrix.** Count observed `(from_node → to_node)` transitions
   across completed jobs **of the same graph**, row-normalise → `M`.
2. **Accumulate arrival probability** over a horizon of `H` hops from the
   current node `c`:

   ```
   p_reach(X) = min(1, Σ_{h=1..H} (M^h)[c][X])
   ```

3. **Split across call types** by their observed frequencies within that node
   (`INFO.list_call_types`, refined from `HISTORY`).
4. **Floor it.** Never publish 0 — a node that "cannot" fire has been
   mispredicted before, and the engine treats a missing row as neutral anyway
   (01 §6, Rule 2), so 0 is strictly worse than a small positive value.

**Why accumulate rather than multiply along a path.** A single-path product
breaks in exactly the two cases that dominate these graphs: multiple paths to
the same node, and **react self-loops**. For a react node `M[X][X]` is large, so
summing over hops makes probability *accumulate* — which is correct, since a
react node is usually the most likely thing to run next. A path product decays
it toward zero, i.e. it would mark the hottest node in the graph as the best
eviction candidate. Get this backwards and the policy is worse than LRU.

**Cold start.** Transition counts are a property of the *workflow*, not the run,
so a fresh job inherits them from prior runs of the same graph — same reasoning
as 04 §6. With no history at all, fall back to uniform transitions over
topological successors: still enough to separate reachable from unreachable.

**No self-vote for the running node.** The old branch gave each live request an
implicit `1.0` for its own agent. Unnecessary here: a running request's blocks
have `ref_cnt > 0`, so they are not in the free queue and cannot be evicted at
all. Structurally impossible rather than defended against.

Deliberately excluded from v1: per-job online learning, content-conditioned
prediction, anything needing a model. All of it can be added behind the same
`PROB` interface once the loop is measurable end to end.

**Risk worth stating plainly `[open]`.** If a workflow has one dominant path,
`p_reach` will be near-uniform across the few live nodes and `prob` carries
almost no discriminating signal — leaving `time_to_next_call` and the decay term
doing all the work. That is not a failure, but it changes where effort should
go. The engine-side offline replay harness (00 Part 6) answers it against
captured traces without running anything.

### 5.3 Computing `time_to_next_call`

Do **not** measure this — 04 already does, from data only vLLM has.

```
time_to_next_call(next_node) ≈ Σ over the predicted path:
      HISTORY[node, call_type].time_taken            (LLM time)
    + HISTORY[node, call_type].tool_execution_time   (the inferred gap)
```

Read those from `HISTORY` (04 §5). For a node with no history, use a configured
default and mark low confidence — do not publish a fabricated precise number.

This field is load-bearing beyond scoring: it is the **TTL of the speculative
protection floor** for prefetched blocks (02 §5). Too short and a correct
prefetch loses protection before it is used; too long and a wrong prediction
protects garbage for the whole window. Being systematically wrong here is worse
than being noisy.

### 5.4 `update_ts` — set it on every write, always

The engine's staleness gate compares it against a cutoff and falls back to LRU
when the row is too old (01 §2). A row that never updates its timestamp defeats
the one mechanism that makes a missed pub/sub message degrade safely instead of
acting on stale data.

Set it to when the row was **computed**, not when it was written.

---

## 6. When to publish

| Trigger | Publish |
|---|---|
| Job start | full `INFO` + initial `PROB` for all nodes |
| Node entry | `PROB` — the distribution just shifted |
| Node exit | `PROB` — successors are now imminent |
| Tool call start | `PROB` — `time_to_next_call` for the resuming node is now known more precisely |
| Job end | final `PROB` with all probabilities at the floor, so blocks decay out |
| Periodic | **no** — publish on change only |

**Publish before the request, not after.** The value of the forecast is that it
reaches the engine while there is still time to act. A `PROB` update published
after the generation request has been submitted describes a decision already
made.

The job-end publish matters more than it looks: without it, a finished job's
blocks keep whatever score they last had, and the engine's age term has to decay
them out slowly. An explicit floor at job end frees them immediately.

---

## 7. Redis write discipline

Identical to 04 §7, and the same top requirement:

- **Only this component writes `INFO` and `PROB`.** `HISTORY` belongs to 04.
- **`PUBLISH` after every `SET`** (03 §3.1). Without it the policy is frozen at
  its seed snapshot — no error, no crash, just a forecast that never updates.
- **Publish on change only.**
- **Never block the workflow on Redis.** A Redis failure must not stall
  execution: fire-and-forget with a bounded queue, drop oldest under pressure.
  A dropped `PROB` update costs a little accuracy; a stalled workflow costs a
  request.
- Key naming, payload shape and units per 03 §3.2–§4 — **pin these before
  writing code**, they are questions 3–6 in 03 §7.

---

## 8. Configuration

| Setting | Purpose |
|---|---|
| Redis endpoint / auth / db | shared with 04 |
| floor probability | the "won't run" value; must be > 0 (§5.1) |
| max hops before floor | §5.2 item 3 |
| default `time_to_next_call` | for nodes with no history (§5.3) |
| transition-count store | where cross-job frequencies persist |
| publish queue depth | §7 backpressure |

---

## 9. Tests

No Redis needed (fake client), no vLLM, no GPU:

1. **A linear graph** A→B→C: at A, `prob(B)` **and** `prob(C)` are both ≈1 —
   both certainly fire. They are separated by `time_to_next_call`, not by
   `prob`. A test asserting `prob(C) < prob(B)` is asserting the *old*,
   time-encoding semantics and must fail (§5.2).
2. **An exclusive terminal branch** with observed 70/30 frequencies reproduces
   0.7/0.3. If the branches reconverge, both successors are ≈1 — also correct,
   and worth a separate case so nobody "fixes" it.
3. **A react self-loop** accumulates rather than decays: over `H` hops its
   probability rises toward 1. Asserting the opposite is the sign error that
   makes the hottest node in the graph the best eviction candidate.
4. **Job end** publishes floors for every node.
5. **`call_type` determinism** — the property test that matters: run the same
   graph path twice and assert an identical label sequence. Then assert every
   label emitted appears in `INFO.list_call_types`.
6. **Label agreement** — the `call_type` in `extra_args` matches the one in the
   `PROB` row for the same call. This is the failure that is otherwise
   completely silent (03 §5).
7. **Missing history** → a default `time_to_next_call` and a low-confidence
   marker, never a fabricated precise value.
8. **Redis down** → the workflow completes normally.

Test 6 is the one to write first. Everything else degrades gracefully; a label
mismatch produces a policy that indexes blocks, receives forecasts, and never
connects the two — with a perfectly healthy-looking log.

---

## 10. Build order

| Step | What | Blocked by |
|---|---|---|
| **1** | Emit `call_type` in `extra_args` on every request, deterministically (§4) | the flattening rule |
| **2** | `INFO` at job start (§3) | key naming, endpoint |
| **3** | `PROB` on state change, graph reachability only, flat probabilities (§5.2 item 1) | step 2 |
| **4** | Transition frequencies + distance decay (§5.2 items 2–4) | a completed-job corpus |
| **5** | `time_to_next_call` from `HISTORY` (§5.3) | 04 publishing `HISTORY` |

Step 1 is the prerequisite for everything in the whole design, on both sides —
the engine cannot key its index without it, and it is also what 03 §8's
conformance capture is designed to inspect. Do it first, and capture the output
before building anything on top of it.
