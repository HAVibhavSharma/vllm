# From Eviction Comparator to KV Cache Controller

**Status:** scope expansion agreed from the architecture diagram; nothing
implemented.
**Written:** 2026-07-30, against branch `vllm-v2` @ `68c7acb14`.
**Reads with:** [`00-design-checkpoint.md`](00-design-checkpoint.md) (where
eviction lives, what the old attempt did) and
[`01-redis-integration.md`](01-redis-integration.md) (Redis transport, index
keys, enforcement rules). **§10 of this doc is the canonical build order** and
supersedes 01 §9. [`03-workflow-contract.md`](03-workflow-contract.md) is the
outward-facing interface for the boxes in §1 that live outside vLLM.

`[verified]` = read at the cited line. `[decided]` = settled. `[open]` = needs an
answer.

---

## 1. What the diagram adds

00 and 01 designed a **comparator**: something that answers "which block is
least valuable" and re-orders the free queue. The diagram specifies a
**controller**: the KV cache manager "manages both Admission/Prefetch and
Eviction" and emits "a list of instructions to adjust the kv cache in HBM to
accommodate the incoming requests."

Re-ordering the free queue remains the correct *enforcement primitive* for
eviction, but it is now one instruction type among several.

### The system as drawn, in vLLM terms

| Diagram box | Where it actually lives |
|---|---|
| Workflow + Prediction Engine | Outside vLLM (LangGraph fork). Writes `INFO`/`PROB` to Redis. Spec: [`05-prediction-engine.md`](05-prediction-engine.md). |
| Trace Analyser | Outside vLLM. Writes `HISTORY` to Redis — but **its timings come from vLLM's own per-request records** (`FileStatLogger`), not from independent instrumentation. Tool time is the inferred gap between consecutive requests of a job. Spec: [`04-trace-analyser.md`](04-trace-analyser.md). |
| Redis | The **only** ingress into the engine (§7). |
| **KV CACHE MANAGER** | New module in the **engine-core process**, driven from `KVCacheManager.new_step_starts()` (`kv_cache_manager.py:539` ← `scheduler.py:459`). `[verified]` |
| vLLM Engine | Consults the manager per request via an O(1) lookup (§2). |

---

## 2. The critical-path reconciliation `[decided]`

The diagram says both:

> "PER Request the Engine should consult the manager"

and

> "This system should not be an overhead as it comes in the critical path, it
> should be fast and lightweight, if there is a miss go to default"

Those are only compatible one way: **the tick computes, the request reads.**

| | Runs where | Cost budget | Does |
|---|---|---|---|
| **Subscriber thread** | daemon thread, engine-core process | unbounded (off critical path) | Redis pub/sub (+ `MGET` seed on connect/reconnect), decode, build immutable snapshot (01 §2) |
| **Tick** | scheduler loop, wall-clock paced | O(tracked log K) + O(K), skippable | score every tracked key, write a plain value table, emit instructions, re-splice the worst K free blocks to the head (01 §6) |
| **Per-request consult** | scheduler loop, every request | **O(1), no allocation** | one dict lookup of a precomputed value; miss → neutral default |

The per-request path must never compute a score, touch Redis, or walk the free
queue. That is also what makes "if there is a miss go to default" true by
construction.

---

## 3. The instruction set, against real vLLM primitives

"A list of instructions" needs to map onto things vLLM can actually do. It
doesn't, entirely — two of the five have no primitive today.

| Instruction | Primitive | Status |
|---|---|---|
| **REORDER** — change eviction order | bounded free-queue re-splice on the tick: `remove` O(1) (`kv_cache_utils.py:284`) + **new** `appendleft` (01 §6.1), candidates taken from the index | **Needs `appendleft`** (~5 lines, additive), then exists. 01 §6. `[verified]` |
| **ADMIT / EVICT-FOR-ADMISSION** — free space for an incoming request | falls out of REORDER: `get_new_blocks` pops the head (`block_pool.py:336`) | **Exists**, no new code. `[verified]` |
| **PROTECT** — keep a block against eviction | no hard primitive; only `ref_cnt > 0` truly protects a block. Served instead by a **decaying score floor** on speculative entries. | **Designed, soft.** §5. |
| **PREFETCH** — pull a prefix into HBM | phantom request via `PhantomPrefetchSubmitter` | **Exists but in the wrong process.** §4. |
| **DEMOTE** — move a block to LMCache instead of destroying it | none. Eviction drops the block. | **Missing.** §9. |

---

## 4. Prefetch origination is in the wrong process `[open]`

The diagram asks: *"Can we not push the prefetching decision to the KV CACHE
MANAGER?"* The decision, yes. The *submission*, not directly.

`PhantomPrefetchSubmitter` holds an `EngineClient` and issues
`self._engine_client.generate(prompt, params, request_id)`
(`vllm/v1/agent_prefetch/submitter.py:61, :141`) `[verified]`. That is the
**front-end (API-server) process**. The KV cache manager lives in engine core.
Engine core has no channel for originating a request — `EngineCoreOutputs`
carries per-request output, not requests.

Three ways out:

- **(a) Split decide from submit** — manager decides and publishes a *want-list*;
  the front end drains it (low-frequency `call_utility` poll, `core_client.py:812`)
  and submits phantoms through the existing submitter. **Recommended for v1**:
  no new transport, no new message types, decision logic stays where the value
  data is.
- **(b) New engine→front-end message type** for "submit this request." Invasive,
  touches the core protocol.
- **(c) Manager drives the KV load directly through the connector**, no phantom
  request at all. Cleanest long-term — it deletes the phantom machinery — but
  needs a connector API to load a prefix by hash outside a request lifecycle,
  which `LMCacheMPConnector` does not expose today.

---

## 5. The pin problem and speculative ownership — one mechanism `[decided]`

The diagram states:

> "now we don't need to worry about prefetched kv blocks accidentally getting
> evicted"

Not yet. `_finalize_prefetch_only_request` (`scheduler.py:2325-2373`) calls
`kv_cache_manager.cache_blocks(...)` and then `self._free_request(request)`
`[verified]`. So a prefetched prefix ends up **registered in the APC hash table
but with `ref_cnt == 0`, sitting in the free queue** — fully evictable.

Today LRU protects it *by accident*: freshly freed blocks go to the tail
(`block_pool.py:420`), and the tail is evicted last. **A score-based re-splice
removes that accident** — a low-scoring prefetched block is moved to the head,
so this policy can evict a prefix it just paid to fetch.

### These are the same problem

01 §7 files "speculative vs confirmed ownership for phantom-created entries" as
a separate open item. It isn't one. Both are the same defect: **an index entry
created by a prediction rather than an observation.** The prefetched prefix
needs protection *because* it is speculative, and it must stop being protected
at exactly the moment the speculation is falsified. One mechanism, three parts.

### Part 1 — Provenance, stamped at insert `[verified]`

`cache_full_blocks` (`block_pool.py:211`) runs for phantom and real requests
alike, and it receives the `Request`. The discriminator already exists:
`Scheduler._is_prefetch_only_request` (`scheduler.py:2321`) reads
`request.kv_transfer_params["prefetch_only"]`. Index entries are stamped
`speculative=True, created_at=now` when created by a phantom. No new plumbing.

### Part 2 — Confirmation is free `[verified]`

`BlockPool.touch()` (`block_pool.py:391`) fires when a request's prefix matches
a cached block and rescues it from the free queue. That *is* the signal "the
prediction came true" — flip `speculative → confirmed` there. No new hook, and
no dependence on the forecast being re-published.

**Where it must be driven from:** `KVCacheManager.allocate_slots`
(`kv_cache_manager.py:225`) `[verified]`. It holds both the whole `Request` and
`new_computed_blocks`. Neither layer below does — `BlockPool.touch()` sees only
blocks, and `allocate_new_computed_blocks`
(`single_type_kv_cache_manager.py:169`, touch at `:218`) sees only `request_id`.

**Gate the confirm on `not prefetch_only`.** A phantom that hits an existing
prefix also calls `touch()`, so an ungated confirm lets one prediction confirm
another — the forecast marking its own homework.

### Part 3 — One decaying floor is the soft pin

```
score = max(base_importance, floor(age))    # speculative entries only
floor: starts above the top of the normal score range,
       decays to BELOW the neutral default over TTL
TTL  = time_to_next_call from PROB          # the prediction's own deadline
```

Why this exact shape:

- **`max`, not `+`.** The floor only ever protects; it never inflates a score.
  Once an entry is confirmed the floor is gone and the block is scored honestly,
  so a prefetch cannot contaminate the ranking it was supposed to serve.
- **Decays past *neutral*, not to zero-ish.** A falsified prediction doesn't
  merely lose protection — it becomes the *preferred* victim. Self-correcting,
  and it is the cheap answer to 01 §7's "speculative entries demoted if no real
  request arrives": demotion is the same decay curve, not a second mechanism.
- **TTL from `time_to_next_call`** — the forecast already states when the block
  is due. Right prediction → touched before decay → confirmed. Wrong → evicted
  first. Neither path needs a timer or a release callback.
- **Multi-owner `max` (01 §4) already covers the shared preamble.** A confirmed
  owner protects the block regardless of a co-owning speculative entry, so a bad
  prediction over a common prefix cannot drag it down.
- **Hard-drop speculative entries at TTL×k**, which subsumes the unbounded-index
  -growth item (01 §5, *Age*) for the speculative half.

### Why not a hard pin `[decided]`

Holding `ref_cnt > 0` from a synthetic owner is genuinely unevictable, but needs
a TTL and a leak-proof release path; one lost release strands the block for the
process lifetime. **A hard pin turns a scoring bug into a leaked-block bug.**
The floor can still be overridden under genuine pressure, which is arguably the
correct behaviour anyway.

### Observability — not optional here `[decided]`

```
speculative_evicted_before_confirm / speculative_created
```

That ratio *is* prefetch waste, and it is the only honest read on whether the
forecast is worth anything. Without it a policy that protects garbage for a full
TTL looks identical to one that works. See 01 §8 (metrics).

### Ordering constraint for the build `[decided]`

**The floor ships in step 3, not step 6.** The re-splice is precisely what
removes LRU's accidental tail protection, so a step-3 build without the floor
can evict a prefix it just paid ≈11.4 s to fetch — the worst regression this
policy is capable of producing. See §10.

---

## 6. Two key schemas — keep them distinct `[decided]`

| Table | Key | Value | Lifetime |
|---|---|---|---|
| **Block ownership index** | `(job_id, node, call_type)` | ordered list of `block_id`; reverse map `block_id → set[key]` | lives and dies with the blocks (01 §4) |
| **Timing memo** | `node_name + (LLM\|TOOL:LLM) name + hash(input)` | mean LLM call time, mean tool call time | persists across jobs (§8) |

They have different granularity and different lifetimes; merging them will
produce a table that is wrong for both purposes.

Note: the `hash(input)` component may be able to reuse vLLM's existing block-hash
machinery (`kv_cache_utils.request_block_hasher:643`) rather than introducing a
second hash of the same tokens. `[open]` — worth checking whether the memo needs
the whole input or just the node's stable prefix.

---

## 7. Ingress discipline `[decided]`

The diagram note says the manager "has to call the trace analyser to get the
information." **Don't.** Trace Analyser *writes* Redis; the manager *reads* Redis
and nothing else.

One ingress means one failure mode, one timeout policy, one staleness rule. Two
ingresses put a second external dependency in the process that owns the
scheduler loop.

---

## 8. Already built — reuse, don't rebuild `[verified]`

> "The system maintains Job-based MRU queue of the recent prefixes used to
> enable prefetching"

This exists: `vllm/v1/agent_prefetch/registry.py` —
`OrderedDict[agent_id, OrderedDict[prefix_hash, PrefixDescriptor]]` (`:100-102`),
MRU promotion via `move_to_end` (`:126`, `:132`), `record()` (`:106`),
`top_k()` (`:158`), `agent_size()` (`:198`), documented O(1) amortized (`:69`).
Bounds are hardcoded: `default_top_k=20`, `max_agents=10_000`,
`max_per_agent=None`.

Two deltas if it is reused as-is: it is keyed by `agent_id`
(`{namespace}:{node_name}`), not by `(job_id, node, call_type)`; and it lives in
the **front-end** process alongside the submitter, not in engine core.

### Persistent value list per node `[open]`

The diagram answers "should there be a persistent value list for each langgraph
node? **YES**." Which persistence is unspecified:

- **Across job end, in-process** — a dict that outlives the blocks. Easy.
- **Across process restart** — needs writing back to Redis, which makes the
  manager a Redis **writer**. New failure mode, new key ownership question (who
  wins if the trace analyser writes the same key).

---

## 9. Demote-to-LMCache instead of destroy `[open]` — scope call needed

> "React agents vs non react agents — we don't need to keep the non react agents
> in the HBM, we can just push it to lm cache"

vLLM eviction **destroys** a block; there is no demote path. Getting this means
a connector-side save triggered by eviction. The adjacent machinery exists —
`LMCacheMPConnector.request_finished` stores and returns `delay_blocks=True`
(`lmcache_mp_connector.py:1038`) — but it is driven by *request completion*, not
by eviction.

**Given 01 §5's v1 scorer assumes L1 residency**, the cheap version of this is:
score non-react nodes low and let them be evicted early, relying on L1 to still
have the prefix. That gets most of the benefit with zero new machinery.
Explicit demote-on-evict is only needed where L1 residency is *not* guaranteed.

**This decision is now data-gated, not a judgement call.** `p_cold` per
`(node, call_type)` (04 §5.4) says directly how often an evicted prefix is
*not* recoverable from L1. If `p_cold ≈ 0`, score-low-and-drop is correct and
demote-on-evict is dead weight. If it is material — especially concentrated on
the non-react nodes this section is about — that is the evidence for building
it. Answer 01 §5 step 1 before spending anything here.

---

## 10. Build order (canonical — supersedes 01 §9)

Each step is testable before the next exists.

| Step | What | Blocked by |
|---|---|---|
| **1** | **Index.** `(job_id, node, call_type) ↔ block_id`, multi-owner reverse map, ordered positions, **plus the `speculative` / `created_at` stamp** (§5 part 1). Maintained from `cache_full_blocks` (`block_pool.py:211`), `_maybe_evict_cached_block` (`:354`), cleared by `reset_prefix_cache` (`:443`). No Redis, no scoring. Unit-testable against a fake pool. | nothing |
| **2** | **Poller + snapshot.** Daemon thread, immutable snapshot, atomic swap, fed from a fixture dict instead of Redis. Verify the loop never blocks. | nothing |
| **3** | **REORDER enforcement.** Add `appendleft` to `FreeKVCacheBlockQueue` (additive), add the O(1) `num_free_fresh` counter, then a bounded-K re-splice on a wall-clock tick driven by a hand-written score map, with candidates taken from the index rather than `get_all_free_blocks()`. Splice is skipped while fresh blocks are plentiful; unscored blocks keep their LRU position. **Includes the speculative floor + confirm-on-touch (§5)** — this step is what removes LRU's accidental protection of prefetched prefixes, so the floor cannot lag it. First point where hit rate moves — and therefore the first point it can move the *wrong* way, so the **observability in 07 ships here too**: always-on counters, the flag-gated eviction decision log, and the score/outcome correlation. A/B flag exercised here, and flag-off must be byte-identical to upstream. | nothing |
| **3.5** | **Value table + per-request consult.** The O(1) lookup the engine reads (§2). Separates "compute" from "read" before any real scoring exists. | nothing |
| **4** | **Real Redis subscriber** behind the same snapshot interface — `psubscribe` + `MGET` seed on connect/reconnect. Only the thread's internals are new. | all of 03 §7 — publish confirmed, channel naming, message payload, shape, units, endpoint |
| **5** | **Real scoring function** — [`08-scoring.md`](08-scoring.md), plus the timing memo (§6). Shape is spec'd; this step is tuning and ablation against the replay harness (08 §7), which should exist by now from step 3. | `τ`, cost constants, worst-K vs threshold |
| **6** | **Prefetch/admission instructions** — want-list + front-end drain (§4 option a). The protection floor is no longer here; it moved to step 3. | §4 decision |
| **7** | **Demote-to-L1**, only if §9 is decided in favour. | §9 decision |

Steps **1, 2, 3 and 3.5 are unblocked now.**

---

## 11. Open items this doc adds

- Prefetch origination path — §4 (a), (b) or (c)
- Persistence scope for the value list — in-process vs Redis write-back (§8)
- Demote-to-L1 in v1 or not (§9)
- Whether the timing memo can reuse the existing block hasher (§6)

Carried from 01 §8 and still open: `call_type` determinism across turns, payload
shape, units, Redis endpoint/auth, write frequency, single-KV-group
confirmation, the formula and its constants.

**Closed by this doc:** the protection mechanism and speculative-vs-confirmed
ownership (01 §7) — one decaying floor with confirm-on-touch, §5. Remaining
sub-item is constants only: floor height, decay curve, and the TTL×k hard drop.
