# Lifecycle of a KV Cache Block in vLLM (v1)

This document walks through every state a single GPU KV-cache block
(`KVCacheBlock`) passes through during its lifetime, from server startup to
shutdown. It is a companion to [`l0_eviction.md`](./l0_eviction.md), which
covers the data structures in detail.

All references are against the vLLM checkout at
`/Users/vibhavsharma/Projects/vllm`.

---

## 1. The five logical states

A `KVCacheBlock` is always in exactly one of these states:

| # | State | `ref_cnt` | `_block_hash` | In free list? | Meaning |
|---|---|---|---|---|---|
| **S0** | **Pristine** | 0 | `None` | yes (initial order: by `block_id`) | Block was created at startup, never used. |
| **S1** | **Allocated–partial** | ≥ 1 | `None` | no | Owned by one or more requests; not yet full. |
| **S2** | **Allocated–full–cached** | ≥ 1 | set | no | Owned and full; indexed in the prefix cache for future hits. |
| **S3** | **Free–cached** | 0 | set | yes (at tail when freshly freed) | Released by its last owner, but its KV content is still indexed in the prefix cache — eligible *both* for eviction and for prefix-hit rescue. |
| **S4** | **Free–uncached** | 0 | `None` | yes | Free, no cache identity. The block was either pristine-and-popped (S0 transient), or just had its hash reset by eviction. About to be allocated to a new request. |

The state graph (arrows are transitions; transition labels are the code paths
in §3):

```
                 [T1: get_new_blocks→partial]
   S0 ──────────────────────────────────────► S1
                                                │
                                       [T2: cache_full_blocks]
                                                ▼
                                                S2
                                                │
                                        [T3: free_blocks]
                                                ▼
                          ┌──────────────────── S3 ────────────────────┐
                          │                                            │
            [T4: touch()  │  prefix hit]                  [T5: popleft+│_maybe_evict_cached_block]
                          ▼                                            ▼
                          S2  ←──────────────────────[T2 again]─────── S4
                                                                       │
                                                              [T6: ref_cnt++]
                                                                       ▼
                                                                       S1
```

The two arrows out of S3 are the heart of the design: a free-but-still-cached
block is **simultaneously** an eviction candidate (T5) and a prefix-hit
candidate (T4). Whichever event happens first wins. This dual eligibility is
why `FreeKVCacheBlockQueue` needs O(1) middle removal (see
[`l0_eviction.md` §2.3](./l0_eviction.md#23-the-rescue-path-blockpooltouch-and-why-we-need-o1-middle-removal)).

---

## 2. Birth and death

### 2.1 Birth (server startup)

`vllm/v1/core/block_pool.py:166-176`

```python
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)
]
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
self.null_block = self.free_block_queue.popleft()
```

- All `num_gpu_blocks` are constructed in state **S0 (Pristine)**:
  `ref_cnt=0`, `_block_hash=None`, in the free list ordered by `block_id`.
- One block is immediately popped to serve as the **null block** — a permanent
  sentinel used for null/padding slots (e.g. sliding-window or Mamba prefix
  caching). The null block is `is_null=True` and never re-enters the free
  list.

After startup, the pool contains `num_gpu_blocks - 1` blocks in state S0,
plus the null block.

### 2.2 Death (server shutdown)

There is no explicit death. The blocks live as long as the `BlockPool`
object. On `reset_prefix_cache()`
(`vllm/v1/core/block_pool.py:445`-onwards) every block is forced back to
S4 (Free–uncached) by clearing every `_block_hash` and rebuilding
`cached_block_hash_to_block`, but the `KVCacheBlock` objects themselves
persist.

---

## 3. The six transitions

### T1: S0/S4 → S1 — "allocate fresh slots"

**Code path:** `BlockPool.get_new_blocks` (`block_pool.py:333-352`)

Triggered when the scheduler needs new physical slots for a request — either
to hold prefix tokens that *were not* served by the prefix cache, or to
extend an existing request as it decodes.

```python
ret = self.free_block_queue.popleft_n(num_blocks)   # pulls LRU victims
for block in ret:
    self._maybe_evict_cached_block(block)           # S3 → S4 (only if cached)
    block.ref_cnt += 1                              # S0/S4 → S1
```

Two sub-cases for the popped block:

- It was in **S0** (Pristine): `_maybe_evict_cached_block` is a no-op
  because `block_hash is None`. We pay nothing for the "eviction."
- It was in **S3** (Free–cached): `_maybe_evict_cached_block` removes the
  entry from `cached_block_hash_to_block` and calls `reset_hash()`, taking
  the block to S4 *internally* before `ref_cnt += 1` lands it in S1.
  A `BlockRemoved(medium=MEDIUM_GPU)` event is emitted on the KV event queue.

End state: **S1 (Allocated–partial)** — the block is in the request's
`req_to_blocks[request_id]` list and the request will start writing KV into
it.

### T2: S1 → S2 — "block becomes full, cache it"

**Code path:** `BlockPool.cache_full_blocks` (`block_pool.py:211-275`)

Called by `KVCacheManager.cache_blocks` after a request writes its
`block_size`-th token into a block. The block now contains a complete,
hashable KV window.

```python
for i, blk in enumerate(new_full_blocks):
    if blk.is_null:
        continue
    assert blk.block_hash is None              # invariant: was in S1
    block_hash = new_block_hashes[i]
    block_hash_with_group_id = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
    blk.block_hash = block_hash_with_group_id            # S1 → S2
    self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
```

End state: **S2 (Allocated–full–cached)**. The block is still owned
(`ref_cnt ≥ 1`) but is now also discoverable via the prefix cache — meaning a
*concurrent* request with the same prefix can `touch()` it and share it
(bumping `ref_cnt` to 2+). Sharing is what makes long system prompts free
across requests.

### T3: S1/S2 → S3/S4 — "request finishes, release"

**Code path:** `SingleTypeKVCacheManager.free` →
`BlockPool.free_blocks` (`single_type_kv_cache_manager.py:303`,
`block_pool.py:408-422`)

```python
# single_type_kv_cache_manager.py:303
req_blocks = self.req_to_blocks.pop(request_id, [])
ordered_blocks = reversed(req_blocks)              # tail-of-chain first
self.block_pool.free_blocks(ordered_blocks)

# block_pool.py:408
for block in blocks_list:
    block.ref_cnt -= 1
self.free_block_queue.append_n(
    [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
)
```

End state depends on whether `ref_cnt` reached zero **and** on whether the
block was cached:

- `ref_cnt` reaches 0 and the block had a hash → **S3 (Free–cached)**:
  appended to the tail of the free list, still indexed in
  `cached_block_hash_to_block`.
- `ref_cnt` reaches 0 and the block had no hash (it was partial) → **S4
  (Free–uncached)**: appended to the tail of the free list, no cache entry
  to preserve.
- `ref_cnt` is still > 0 (another concurrent request still owns it via
  shared prefix) → stays in **S2**.

The `reversed()` in the caller is what enforces the tail-of-chain eviction
tiebreaker. By reversing, tail blocks get appended first (so they end up
closer to the LRU head) — meaning when memory pressure hits, tail-of-chain
(less-shared) blocks die before root-of-chain (more-shared) blocks.

### T4: S3 → S2 — "prefix-cache rescue"

**Code path:** `BlockPool.touch` (`block_pool.py:391-404`)

This is the transition that justifies the custom data structure. Triggered
when a new request's prefix-hash lookup hits a block currently sitting in the
free list.

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)        # O(1) middle removal
        block.ref_cnt += 1
```

The block jumps directly from the free list back into ownership without ever
being repurposed or having its KV data touched. Its `_block_hash` is
unchanged. Its KV bytes are unchanged. The new request just inherits the
existing computed KV — that's prefix caching paying off.

End state: **S2 (Allocated–full–cached)** with `ref_cnt` bumped.

### T5: S3 → S4 — "lose the race, get evicted"

**Code path:** the eviction half of `get_new_blocks` (T1 above).

When a block is in S3 and another allocation reaches it via `popleft_n`
before any prefix-hit rescue arrives, the block is evicted from the prefix
cache: its `cached_block_hash_to_block` entry is removed, `_block_hash` is
reset to `None`, a `BlockRemoved` event fires, and the block is now in
**S4 (Free–uncached)** — momentarily — before `ref_cnt += 1` takes it to S1.

Note: S4 is essentially a *transient* state during the body of
`get_new_blocks`. A block almost never sits in S4 across function calls
unless something has manually invalidated its hash (see §4 below).

### T6: S4 → S1 — finalize allocation

The last two lines of `get_new_blocks`:

```python
block.ref_cnt += 1
metrics_collector.on_block_allocated(block)
```

The block is now owned by the new request and will be written into.

---

## 4. The escape hatch: explicit invalidation

`BlockPool.evict_blocks(block_ids)` (`block_pool.py:424-441`) and
`reset_prefix_cache()` (`block_pool.py:445`) allow the system to forcibly
push blocks from S2/S3 to S1/S4 respectively by tearing down their cache
entries. These exist for:

- Test isolation between runs.
- Administrative cache flushes (e.g. when the LMCache connector requests a
  prefix-cache reset).
- Sliding-window / Mamba sparse-attention configurations that periodically
  reclaim null blocks.

These calls do **not** touch `ref_cnt` — blocks currently in use stay
allocated; only the prefix-cache identity is dropped.

---

## 5. A worked example: two requests sharing a system prompt

Walk through what happens to one specific block as two requests share a
prefix. Assume `block_size=16` tokens and a system prompt of 32 tokens
(2 blocks: B0 holds tokens 0-15, B1 holds tokens 16-31).

```
t=0   Server starts. B0, B1 are in S0 (pristine).

t=1   R1 arrives, prefix-miss (cold cache).
      Scheduler calls allocate_slots(R1, …) → get_new_blocks(2) →
        popleft_n pulls B0, B1 (they were S0; eviction is a no-op).
      B0, B1 transition S0 → S1. ref_cnt=1 each.

t=2   R1's forward pass writes KV for tokens 0..31 into B0, B1.
      After tokens 15 and 31 land, cache_blocks(R1, 32) calls cache_full_blocks.
      B0, B1 transition S1 → S2.
      B0.block_hash = H0, B1.block_hash = H1.
      cached_block_hash_to_block[H0] = B0, [H1] = B1.

t=3   R1 finishes generation.
      free(R1) reverses block list → free_blocks([B1, B0]).
      B1.ref_cnt → 0, B0.ref_cnt → 0.
      Both appended to free list tail (B1 first, then B0). Now in S3.

t=4   Other requests come and go. The free list churns.
      B0, B1 drift toward the LRU head but are not yet popped.

t=5   R2 arrives with the SAME system prompt.
      scheduler computes block hashes for R2's prefix.
      get_computed_blocks(R2) → coordinator.find_longest_cache_hit →
        BlockPool.get_cached_block(H0) returns B0, get_cached_block(H1) returns B1.
      → prefix cache HIT for the first 32 tokens.

t=6   allocate_slots calls touch([B0, B1]).
      For each: ref_cnt==0 → free_block_queue.remove(block) [O(1) splice],
                ref_cnt += 1.
      B0, B1 transition S3 → S2.
      R2 spent ZERO compute on those 32 tokens — it inherits R1's KV verbatim.

t=7   R2 generates new tokens past position 32, gets a new block B2.
      B2 transitions S0/S4 → S1 → eventually S2.

t=8   R2 finishes. free(R2) re-runs the cycle. B0, B1 go S2 → S3 again.

t=9   Days later, a memory-pressure burst hits. B0 drifts to LRU head.
      A new request needs blocks. get_new_blocks pops B0.
      _maybe_evict_cached_block: cached_block_hash_to_block.pop(H0) → done.
      B0.reset_hash(). BlockRemoved event fires.
      B0 transitions S3 → S4 → S1 (in one function body).
      A different prefix's KV is now written into B0.
```

The key observations:

- At `t=4` B0/B1 are in a peculiar state: **the request that owns them is
  long gone, but their content is alive and recallable.** That's S3.
- The `touch()` at `t=6` is the entire reason for the doubly-linked list
  design — without O(1) middle removal, that "free hit" would scale linearly
  with how many free blocks happen to be ahead of B0/B1 in the list.
- The `BlockRemoved` event at `t=9` is what LMCache's `l0_lifecycle`
  observability subscriber consumes — it never sees the block at all until
  the moment it's evicted from L0.

---

## 6. Summary

| Phase | Owns | Indexed | Free list | Triggered by |
|---|---|---|---|---|
| Pristine (S0) | nobody | no | yes | startup |
| Allocated–partial (S1) | ≥1 request | no | no | `get_new_blocks` |
| Allocated–full–cached (S2) | ≥1 request | yes | no | `cache_full_blocks` |
| Free–cached (S3) | nobody | **yes** | **yes** | `free_blocks` after all refs released |
| Free–uncached (S4) | nobody | no | yes | `_maybe_evict_cached_block` or partial-then-freed |

The unusual state is **S3**: simultaneously freed and indexed. Every other
KV-cache design choice in vLLM v1 — the intrusive linked list, the
`_block_hash`-on-the-block field, the reverse-on-free ordering, the
`touch()`/`get_cached_block` pair, the `BlockRemoved` event timing — exists
to make S3 work efficiently for both possible exits (rescue at T4, eviction
at T5).
