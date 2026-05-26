# vLLM L0 (GPU KV Cache) — Data Structures & Eviction Logic

This document describes how vLLM v1 manages its GPU KV cache blocks ("L0" in the
LMCache tier terminology) and how it chooses a victim when the pool is full and
a new request needs a slot.

All file/line references are against this checkout of vLLM at
`/Users/vibhavsharma/Projects/vllm`.

---

## 1. The core data structures

vLLM's L0 manager keeps the GPU's paged-attention block pool as an array of
fixed-size blocks plus auxiliary bookkeeping. The relevant types live in
`vllm/v1/core/kv_cache_utils.py` and `vllm/v1/core/block_pool.py`.

### 1.1 `KVCacheBlock` — per-block metadata

`vllm/v1/core/kv_cache_utils.py:113`

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                                      # 0 .. num_gpu_blocks-1
    ref_cnt: int = 0                                   # active references
    _block_hash: BlockHashWithGroupId | None = None    # prefix-cache key
    prev_free_block: "KVCacheBlock | None" = None      # doubly-linked list
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False
```

Key invariants:

| Field | Meaning |
|---|---|
| `block_id` | Index into the physical GPU KV tensor. Stable for the block's lifetime. |
| `ref_cnt` | Number of in-flight requests currently using this block. `ref_cnt > 0` ⇒ the block is **never** an eviction candidate. |
| `_block_hash` | Set only when the block is *full* (every slot written) and *cached* in the prefix cache. `None` means the block is either partial or not cached (e.g. just evicted via `reset_hash()`). |
| `prev_free_block` / `next_free_block` | The block's links inside `FreeKVCacheBlockQueue` when `ref_cnt == 0`. `None` when the block is in use. |
| `is_null` | A sentinel for the special "null block" that is popped once at init and never recycled. |

`reset_hash()` (`kv_cache_utils.py:144`) clears `_block_hash` and is called by
`BlockPool._maybe_evict_cached_block` when the block is taken away from the
prefix cache.

### 1.2 `FreeKVCacheBlockQueue` — the eviction-ordered free list

`vllm/v1/core/kv_cache_utils.py:162`

This is the heart of the eviction policy. It is **not** a `collections.deque`
— it is a **hand-rolled, intrusive, doubly-linked list** with fake head/tail
sentinels, threaded through the `prev_free_block` / `next_free_block` fields
of `KVCacheBlock` itself.

```
fake_head ⇄ block_A ⇄ block_B ⇄ … ⇄ block_Z ⇄ fake_tail
            ↑                              ↑
            LRU (evict next)               MRU (most-recently-freed)
```

Why a custom structure (`kv_cache_utils.py:163-167`):

> *We implement this class instead of using Python builtin deque to support
> removing a block in the middle of the queue in O(1) time. To close the
> performance gap to the builtin deque which is implemented in C++, this class
> does not allocate any Python objects when manipulating the linked list.*

The structure must support **O(1) middle removal** because the prefix cache
can "rescue" a free-but-still-cached block when a new request hits its hash —
that block needs to be pulled out of the free list without scanning.

Ordering contract (`kv_cache_utils.py:171-176`):

> *The queue is ordered by block ID in the beginning. When a block is allocated
> and then freed, it will be appended back with the eviction order:*
>
> *1. The least recent used block is at the front (LRU).*
> *2. If two blocks have the same last accessed time (allocated by the same
>    sequence), the one with more hash tokens (the tail of a block chain) is
>    at the front.*

The tail-of-chain tiebreaker is enforced by `SingleTypeKVCacheManager.free`
(`single_type_kv_cache_manager.py:303-317`), which reverses a request's block
list before appending:

```python
# Free blocks in reverse order so that the tail blocks are freed first.
ordered_blocks = reversed(req_blocks)
self.block_pool.free_blocks(ordered_blocks)
```

Reversing means **tail blocks are appended first** → they sit closer to the
LRU head → they are evicted *before* root/shared blocks. This is correct
because root-of-chain blocks are shared by more prefixes and therefore have
higher reuse value.

#### Public API and complexity

All operations are **O(1)** because of the doubly-linked, sentinel-bounded
design:

| Method | Purpose | Complexity |
|---|---|---|
| `popleft()` | Pop the LRU block (head). Allocation path. | O(1) |
| `popleft_n(n)` | Pop the `n` LRU blocks in one walk. Bulk allocation. | O(n) |
| `append(block)` | Append a newly-freed block at the MRU tail. | O(1) |
| `append_n(blocks)` | Append several freed blocks at the tail in order. | O(\|blocks\|) |
| `remove(block)` | Excise a block from anywhere in the list (used by `touch()` when a free block is rescued by a prefix-cache hit). | **O(1)** — the key win over `deque`. |
| `get_all_free_blocks()` | Iterate the list. Test-only. | O(n) |

### 1.3 `BlockPool` — the owning manager

`vllm/v1/core/block_pool.py`

The pool owns the full block array, the free queue, and the prefix-cache hash
table:

```python
self.blocks: list[KVCacheBlock]                              # all blocks
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)   # LRU list
self.cached_block_hash_to_block: dict[BlockHashWithGroupId, dict[int, KVCacheBlock]]
self.null_block = self.free_block_queue.popleft()            # init-time sentinel
```

`cached_block_hash_to_block` is the **prefix cache index** that lets vLLM
answer "do I already have a GPU block holding this exact prefix content?"
without scanning. Eviction must keep this index in sync with the free queue.

---

## 2. The eviction logic, step by step

Scenario from the question: **L0 is full, a new request comes in with a prefix
miss, vLLM needs blocks for it (and will then call LMCache to populate them).**

### 2.1 Allocation path: `BlockPool.get_new_blocks`

`vllm/v1/core/block_pool.py:333-352`

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(...)

    ret = self.free_block_queue.popleft_n(num_blocks)   # (1) pick N LRU victims

    if self.enable_caching:
        for block in ret:
            self._maybe_evict_cached_block(block)       # (2) detach from prefix cache
            assert block.ref_cnt == 0
            block.ref_cnt += 1                          # (3) take ownership
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
    else:
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            ...
    return ret
```

Three things happen, in order:

1. **Pick victims.** `free_block_queue.popleft_n(N)` returns the `N`
   least-recently-used free blocks from the head of the list. These are the
   eviction victims. Note: blocks with `ref_cnt > 0` are **not in the free
   list at all** — only unreferenced cached blocks are eligible.
2. **Detach from the prefix cache.** For each victim, `_maybe_evict_cached_block`
   removes the entry from `cached_block_hash_to_block` and clears
   `block._block_hash` via `reset_hash()`. The block's cached identity is now
   gone — future requests that hash to the same prefix will **miss** L0 (and
   that is exactly the cue for LMCache to repopulate from L1/L2).
3. **Re-own.** `ref_cnt += 1`. The block now belongs to the new request.

### 2.2 The detach step: `_maybe_evict_cached_block`

`vllm/v1/core/block_pool.py:354-389`

```python
def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
    if self.metrics_collector:
        self.metrics_collector.on_block_evicted(block)

    block_hash = block.block_hash
    if block_hash is None:
        return False                                    # never cached → nothing to do

    if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
        return False                                    # already gone

    block.reset_hash()

    if self.enable_kv_cache_events:
        self.kv_event_queue.append(
            BlockRemoved(
                block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                medium=MEDIUM_GPU,
                group_idx=get_group_id(block_hash),
            )
        )
    return True
```

The emitted `BlockRemoved(medium=MEDIUM_GPU)` event is the same signal
LMCache's `l0_lifecycle` subscriber consumes to compute
`lmcache_mp_l0_block_lifetime_seconds`.

### 2.3 The "rescue" path: `BlockPool.touch` and why we need O(1) middle removal

`vllm/v1/core/block_pool.py:391-404`

This is the subtle part. To understand it, you have to internalize one fact
about vLLM's L0:

> **A block can be in the free list and still be in the prefix cache at the
> same time.**

That is, `ref_cnt == 0` (no active request is using it) does **not** imply
"this block is empty / available to overwrite." It means: *no request is
currently holding a reference, so it is **eligible** to be reused if we run
out of fresh blocks — but until that actually happens, its content is still
valid and is still indexed by its `block_hash` in
`cached_block_hash_to_block`.*

This is the whole point of the GPU prefix cache: KV data lingers on the GPU
after the request that produced it finishes, so that a *future* request with
the same prefix can reuse it instead of recomputing.

#### Walk through the lifecycle of one block on a cache hit

Imagine this timeline:

```
t=0   Request R1 ("Hello, my name is …") writes KV into block B.
      B.ref_cnt = 1
      B.block_hash = H("Hello, my name is …")
      cached_block_hash_to_block[H] = B
      B is NOT in the free list.

t=1   R1 finishes.
      free_blocks() decrements B.ref_cnt to 0.
      free_block_queue.append(B)  → B is now at the tail (MRU end) of the free list.
      B.block_hash is STILL set. cached_block_hash_to_block still points to B.

t=2   (some time passes; other requests come and go, B drifts toward the head)

t=3   Request R2 arrives with prefix "Hello, my name is …"
      The scheduler hashes R2's prefix blocks and calls BlockPool.get_cached_block(H).
      It returns B — a prefix-cache HIT.

t=4   The scheduler now calls BlockPool.touch([B]) to take ownership of B for R2.
```

At `t=4`, **B is sitting in the middle of the free list.** It is not at the
head, it is not at the tail — its position depends entirely on what has been
freed since `t=1`. We need to:

1. Remove B from the free list (so a later `popleft()` does not hand it to
   someone else as an eviction victim), **and**
2. Bump `B.ref_cnt` to 1 so R2 now owns it.

That's exactly what `touch` does:

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        # ref_cnt=0 means this block is in the free list (i.e. eviction
        # candidate), so remove it.
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

#### Why O(1) is non-negotiable here

Look at the access pattern. A single request can hit *many* cached blocks on
its prefix — easily dozens for a long shared system prompt, sometimes
hundreds. Every one of those hits triggers a `touch()` call, and every cached
block that happens to be in the free list triggers a middle-of-list
`remove()`.

If `remove()` were `O(n)` (where `n` = number of free blocks, often tens of
thousands on a large GPU), then a single prefix hit could cost
`O(hits × num_free_blocks)` — turning a fast-path optimization (prefix cache
hit, no compute needed) into a scheduler hot spot.

`collections.deque` cannot do middle removal in O(1). Even though deque is
implemented in C and has fast head/tail ops, removing an arbitrary element
requires scanning. That's the reason for the hand-rolled intrusive linked
list.

The trick that makes it O(1) is that the free-list pointers
(`prev_free_block`, `next_free_block`) live **directly on the
`KVCacheBlock`** rather than in a separate node object. So given a block
reference (which we already have from the prefix-cache lookup), we can splice
it out of the list in three pointer assignments without searching:

```python
# from FreeKVCacheBlockQueue.remove() (kv_cache_utils.py:284-302)
block.prev_free_block.next_free_block = block.next_free_block
block.next_free_block.prev_free_block = block.prev_free_block
block.prev_free_block = block.next_free_block = None
self.num_free_blocks -= 1
```

No iteration, no hashing, no search — just three writes. That is the
algorithmic property the data structure exists to provide.

#### Why the block isn't simply "removed from the free list when cached"

A natural alternative would be: when a block becomes cached (gets a
`block_hash`), take it out of the free pool entirely, and only put it back
when the prefix cache decides to evict it. But that breaks the eviction
policy: we *want* cached-but-unreferenced blocks to be eviction candidates,
because if memory pressure ever exceeds the prefix-cache working set, we
need to reclaim them. So they have to stay in the free list, ordered by LRU,
right alongside genuinely empty blocks — and only get pulled out when either
(a) the LRU pointer reaches them and they get repurposed, or (b) a prefix
hit "rescues" them via `touch()`.

That dual eligibility is why the free list needs cheap middle removal in the
first place.

### 2.4 The freeing path: `BlockPool.free_blocks`

`vllm/v1/core/block_pool.py:408-422`

When a request finishes, its blocks are returned to the tail (MRU end) of the
queue, ordered tail-of-chain first:

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1
    self.free_block_queue.append_n(
        [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
    )
```

Combined with the `reversed(req_blocks)` in
`SingleTypeKVCacheManager.free`, this realizes the tail-of-chain tiebreaker
described in §1.2.

---

## 3. Eviction policy in one sentence

vLLM's L0 eviction policy is **LRU over the unreferenced-block set, with
tail-of-chain blocks preferred over root-of-chain blocks on ties**, implemented
as an O(1) intrusive doubly-linked list (`FreeKVCacheBlockQueue`) threaded
through the per-block metadata. A block is eligible only when `ref_cnt == 0`;
when evicted, its prefix-cache entry is removed and its hash is reset, and a
`BlockRemoved(medium=GPU)` event is emitted.

---

## 4. Interaction with LMCache

When vLLM evicts a block from L0, the KV bytes themselves are simply
overwritten in place by the next request's writes. The data is **not lost**
from the system, however, because LMCache already copied that block's
contents to L1 (CPU DRAM) at `STORE` time, and possibly to L2 (storage) via
the async `StoreController`.

So the end-to-end flow on the miss scenario is:

```
1. New request arrives, needs N GPU blocks.
2. BlockPool.get_new_blocks(N) pops N LRU victims, drops their prefix-cache entries.
3. vLLM allocator hands the new request the empty slots.
4. vLLM connector calls LMCache RETRIEVE for the request's prefix.
5. LMCache hits L1 → DMA L1→L0 into the slots. (Or L1 miss → fetch L2→L1, then L1→L0.)
6. New request executes attention over the populated GPU blocks.
```

vLLM is the sole arbiter of which L0 block dies; LMCache simply fills whatever
empty slot vLLM hands it.

---

## 5. File index

| File | What lives there |
|---|---|
| `vllm/v1/core/kv_cache_utils.py:113` | `KVCacheBlock` dataclass |
| `vllm/v1/core/kv_cache_utils.py:162` | `FreeKVCacheBlockQueue` — the LRU list and its O(1) operations |
| `vllm/v1/core/block_pool.py` | `BlockPool` — owns the array, the free queue, the prefix-cache index, and the allocate/evict/free/touch entry points |
| `vllm/v1/core/single_type_kv_cache_manager.py:303` | `free()` — reverses the per-request block list to enforce the tail-of-chain tiebreaker |
| `vllm/v1/core/kv_cache_manager.py` | Higher-level manager that calls into `BlockPool` |
