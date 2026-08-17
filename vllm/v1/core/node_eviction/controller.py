# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The node-aware eviction controller — build steps 3 and 3.5 (02 §10).

Three costs, three budgets (02 §2):

| | Runs where | Budget | Does |
|---|---|---|---|
| subscriber thread | daemon thread | unbounded | Redis -> immutable snapshot |
| tick | scheduler loop, wall-clock paced | O(tracked log K) + O(K) | score, splice |
| per-request consult | scheduler loop, every request | **O(1)** | one dict lookup |

The per-request path must never compute a score, touch Redis, or walk the
free queue. That is also what makes "if there is a miss, go to default" true
by construction.

Enforcement is a bounded re-splice: `get_new_blocks` is **not modified** and
stays a plain `popleft_n`. The policy acts by re-ordering the queue it pops
from. A banded free queue is the better data structure and was rejected on
blast radius — it replaces the structure every allocation and every free goes
through, so a banding bug is a lost or double-owned block rather than a
merely bad eviction, and flag-off would no longer reproduce upstream
behaviour (01 §6.5).
"""

import heapq
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.index import BlockOwnershipIndex
from vllm.v1.core.node_eviction.metrics import (
    CacheMovementTracker,
    EvictionObserver,
    GapStats,
    TTFTTracker,
)
from vllm.v1.core.node_eviction.scoring import (
    apply_speculative_floor,
    build_value_table,
    is_stale,
    score_key,
    speculative_floor,
)
from vllm.v1.core.node_eviction.snapshot import SnapshotSource
from vllm.v1.core.node_eviction.types import (
    EMPTY_SNAPSHOT,
    ImportanceSnapshot,
    NodeKey,
    ScoreBreakdown,
)
from vllm.v1.core.node_eviction.wantlist import PrefetchWant, PrefetchWantList

logger = init_logger(__name__)

# Used when the workflow does not send `call_type`, and when the config
# disables call-type keying because the flattening turned out not to be
# deterministic for the same physical prefix (03 §1, requirement 2).
DEFAULT_CALL_TYPE = ""

# How many `(job, node, call_type)` prefix sizes to remember. Sized against
# `index_hard_drop_age_ms`-worth of keys rather than tuned: the map is one
# small int per key and only exists so a fully-evicted prefix can still be
# priced, so over-retaining costs bytes while under-retaining costs the
# `prefetch_demand_mb` figure accuracy.
PREFIX_SIZE_CACHE_MAX_KEYS = 16_384


def node_key_for_request(
    request, use_call_type: bool = True
) -> NodeKey | None:
    """Read `(job_id, langgraph_node, call_type)` off the request.

    All three ride in `sampling_params.extra_args` and already cross the
    front-end -> engine-core boundary, so no protocol change and no new
    request field is needed (00 Part 4).

    Returns None when the request carries no identity: its blocks are then
    unscored, keep their LRU position, and the policy is inert for them.
    """
    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is None:
        return None
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return None
    job_id = extra_args.get("job_id")
    node = extra_args.get("langgraph_node")
    if job_id is None or node is None:
        return None
    call_type = extra_args.get("call_type") if use_call_type else None
    return NodeKey(
        str(job_id),
        str(node),
        str(call_type) if call_type is not None else DEFAULT_CALL_TYPE,
    )


def _is_prefetch_only(request) -> bool:
    """Discriminate a phantom prefetch from a real request.

    Mirrors `Scheduler._is_prefetch_only_request`; an index entry created by
    a phantom is a prediction rather than an observation and is stamped
    speculative (02 §5 part 1).
    """
    params = getattr(request, "kv_transfer_params", None)
    return bool(params and params.get("prefetch_only"))


class NodeEvictionController:
    """Owns the index, the value table and the splice."""

    def __init__(
        self,
        block_pool,
        config: NodeEvictionConfig,
        snapshot_source: SnapshotSource | None = None,
        num_kv_cache_groups: int = 1,
        block_size_bytes: int = 0,
        block_size_tokens: int = 0,
        kv_bytes_per_token: float = 0.0,
    ) -> None:
        config.validate()
        self.config = config
        self.block_pool = block_pool
        self.block_size_bytes = block_size_bytes
        self.block_size_tokens = block_size_tokens
        # KV bytes one token of context costs on *this* model, summed over
        # cache groups. Supplied by the caller, which reads it off the model's
        # own `KVCacheSpec` (page bytes / block tokens per group) rather than
        # inferring it from the two block figures above — those collapse a
        # hybrid model's differing block sizes into one number and would price
        # its tokens wrong. Falls back to that ratio when the caller has
        # nothing better, and to 0.0 when even that is unknown, in which case
        # every byte figure derived from it reports 0.0 rather than a guess.
        if kv_bytes_per_token <= 0.0 and block_size_bytes and block_size_tokens:
            kv_bytes_per_token = block_size_bytes / block_size_tokens
        self.kv_bytes_per_token = max(kv_bytes_per_token, 0.0)
        self.index = BlockOwnershipIndex()
        self.snapshot_source = snapshot_source
        self.observer = EvictionObserver(
            decision_log_path=config.decision_log_path,
            regret_window_ms=config.regret_window_ms,
            regret_buffer_size=config.regret_buffer_size,
        )

        # There is exactly one free queue, shared by every KV cache group,
        # and for hybrid models a cache hit at prefix position p requires a
        # hit in *every* group. Evicting one group's block at p therefore
        # destroys the hit at p for all groups while reclaiming a single
        # block. Compounding it, groups may have different block sizes, so
        # equal block counts are not equal value (01 §7).
        self.enabled = True
        if num_kv_cache_groups > 1:
            logger.warning(
                "Node-aware KV eviction is disabled: the model has %d KV "
                "cache groups and the policy is only correct for a single "
                "group. Falling back to LRU.",
                num_kv_cache_groups,
            )
            self.enabled = False

        self.observer.counters.policy_enabled = self.enabled

        # Scored value table, rebuilt on the tick. The per-request consult is
        # a lookup in this and nothing else.
        self._value_table: dict[NodeKey, ScoreBreakdown] = {}
        # Score attributed to each block at the last splice, so an eviction
        # can be logged with the terms that condemned the block.
        self._spliced_scores: dict[int, ScoreBreakdown] = {}
        self._spliced_ranks: dict[int, int] = {}

        self._last_snapshot: ImportanceSnapshot = EMPTY_SNAPSHOT
        self._last_tick_monotonic = 0.0
        self._last_gc_monotonic = 0.0
        self._last_restage_monotonic = 0.0
        self._evictions_since_tick = 0

        # Step 6. None when origination is off, which is the default: the
        # reorder half is free, the prefetch half buys prefill work.
        self._wants: PrefetchWantList | None = None
        self._last_prefetch_log_monotonic = 0.0
        self._last_prefetch_fingerprint: tuple[int, ...] | None = None

        # HBM accounting line. `_evictions_by_key` counts per window and is
        # cleared when the line is emitted: a job that finished stops being
        # named instead of pinning its `job_id` in a dict for the lifetime
        # of the server.
        self._last_hbm_log_monotonic = 0.0
        self._last_hbm_fingerprint: tuple[int, ...] | None = None
        self._evictions_by_key: dict[str, int] = {}

        # Largest prefix ever seen under each key, in blocks. Kept *outside*
        # the ownership index on purpose: `BlockOwnershipIndex` deletes an
        # entry the moment its last block is evicted (`index.py:133`), which
        # is precisely the state a key is in when it becomes something to
        # prefetch — so the index can never answer "how big is the thing we
        # are about to pull back in". Bounded LRU rather than unbounded: a
        # long-lived server sees a new `job_id` per workflow run forever.
        self._prefix_blocks: OrderedDict[NodeKey, int] = OrderedDict()

        # Gauges recomputed on the tick and read by `hbm_summary`, which runs
        # before the snapshot is consulted and must not do the walk itself.
        # `_ptn_*` summarise `time_to_next_call_ms` — the field the Redis
        # reader parses off `PROB` (`snapshot.py:257`), and the same quantity
        # the workload harness calls `t3_parsed_to_next_start_ns`: the wall
        # clock between a routing decision being parsed and the next node
        # starting, i.e. the budget a prefetch has to land inside.
        self._ptn_snapshot: ImportanceSnapshot | None = None
        self._ptn_n = 0
        self._ptn_mean_ms = 0.0
        self._ptn_p50_ms = 0.0
        self._ptn_min_ms = 0.0
        # Non-resident forecast demand: how much KV the forecast says will be
        # needed and HBM does not hold. `_unsized` is the honesty field —
        # keys whose prefix length was never observed contribute nothing to
        # the blocks total, so a large one means the GB figure is a floor.
        self._demand_keys = 0
        self._demand_blocks = 0
        self._demand_unsized = 0

        # Measurement epoch. Bumped by `reset_measurement`, which the
        # benchmark harness calls over `POST /v1/kv_metrics/reset` once its
        # warmup has finished. Every `kv_hbm` line carries the epoch it was
        # measured in, so a line from the warmup can never be mistaken for a
        # line from the run — the two are otherwise identical in shape.
        self.metrics_epoch = 0
        self._epoch_started_monotonic = time.monotonic()
        self._epoch_started_wall = time.time()
        self._epoch_label = "boot"

        # Hit rate and the movement number. The default window is one full
        # pool turnover: a block evicted longer ago than that is not
        # something this policy is still answerable for.
        self.movement = CacheMovementTracker(
            window_blocks=(
                config.remat_window_blocks
                if config.remat_window_blocks > 0
                else getattr(block_pool, "num_gpu_blocks", 0)
            ),
            block_size_bytes=block_size_bytes,
        )
        # What a miss actually costs, next to how often it happened.
        self.ttft = TTFTTracker()

        # The prefetch budget as *measured*, next to the published forecast.
        # `call_gap` is the server's idle window — one chat completion
        # finishing to the next arriving — which is the wall time a warm has
        # to land inside. `key_gap` is the same clock applied per eviction
        # key: how long between two calls of the same (job, node, call_type),
        # i.e. the quantity `time_to_next_call` is predicting.
        #
        # Phantom prefetches are excluded from both. They are traffic this
        # policy originated, arrive back-to-back with whatever triggered them,
        # and would collapse the measured gap toward zero — reporting that
        # prefetch has no time to work *because* prefetch is working.
        self.call_gap = GapStats()
        self.key_gap = GapStats()
        self._last_finish_wall: float | None = None
        self._last_arrival_by_key: OrderedDict[NodeKey, float] = OrderedDict()
        if (
            config.prefetch_wants_enabled
            and self.enabled
            and not config.observe_only
        ):
            self._wants = PrefetchWantList(
                max_outstanding=config.prefetch_max_outstanding,
                want_ttl_ms=config.prefetch_want_ttl_ms,
                resubmit_backoff_ms=config.prefetch_resubmit_backoff_ms,
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.snapshot_source is not None:
            self.snapshot_source.start()

    def close(self) -> None:
        if self.snapshot_source is not None:
            self.snapshot_source.close()
        self.observer.close()

    # -- hooks from BlockPool ---------------------------------------------

    def on_blocks_cached(
        self,
        request,
        blocks: Sequence,
        start_position: int,
    ) -> None:
        """`BlockPool.cache_full_blocks` just gave these blocks a hash.

        `blocks` arrives in prefix order, so the position of `blocks[i]` is
        `start_position + i`. Null blocks are skipped, which is why positions
        are stored sparsely.
        """
        if not self.enabled:
            return

        # Above the identity check on purpose. Movement is a property of the
        # cache, not of the workflow: a block rebuilt for an anonymous
        # request cost exactly as much as one rebuilt for a named node, and
        # the LRU baseline counts all of them. Skipping the unnamed ones here
        # would make the two runs' `remat_blocks` incomparable.
        for block in blocks:
            if not block.is_null:
                self.movement.on_block_cached(block.block_hash)

        key = node_key_for_request(request, self.config.use_call_type)
        if key is None:
            return

        speculative = _is_prefetch_only(request)
        pairs = [
            (block.block_id, start_position + i)
            for i, block in enumerate(blocks)
            if not block.is_null and block.block_hash is not None
        ]
        if not pairs:
            return

        now = time.monotonic()
        existing = self.index.get_entry(key)
        blocks_before = len(existing.positions) if existing is not None else 0
        entry = self.index.add_blocks(key, pairs, now, speculative=speculative)
        self._note_prefix_size(key, entry.run_len)
        if speculative:
            counters = self.observer.counters
            if existing is None:
                counters.speculative_created += 1
            # Counted in blocks as well as keys, because the waste ratio's
            # numerator is per-block: a 50-block prefix fully evicted before
            # confirmation is one wasted key but fifty wasted blocks, and
            # dividing one by the other reports a waste of 5000%.
            added = len(entry.positions) - blocks_before
            if added > 0:
                counters.speculative_blocks_created += added

        if self.snapshot_source is not None:
            self.snapshot_source.note_job(key.job_id)

    def on_block_evicted(self, block_id: int, num_free_fresh: int = 0) -> None:
        """A cached block just lost its prefix-cache identity.

        This is the decision side of every join in 07: the block is gone, and
        one frame later the next request for its key says whether that was a
        mistake.
        """
        if not self.enabled:
            return
        breakdown = self._spliced_scores.pop(block_id, None)
        rank_in_splice = self._spliced_ranks.pop(block_id, None)
        keys = list(self.index.owners_of(block_id))
        entries = [
            entry
            for entry in (self.index.get_entry(key) for key in keys)
            if entry is not None
        ]
        prefix_pos = None
        run_len = None
        if entries:
            prefix_pos = entries[0].positions.get(block_id)
            run_len = entries[0].run_len
        if breakdown is None and keys:
            breakdown = self._value_table.get(keys[0])

        # Provenance comes from the index, not from the score breakdown. The
        # breakdown only exists once a tick has scored the key, and a block
        # prefetched and evicted inside one tick period is exactly the case
        # the waste ratio is meant to catch — reading `breakdown.speculative`
        # alone silently drops it into the unscored bucket instead.
        #
        # `all`, not `any`: a block co-owned by a confirmed key was genuinely
        # used, and multi-owner `max` already protects it (01 §4). Counting
        # it as waste would blame the prefetch for a block that earned its
        # place.
        speculative = bool(entries) and all(e.speculative for e in entries)

        self.observer.record_eviction(
            block_id=block_id,
            keys=keys,
            breakdown=breakdown,
            now_ms=time.time() * 1000.0,
            prefix_pos=prefix_pos,
            run_len=run_len,
            rank_in_splice=rank_in_splice,
            snapshot_age_ms=self._snapshot_age_ms(),
            num_free_fresh=num_free_fresh,
            was_spliced=breakdown is not None and breakdown.scored,
            speculative=speculative,
        )
        # Attribute the eviction to the node that owned the block, so the
        # summary line can say *whose* prefix was destroyed rather than only
        # how many blocks went. A block with several owners is charged to
        # each: the eviction cost all of them their hit.
        for key in keys:
            label = f"{key.job_id}:{key.node}"
            self._evictions_by_key[label] = self._evictions_by_key.get(label, 0) + 1
        if not keys:
            self._evictions_by_key["<untracked>"] = (
                self._evictions_by_key.get("<untracked>", 0) + 1
            )

        # Remember the hash so a later re-cache of the same prefix can be
        # recognised as redone work. Read off the block rather than passed
        # in: `BlockPool` calls this *before* `reset_hash()`, so the hash is
        # still there, and taking it here keeps the hook signature stable.
        self.movement.on_block_evicted(self._hash_of(block_id))

        self.index.remove_block(block_id)
        self._evictions_since_tick += 1

    def _hash_of(self, block_id: int):
        blocks = getattr(self.block_pool, "blocks", None)
        if blocks is None or block_id >= len(blocks):
            return None
        return blocks[block_id].block_hash

    def on_cache_query(
        self,
        num_tokens: int,
        num_hits: int,
        request=None,
    ) -> None:
        """One prefix-cache lookup, from `KVCacheManager.get_computed_blocks`.

        Counted here rather than read off `PrefixCacheStats` because that
        object is drained by whoever polls the metrics loggers — reading it
        would make these numbers depend on whether anything else was
        scraping, and on `log_stats` being on at all.

        `request` is optional and used only to tell a phantom prefetch from
        real demand, the same discrimination `on_blocks_cached` and
        `on_prefix_hit` already make. Deriving it here rather than at the
        call site keeps the notion of a phantom inside this package.
        """
        if not self.enabled:
            return
        phantom = request is not None and _is_prefetch_only(request)
        self.movement.on_cache_query(num_tokens, num_hits, phantom)
        if request is not None and not phantom:
            # This hook is the one place a request is seen exactly once
            # (`kv_cache_manager.py:268` gates it on `cache_query_counted`),
            # which is what makes it the arrival edge. `on_prefix_hit` and
            # `on_blocks_cached` both fire repeatedly per request.
            self._note_call_arrival(request)

    def _note_call_arrival(self, request) -> None:
        """Close both gap clocks against this request's arrival.

        `arrival_time` is wall seconds set when the engine took the request
        (`request.py:94`), the same clock `on_request_finished` stamps, so the
        subtraction is meaningful across the two hooks.

        A negative result is dropped rather than clamped (`GapStats.record`):
        it means the next call arrived while the previous was still running,
        which is concurrency, not a budget of zero.
        """
        arrival = getattr(request, "arrival_time", None)
        if arrival is None:
            return
        if self._last_finish_wall is not None:
            self.call_gap.record((arrival - self._last_finish_wall) * 1000.0)
        key = node_key_for_request(request, self.config.use_call_type)
        if key is None:
            return
        previous = self._last_arrival_by_key.get(key)
        if previous is not None:
            self.key_gap.record((arrival - previous) * 1000.0)
        self._last_arrival_by_key[key] = arrival
        self._last_arrival_by_key.move_to_end(key)
        while len(self._last_arrival_by_key) > PREFIX_SIZE_CACHE_MAX_KEYS:
            self._last_arrival_by_key.popitem(last=False)

    def on_request_finished(self, request) -> None:
        """`KVCacheManager.free` — the request is done with its blocks.

        Called on preemption too, which is why the sample is gated on
        `ttft_recorded`: a preempted request keeps its original
        `first_token_ts`, so counting it twice would weight slow requests by
        how often they were preempted.
        """
        if not self.enabled:
            return
        # Before the TTFT guards below: the gap clock closes when the request
        # releases its blocks whether or not it produced a token, and a
        # preempted request re-finishing only moves the mark forward.
        if not _is_prefetch_only(request):
            self._last_finish_wall = time.time()
        if request.ttft_recorded:
            return
        first = getattr(request, "first_token_ts", None)
        if first is None:
            # Finished before producing a token — aborted, or a phantom with
            # nothing to emit. No prefill latency to attribute.
            return
        request.ttft_recorded = True
        self.ttft.record((first - request.arrival_time) * 1000.0)

    def on_reset_prefix_cache(self) -> None:
        if not self.enabled:
            return
        self.movement.on_reset()
        self.index.clear()
        self._value_table = {}
        self._spliced_scores.clear()
        self._spliced_ranks.clear()
        if self._wants is not None:
            # Every block hash in the pool has just been invalidated, so an
            # outstanding want's prefix is gone whether or not its phantom
            # landed. Keeping the want would block a re-offer for the whole
            # backoff, exactly when everything needs re-warming.
            self._wants.clear()

    # -- hooks from KVCacheManager ----------------------------------------

    def on_prefix_hit(self, request, blocks_per_group: Sequence[Sequence]) -> None:
        """A request matched cached blocks — `BlockPool.touch()` fired.

        That *is* the signal that a speculative prediction came true, so it
        flips the owning entries to confirmed. Gated on `not prefetch_only`,
        or one phantom confirms another's guess: the forecast marking its own
        homework (02 §5 part 2).

        Also the admission side of the regret counter: the key is already in
        hand, so checking it against recent evictions is one dict lookup.
        """
        if not self.enabled:
            return
        key = node_key_for_request(request, self.config.use_call_type)
        if key is None:
            return

        now_ms = time.time() * 1000.0
        self.observer.record_admission(key, now_ms)

        if _is_prefetch_only(request):
            return

        now = time.monotonic()
        confirmed = 0
        for group_blocks in blocks_per_group:
            for block in group_blocks:
                confirmed += self.index.confirm_block(block.block_id, now)
                # A touched block leaves the free queue, so whatever position
                # the splice gave it is gone. Dropping the bookkeeping here is
                # what lets `_splice` treat "still recorded" as "still parked
                # at the head" — without it a block that was hit and later
                # freed would be wrongly skipped and never re-spliced.
                self._spliced_scores.pop(block.block_id, None)
                self._spliced_ranks.pop(block.block_id, None)
        if confirmed:
            self.observer.counters.speculative_confirmed += confirmed

    def get_value(self, key: NodeKey | None) -> float | None:
        """The O(1) per-request consult (02 §2, step 3.5).

        One dict lookup of a precomputed value. A miss returns None, which
        callers read as "no opinion, use the default".
        """
        if key is None or not self.enabled:
            return None
        breakdown = self._value_table.get(key)
        return breakdown.score if breakdown is not None else None

    def get_value_for_request(self, request) -> float | None:
        return self.get_value(
            node_key_for_request(request, self.config.use_call_type)
        )

    # -- the tick ----------------------------------------------------------

    def maybe_tick(self) -> None:
        """Called once per scheduler step from `new_step_starts`.

        Runs inside the engine-core loop on the thread that owns `BlockPool`,
        so there is no locking and no background work here. Every gate below
        exists to make the common case a handful of comparisons.
        """
        if not self.enabled:
            return

        now = time.monotonic()
        if (now - self._last_tick_monotonic) * 1000.0 < self.config.tick_period_ms:
            return
        self._last_tick_monotonic = now

        counters = self.observer.counters
        counters.ticks_total += 1

        snapshot = (
            self.snapshot_source.get_snapshot()
            if self.snapshot_source is not None
            else EMPTY_SNAPSHOT
        )
        # Hoisted above the summary line, not left with the rest of the
        # snapshot handling below it: the line reports these gauges, and the
        # early returns underneath can skip a tick entirely, so computing
        # them after would publish a `parsed_to_next_node` belonging to a
        # snapshot that has already been superseded. Identity-gated inside,
        # so a tick where Redis published nothing costs one comparison.
        self._update_forecast_gauges(snapshot)

        # Above the early returns below: HBM occupancy and the splice volume
        # are exactly what needs reporting when the tick is skipping, since
        # a tick that never reaches the splice is the failure mode the line
        # is meant to expose.
        self._maybe_log_hbm_summary(now)

        # Skip when nothing can have changed: same snapshot object *and* no
        # eviction since the last tick means the ranking and the free queue
        # are both as we left them.
        if snapshot is self._last_snapshot and self._evictions_since_tick == 0:
            counters.ticks_skipped_unchanged += 1
            return
        self._last_snapshot = snapshot
        self._evictions_since_tick = 0

        counters.snapshot_revision = snapshot.revision
        counters.snapshot_stale_seconds = self._snapshot_age_ms() / 1000.0
        counters.index_keys = self.index.num_keys
        counters.index_blocks = self.index.num_blocks

        now_ms = time.time() * 1000.0
        self.observer.expire(now_ms)
        self._maybe_gc(now)

        # Deliberately *above* the fresh-block early return below. Wanting a
        # prefix is most useful precisely when there is free space to put it
        # in — that is the cheap case, not the one to skip — whereas the
        # splice only matters once cached blocks are being destroyed.
        self._rebuild_want_list(snapshot, now, now_ms)
        self._maybe_log_prefetch_summary(now)

        # Rule 1, measured **at the head** rather than over the whole queue.
        #
        # The rule is: never splice a cached block in front of a block that
        # would otherwise have been used for free. `appendleft` inserts at the
        # absolute head, so the only fresh blocks that can be jumped are the
        # ones already at the head — and `popleft` only ever takes from there.
        #
        # The whole-queue count `num_free_fresh` looks like it says the same
        # thing and does not. A block freed with no hash (a partial tail
        # block) is appended to the *tail* and counted as fresh, so a steady
        # workload parks a permanent floor of fresh-but-unreachable blocks
        # behind the cached ones. Measured on a real run: `num_free_fresh`
        # never dropped below 21 across 21k evictions, so with the documented
        # `fresh_skip_threshold=0` the tick skipped **every** time, the value
        # table was never built, and the policy silently degraded to LRU —
        # every score and rank in the decision log was null.
        head_fresh = self._head_fresh_run(self.config.fresh_skip_threshold + 1)
        if head_fresh > self.config.fresh_skip_threshold:
            counters.ticks_skipped_fresh += 1
            return

        self._rebuild_value_table(snapshot, now, now_ms)
        self._splice(now)

    def _head_fresh_run(self, limit: int) -> int:
        """How many blocks at the head of the free queue are fresh, capped.

        `limit` bounds the walk, so the default `fresh_skip_threshold=0` costs
        exactly one pointer dereference per tick — the same O(1) the whole-
        queue counter was chosen for, without its blind spot.
        """
        queue = self.block_pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        seen = 0
        while (
            seen < limit
            and block is not None
            and block is not tail
            and block.block_hash is None
        ):
            seen += 1
            block = block.next_free_block
        return seen

    def _note_prefix_size(self, key: NodeKey, run_len: int) -> None:
        """Remember how long this key's prefix got, in blocks.

        Monotonic per key, matching `NodeEntry.max_position`: the question
        this answers is "how much KV would putting this prefix back cost",
        and that is set by the longest form of the prefix the key ever had,
        not by whatever fraction happens to survive right now.
        """
        if run_len <= 0:
            return
        cache = self._prefix_blocks
        previous = cache.get(key)
        if previous is None or run_len > previous:
            cache[key] = run_len
        cache.move_to_end(key)
        while len(cache) > PREFIX_SIZE_CACHE_MAX_KEYS:
            cache.popitem(last=False)

    def _blocks_to_mb(self, blocks: int) -> float:
        """Blocks of KV as MB of HBM, or 0.0 when the page size is unknown.

        Same rule as `remat_mb`: reported rather than guessed. A fabricated
        byte count is worse than an absent one when the number's whole job is
        to be diffed between two runs.
        """
        return blocks * self.block_size_bytes / 1e6

    def _tokens_to_mb(self, tokens: int) -> float:
        """Tokens of context as MB of KV on this model.

        The conversion the block figures cannot do: phantom traffic is
        measured in tokens queried and tokens hit, never in blocks, because a
        request's hit is counted before any block is allocated to it.
        """
        return tokens * self.kv_bytes_per_token / 1e6

    def _update_forecast_gauges(self, snapshot: ImportanceSnapshot) -> None:
        """Summarise `time_to_next_call_ms` over the rows Redis published.

        Guarded on snapshot identity because the tick runs four times a
        second and the snapshot only moves when a publisher writes; the walk
        is O(rows) and the median sort O(rows log rows), neither of which
        belongs on every step.

        Zero-valued rows are excluded rather than averaged in. A publisher
        that has not measured a gap yet emits `time_to_next_call: 0`, and a
        mean pulled toward zero by unmeasured rows reads as "the prefetch
        budget is tiny" — the exact wrong conclusion, drawn from missing data
        rather than from the workload.
        """
        if snapshot is self._ptn_snapshot:
            return
        self._ptn_snapshot = snapshot
        values = sorted(
            row.time_to_next_call_ms
            for row in snapshot.rows.values()
            if row.time_to_next_call_ms > 0.0
        )
        self._ptn_n = len(values)
        if not values:
            self._ptn_mean_ms = 0.0
            self._ptn_p50_ms = 0.0
            self._ptn_min_ms = 0.0
            return
        self._ptn_mean_ms = sum(values) / len(values)
        self._ptn_p50_ms = values[len(values) // 2]
        # The tightest budget any predicted call is working with, which is
        # the one that decides whether prefetching can win at all.
        self._ptn_min_ms = values[0]

    def _is_resident(self, key: NodeKey) -> bool:
        """Is this key's prefix in HBM to the degree that warming it would
        buy nothing?

        Not `get_entry(key) is not None`. An entry survives on a single
        surviving block (`index.py:133` only deletes it when `positions`
        empties), so mere existence says "at least one block of this prefix
        is left", which is not the question. A prefix is worth anything only
        as a contiguous run from position 0, so a key holding 40 of its 900
        blocks is, for hit-rate purposes, absent.

        `run_len` comes from `max_position`, the highest position *ever*
        recorded, which never decreases — so the ratio falls as blocks are
        evicted rather than tracking the survivors.
        """
        entry = self.index.get_entry(key)
        if entry is None:
            return False
        threshold = self.config.prefetch_min_coverage
        if threshold <= 0.0:
            # Any surviving block counts: the original behaviour, kept
            # reachable for an A/B against it.
            return True
        run_len = entry.run_len
        if run_len <= 0:
            # Nothing was ever positioned, so there is no run to be a
            # fraction of. Treat presence as residency rather than dividing
            # by zero.
            return True
        return (entry.num_blocks / run_len) >= threshold

    def _rebuild_want_list(
        self, snapshot: ImportanceSnapshot, now: float, now_ms: float
    ) -> None:
        """Decide which prefixes should be pulled into HBM (02 §4, step 6).

        The diagram's admission question, in order:

            "after the value is assessed it checks the hbm if its present or
             not; if not present -> prefetch call"

        Presence is the ownership index: a key with an entry has blocks in
        HBM right now. Absence plus an imminent, likely call is a want.

        The other half of the diagram's rule — *"if it can't be admitted to
        HBM due to it being full: evict others in order of least value except
        this"* — needs no separate mechanism here. The splice has already
        moved the least valuable blocks to the head, so `get_new_blocks` pops
        those first.

        The "except this" clause used to be the speculative floor, which put
        a freshly prefetched prefix above the whole score range. That is off
        by default now (`config.speculative_floor_high`): a prefetch is
        ranked on the same row that justified fetching it, so a prefix worth
        warming is protected by its own score or not at all.
        """
        wants = self._wants
        counters = self.observer.counters
        if wants is not None:
            dropped_pending, dropped_outstanding = wants.expire(now)
            counters.prefetch_wants_expired += (
                dropped_pending + dropped_outstanding
            )

        # Demand accounting runs whether or not origination is on, because a
        # baseline arm with no want-list still has to report *how much* KV the
        # forecast said would be needed and HBM did not hold — that is the
        # quantity the policy arm is claiming to have moved, and a field that
        # only exists on one side cannot be diffed.
        demand_keys = 0
        demand_blocks = 0
        demand_unsized = 0

        cfg = self.config
        for key, row in snapshot.rows.items():
            if self._is_resident(key):
                # Already resident. Retiring the want here rather than on the
                # phantom's completion means it works no matter who put the
                # prefix there — the phantom, a real request, or a co-owning
                # node's identical preamble.
                if wants is not None and wants.note_satisfied(key):
                    counters.prefetch_wants_satisfied += 1
                continue
            if not cfg.prefetch_ignore_staleness and is_stale(
                row, now_ms, cfg.staleness_cutoff_ms
            ):
                # A stale row must not buy prefill work. The eviction half
                # degrades to LRU on staleness; this half degrades to doing
                # nothing, which is the same conservative direction. Off by
                # default — see `prefetch_ignore_staleness`.
                continue
            # Both gates below are disabled by default (0.0 admits every row,
            # since prob >= 0 and ttnc >= 0 always). They are kept as knobs
            # rather than deleted so a fractional forecast can switch them
            # back on without a code change.
            if cfg.prefetch_min_prob > 0.0 and row.prob < cfg.prefetch_min_prob:
                continue
            if (
                cfg.prefetch_horizon_ms > 0.0
                and row.time_to_next_call_ms > cfg.prefetch_horizon_ms
            ):
                continue

            # Past every admission gate: the forecast says this prefix will
            # be wanted and HBM does not have it. Counted here rather than
            # after `offer` so a want that was merely deduplicated against an
            # in-flight one still shows up as outstanding demand — the KV has
            # to move either way.
            demand_keys += 1
            blocks = self._prefix_blocks.get(key, 0)
            if blocks > 0:
                demand_blocks += blocks
            else:
                demand_unsized += 1

            if wants is None or wants.is_tracked(key):
                continue

            # `num_blocks=1` makes this `prob * decay * E_miss`: expected ms
            # saved, not ms saved per block. The density form is right for
            # ranking blocks already held against each other; for choosing
            # what to *fetch*, the prefix size is unknown until it lands, and
            # a big prefix is more worth fetching, not less.
            breakdown = score_key(row, 1, cfg)
            want = PrefetchWant(
                key=key,
                agent_id=f"{cfg.prefetch_agent_namespace}:{key.node}",
                score=breakdown.score,
                prob=breakdown.prob,
                time_to_next_call_ms=row.time_to_next_call_ms,
                created_at=now,
            )
            if wants.offer(want):
                counters.prefetch_wants_created += 1
            else:
                counters.prefetch_wants_dropped += 1

        self._demand_keys = demand_keys
        self._demand_blocks = demand_blocks
        self._demand_unsized = demand_unsized

        if wants is None:
            return
        counters.prefetch_wants_pending = wants.num_pending
        counters.prefetch_wants_outstanding = wants.num_outstanding

    def prefetch_summary(self) -> str | None:
        """One line of prefetch state, or None when origination is off.

        Both halves of the question, on one line: did the *instruction* reach
        HBM (`wants`), and was what landed actually *used* (`speculative`).
        Reporting only the first makes a forecast that predicts the wrong node
        look healthy; only the second cannot distinguish a wrong forecast from
        a phantom that never ran.
        """
        if self._wants is None:
            return None
        c = self.observer.counters
        return (
            "node_eviction prefetch: "
            f"wants created={c.prefetch_wants_created} "
            f"drained={c.prefetch_wants_drained} "
            f"satisfied={c.prefetch_wants_satisfied} "
            f"expired={c.prefetch_wants_expired} "
            f"dropped={c.prefetch_wants_dropped} "
            f"pending={self._wants.num_pending} "
            f"inflight={self._wants.num_outstanding} "
            f"hit_rate={c.prefetch_want_hit_rate:.0%} | "
            f"speculative keys={self.index.num_speculative_keys} "
            f"blocks={c.speculative_blocks_created} "
            f"confirmed={c.speculative_confirmed} "
            f"evicted_unconfirmed={c.speculative_evicted_before_confirm} "
            f"waste={c.speculative_waste:.0%}"
        )

    def _maybe_log_prefetch_summary(self, now: float) -> str | None:
        """Rate-limited *and* change-gated, so an idle server logs nothing.

        Without the change gate this would print an identical line every 30s
        forever, which trains everyone to filter it out — and the one time it
        matters is the time it changed.

        Returns the line it logged, or None. The return value exists so the
        gating is testable without capturing log output: `vllm`'s root logger
        sets `propagate=False`, so pytest's `caplog` sees nothing and a test
        written against it would pass vacuously.
        """
        if self._wants is None or self.config.prefetch_summary_period_ms <= 0:
            return None
        period_s = self.config.prefetch_summary_period_ms / 1000.0
        if now - self._last_prefetch_log_monotonic < period_s:
            return None

        c = self.observer.counters
        fingerprint = (
            c.prefetch_wants_created,
            c.prefetch_wants_drained,
            c.prefetch_wants_satisfied,
            c.prefetch_wants_expired,
            c.prefetch_wants_dropped,
            c.speculative_blocks_created,
            c.speculative_confirmed,
            c.speculative_evicted_before_confirm,
        )
        if fingerprint == self._last_prefetch_fingerprint:
            return None
        self._last_prefetch_log_monotonic = now
        self._last_prefetch_fingerprint = fingerprint
        summary = self.prefetch_summary()
        logger.info("%s", summary)
        return summary

    @property
    def _variant(self) -> str:
        """The arm this server is running, as it appears on the log line.

        Keyed on whether the splice *can* run, not on the observe flag alone:
        `splice_max_blocks: 0` in the JSON is behaviourally the same arm and
        used to report itself as `node_eviction`, which made the two runs
        indistinguishable in the field the timeline tool keys on.
        """
        return (
            "baseline"
            if self.config.observe_only or self.config.splice_max_blocks <= 0
            else "node_eviction"
        )

    def hbm_summary(self) -> str:
        """One line of HBM block accounting, in `key=value` form.

        The LRU baseline emits the *same* line with `variant=baseline` and
        `splices=0`, which is the whole point: the two runs are meant to be
        diffed field by field, and a format that drifts between them turns
        the comparison into manual reading.

        `top_evicted` names `job_id:node` because "we evicted 12k blocks" is
        not actionable and "we evicted 12k blocks, 9k of them job7:research"
        is — it says the forecast for one node is wrong, not that the cache
        is small.
        """
        pool = self.block_pool
        total = pool.num_gpu_blocks
        free = pool.get_num_free_blocks()
        used = total - free
        c = self.observer.counters
        m = self.movement
        top = sorted(
            self._evictions_by_key.items(), key=lambda kv: (-kv[1], kv[0])
        )[: self.config.hbm_summary_top_keys]
        top_str = ",".join(f"{label}={n}" for label, n in top) or "-"
        return (
            f"kv_hbm variant={self._variant} "
            f"total={total} used={used} free={free} "
            f"usage={(used / total * 100.0) if total else 0.0:.1f}% "
            f"queue={pool.free_block_queue.num_free_blocks} "
            f"splices={c.splices_total} "
            f"spliced_blocks={c.blocks_spliced_total} "
            # `staged` is how deep the policy's authority reaches into the
            # free queue; `deficit` is what it wanted to rank and could not.
            # Everything past `staged` is evicted by raw LRU age, so these
            # two say whether the scores are governing evictions at all.
            f"staged={c.splice_staged_blocks} "
            f"deficit={c.splice_deficit_blocks} "
            f"evicted={c.evictions_total} "
            f"evicted_by_score={c.evictions_by_score_total} "
            f"regret={c.regret_rate:.3f} "
            f"hit_rate={m.hit_rate:.4f} "
            f"hit_rate_win={m.window_hit_rate:.4f} "
            # Engine-side TTFT — what a miss cost, next to how often it
            # happened. Hit rate alone cannot separate a policy that kept
            # cheap-to-rebuild blocks from one that kept expensive ones.
            f"ttft_ms={self.ttft.mean_ms:.1f} "
            f"ttft_win_ms={self.ttft.window_mean_ms:.1f} "
            # The median sits between the two means and the tail on purpose:
            # read left to right it goes aggregate cost, this window, what a
            # typical request saw, what the worst 5% saw.
            f"ttft_p50_ms={self.ttft.p50_ms:.1f} "
            f"ttft_p95_ms={self.ttft.p95_ms:.1f} "
            f"ttft_n={self.ttft.count} "
            f"hit_tokens={m.hit_tokens} "
            f"query_tokens={m.query_tokens} "
            # Phantom traffic, excluded from every rate above. Reported so
            # the prefill origination bought is visible next to the hit rate
            # it was meant to raise, rather than hidden inside it.
            f"phantom_hit_rate={m.phantom_hit_rate:.4f} "
            f"phantom_query_tokens={m.phantom_query_tokens} "
            # What the phantoms actually cost in KV movement, priced on this
            # model's per-token KV footprint.
            #
            #   expected = every token the phantoms asked to have resident,
            #              i.e. the movement if HBM had held none of it. The
            #              ceiling the warm was worth at most.
            #   moved    = the tokens that missed and therefore had to be
            #              built, which is the KV that really crossed into
            #              HBM. Their difference is warming already-resident
            #              blocks: work the prefetch paid for and got nothing
            #              back from.
            #
            # Both are phantom-only. Real requests are excluded from every
            # phantom counter upstream (`metrics.py:334`), so these never
            # double-count demand traffic.
            f"phantom_kv_expected_mb={self._tokens_to_mb(m.phantom_query_tokens):.1f} "
            f"phantom_kv_moved_mb={self._tokens_to_mb(m.phantom_moved_tokens):.1f} "
            f"phantom_moved_tokens={m.phantom_moved_tokens} "
            f"remat_blocks={m.remat_blocks} "
            f"remat_mb={m.remat_mb:.1f} "
            f"remat_ratio={m.remat_ratio:.4f} "
            f"blocks_cached={m.blocks_cached} "
            # The prefetch budget, straight off the forecast rows the Redis
            # reader parsed. `min` is the field that decides feasibility: a
            # prefetch that takes longer than the shortest gap to the next
            # call cannot land in time no matter how good the ranking is.
            f"parsed_to_next_node_ms={self._ptn_mean_ms:.1f} "
            f"parsed_to_next_node_p50_ms={self._ptn_p50_ms:.1f} "
            f"parsed_to_next_node_min_ms={self._ptn_min_ms:.1f} "
            f"parsed_to_next_node_n={self._ptn_n} "
            # The same budget, measured instead of published. `call_gap` is
            # the idle window between one chat completion finishing and the
            # next arriving; `key_gap` is the interval between two calls of
            # the same eviction key, which is what the forecast above is
            # predicting. Phantoms excluded from both. Unlike the forecast
            # gauges, these are epoch-scoped, so a warmup's gaps never leak
            # into the measured run.
            f"call_gap_ms={self.call_gap.mean_ms:.1f} "
            f"call_gap_p50_ms={self.call_gap.p50_ms:.1f} "
            f"call_gap_min_ms={self.call_gap.min_ms:.1f} "
            f"call_gap_n={self.call_gap.count} "
            f"key_gap_ms={self.key_gap.mean_ms:.1f} "
            f"key_gap_p50_ms={self.key_gap.p50_ms:.1f} "
            f"key_gap_min_ms={self.key_gap.min_ms:.1f} "
            f"key_gap_n={self.key_gap.count} "
            # And what the *forecast* says has to move inside that budget —
            # unlike the phantom fields above, this is demand nobody has acted
            # on yet. MB is the comparable form: against HBM capacity, and
            # against L1->HBM bandwidth, which is what decides whether the
            # budget above is enough. `unsized` is how many of those keys had
            # no observed prefix length, i.e. how much of a floor the MB is.
            f"prefetch_demand_mb={self._blocks_to_mb(self._demand_blocks):.1f} "
            f"prefetch_demand_tokens={self._demand_blocks * self.block_size_tokens} "
            f"prefetch_blocks={self._demand_blocks} "
            f"prefetch_keys={self._demand_keys} "
            f"prefetch_unsized={self._demand_unsized} "
            f"index_keys={self.index.num_keys} "
            f"index_blocks={self.index.num_blocks} "
            # Which measurement epoch these numbers belong to, and how long it
            # has been running. `epoch=0` is everything before the harness
            # said its warmup was done; the run to report on is the highest
            # epoch present. Placed ahead of `top_evicted` because that field
            # is free-form (it contains `=` and `,`) and has to stay last.
            f"epoch={self.metrics_epoch} "
            f"epoch_age_s={self.epoch_age_s:.1f} "
            f"top_evicted={top_str}"
        )

    @property
    def epoch_age_s(self) -> float:
        """Seconds since the current measurement epoch began.

        This is the denominator for anything rate-like computed off the line,
        and the reason end-to-end timing does not have to include the warmup:
        the epoch clock restarts when the harness posts its reset.
        """
        return max(time.monotonic() - self._epoch_started_monotonic, 0.0)

    def reset_measurement(
        self,
        label: str = "",
        before_reset: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        """Discard everything measured so far and start a new epoch.

        The workload calls this over `POST /v1/kv_metrics/reset` once its
        warmup has populated the caches. It exists because a warmup pass
        corrupts every headline number three separate ways: the TTFT mean,
        median and p95 carry cold prefills no measured request paid; the hit
        rate is cumulative, so a cold pass permanently drags it down; and the
        elapsed window used for end-to-end figures starts at server boot
        rather than at the first measured request.

        `before_reset` is the hook the caller uses to flush HBM at the same
        boundary (`EngineCore.reset_kv_metrics(flush_hbm=True)`). It runs
        *after* the cold-phase numbers have been captured and *before* the
        counters are zeroed, which is the only ordering that works: flushing
        first would fire `on_reset_prefix_cache`, which zeroes the token
        counters itself, and the marker line would report a cold phase that
        looks like it never ran.

        With no hook, the KV cache is left alone — a warmup exists to fill it.
        With one, the resident blocks go but a KV connector's store does not,
        so the warm phase starts with an empty HBM cache in front of a
        populated CPU tier.

        The final pre-reset `kv_hbm` line is emitted first, ungated, so the
        cold phase's numbers survive in the log rather than being thrown away;
        then a `kv_hbm_reset` marker line names the boundary. Returns the
        discarded stats so the caller can file them with its own results.
        """
        # Ungated: `_maybe_log_hbm_summary` would suppress this if the period
        # had not elapsed or nothing had changed, and the one line guaranteed
        # to matter is the last one before the boundary.
        discarded_line = self.hbm_summary()
        logger.info("%s", discarded_line)

        discarded = self.stats()
        previous_epoch = self.metrics_epoch
        epoch_age_s = self.epoch_age_s

        # Whatever the hook does to the cache — including the counter writes
        # its own hooks make — is undone by the zeroing below, so the new
        # epoch starts at zero either way.
        hbm_flushed: str = "skipped"
        if before_reset is not None:
            hbm_flushed = "true" if before_reset() else "false"

        self.observer.reset_measurement()
        self.observer.counters.policy_enabled = self.enabled
        self.movement.reset_measurement()
        self.ttft.reset_measurement()
        self._evictions_by_key.clear()
        # Measured gaps are epoch-scoped: a warmup's idle windows are not the
        # measured run's budget. The last-seen marks go too, so the first gap
        # of the new epoch is not the boundary itself — which would be the
        # reset's own duration, including an HBM flush.
        self.call_gap.reset_measurement()
        self.key_gap.reset_measurement()
        self._last_finish_wall = None
        self._last_arrival_by_key.clear()

        self.metrics_epoch += 1
        self._epoch_started_monotonic = time.monotonic()
        self._epoch_started_wall = time.time()
        self._epoch_label = label or "reset"

        # So the first line of the new epoch is not withheld by the rate limit
        # or the change gate — with the counters at zero the fingerprint would
        # otherwise have to move before anything printed.
        self._last_hbm_log_monotonic = 0.0
        self._last_hbm_fingerprint = None

        marker = (
            f"kv_hbm_reset variant={self._variant} "
            f"epoch={self.metrics_epoch} "
            f"prev_epoch={previous_epoch} "
            f"label={self._epoch_label} "
            f"at={self._epoch_started_wall:.3f} "
            f"prev_epoch_age_s={epoch_age_s:.1f} "
            # Whether the resident blocks went with the counters. Reading a
            # hit rate without knowing this is meaningless: an epoch that
            # started with an empty HBM cache and one that inherited a full
            # one are not the same measurement.
            f"hbm_flushed={hbm_flushed} "
            f"discarded_ttft_n={discarded.get('ttft_n', 0)} "
            f"discarded_ttft_ms={discarded.get('ttft_ms', 0.0):.1f} "
            f"discarded_ttft_p50_ms={discarded.get('ttft_p50_ms', 0.0):.1f} "
            f"discarded_ttft_p95_ms={discarded.get('ttft_p95_ms', 0.0):.1f} "
            f"discarded_hit_rate={discarded.get('hit_rate', 0.0):.4f} "
            f"discarded_query_tokens={discarded.get('query_tokens', 0)} "
            f"discarded_evicted={discarded.get('evictions_total', 0)}"
        )
        logger.info("%s", marker)

        return {
            "ok": True,
            "variant": self._variant,
            "epoch": self.metrics_epoch,
            "prev_epoch": previous_epoch,
            "label": self._epoch_label,
            "at": self._epoch_started_wall,
            "prev_epoch_age_s": epoch_age_s,
            "hbm_flushed": hbm_flushed,
            "discarded_line": discarded_line,
            "discarded_stats": discarded,
            "marker": marker,
        }

    def _maybe_log_hbm_summary(self, now: float) -> str | None:
        """Rate-limited and change-gated, exactly like the prefetch line.

        Returns the line it logged, or None — `vllm`'s root logger sets
        `propagate=False`, so a test written against pytest's `caplog` would
        see nothing and pass vacuously.
        """
        if self.config.hbm_summary_period_ms <= 0:
            return None
        period_s = self.config.hbm_summary_period_ms / 1000.0
        if now - self._last_hbm_log_monotonic < period_s:
            return None

        c = self.observer.counters
        # Occupancy is deliberately *not* in the fingerprint: it moves by a
        # block on every step, so including it would defeat the change gate
        # and print a line every period forever.
        fingerprint = (
            c.splices_total,
            c.blocks_spliced_total,
            c.evictions_total,
            c.evictions_by_score_total,
            self.movement.query_tokens,
            # Phantom queries no longer move `query_tokens`, so without this
            # a window of pure origination activity would compare equal and
            # print nothing — origination invisible exactly when it is doing
            # the most work.
            self.movement.phantom_query_tokens,
            self.movement.remat_blocks,
            # A window where only latency moved is still worth a line.
            self.ttft.count,
            # A window where only the forecast moved is too: outstanding
            # prefetch demand changing is the input to every decision the
            # policy then makes, and an idle-looking cache with a moving
            # demand figure is exactly the state worth seeing on a timeline.
            # The integer block count, not the float GB or the ttnc mean —
            # those jitter on rounding and would defeat the gate.
            self._demand_blocks,
        )
        if fingerprint == self._last_hbm_fingerprint:
            return None
        self._last_hbm_log_monotonic = now
        self._last_hbm_fingerprint = fingerprint
        summary = self.hbm_summary()
        logger.info("%s", summary)
        # Per-window, so the next line names who is evicting *now* and shows
        # the hit rate *now* rather than one dominated by whatever the
        # workload happened to do first.
        self._evictions_by_key.clear()
        self.movement.reset_window()
        self.ttft.reset_window()
        return summary

    def drain_prefetch_wants(self, max_items: int) -> list[dict[str, str | float]]:
        """Hand pending wants to the front end (02 §4 option a).

        Reached over `call_utility`, which the engine-core busy loop dispatches
        on its own thread — the same one that runs the tick — so this needs no
        lock despite being called from another process.

        Returns plain dicts: the front end submits phantoms and has no reason
        to import engine-core types.
        """
        if self._wants is None or not self.enabled:
            return []
        drained = self._wants.drain(
            min(max_items, self.config.prefetch_max_per_drain), time.monotonic()
        )
        if drained:
            counters = self.observer.counters
            counters.prefetch_wants_drained += len(drained)
            counters.prefetch_wants_pending = self._wants.num_pending
            counters.prefetch_wants_outstanding = self._wants.num_outstanding
            logger.debug(
                "node_eviction: handed %d prefetch want(s) to the front end: %s",
                len(drained),
                [w.agent_id for w in drained],
            )
        return [want.as_dict() for want in drained]

    def _maybe_gc(self, now: float) -> None:
        if (now - self._last_gc_monotonic) * 1000.0 < self.config.index_gc_period_ms:
            return
        self._last_gc_monotonic = now
        # Speculative entries used to additionally drop at `TTL x multiple`.
        # That was the same `time_to_next_call_ms` clock the floor decay ran
        # on, so removing one and keeping the other would have left a
        # prefetched prefix protected by the floor right up until the GC
        # silently deleted its entry. Speculative and confirmed entries now
        # age out on one clock: `index_hard_drop_age_ms`.
        self.index.gc(
            now, hard_drop_age=self.config.index_hard_drop_age_ms / 1000.0
        )

    def _rebuild_value_table(
        self, snapshot: ImportanceSnapshot, now: float, now_ms: float
    ) -> None:
        key_block_counts = {
            entry.key: entry.num_blocks for entry in self.index.entries()
        }
        table = build_value_table(
            snapshot, key_block_counts, self.config, now_ms
        )

        # The speculative floor, applied after the base score so it can only
        # ever protect (max, not +). Off by default now — see
        # `config.speculative_floor_high` for why. With it off a prefetched
        # entry is simply absent from this loop and scored like any other
        # key: from its row if it has one, and not at all if it does not, in
        # which case it keeps its LRU position (Rule 2) rather than becoming
        # a first-out victim.
        #
        # The gate is on the height rather than on a separate bool because
        # `apply_speculative_floor` is a `max`: at 0 it would already be a
        # no-op for every scored key, and skipping the loop only additionally
        # stops it manufacturing table rows for unscored keys.
        if speculative_floor(self.config) <= 0.0:
            self._value_table = table
            return

        for entry in self.index.entries():
            if not entry.speculative:
                continue
            base = table.get(entry.key)
            if base is None:
                # No forecast for a predicted node: the floor alone drives
                # it. The score here is irrelevant to the ranking — the floor
                # is a `max` and always wins — so it is 0.0 rather than a
                # sentinel, and the row exists only to carry `blocks` and the
                # forecast terms into the decision log.
                row = snapshot.get(entry.key)
                base = ScoreBreakdown(
                    score=0.0,
                    prob=0.0,
                    ttnc_ms=(
                        row.time_to_next_call_ms if row is not None else 0.0
                    ),
                    decay=0.0,
                    e_miss_ms=0.0,
                    blocks=entry.num_blocks,
                )
            table[entry.key] = apply_speculative_floor(base, self.config)

        self._value_table = table

    def _splice(self, now: float) -> None:
        """Stage the worst-scoring free blocks at the head of the queue.

        Candidates come from the **index**, not from `get_all_free_blocks()`.
        That scan is O(F) — roughly 24k iterations of interpreted pointer
        chasing — and would dominate the K bound by about two orders of
        magnitude, so K would bound only the relink and not the tick. The
        index already holds every scored block and `ref_cnt == 0` tests
        freeness in O(1) (01 §6.4).

        The policy governs exactly the prefix of the free queue it has
        staged; past that, `popleft_n` evicts by raw LRU age and the scores
        have no say. So the goal is a staged region deep enough to cover a
        burst, which is what `splice_max_blocks` now allows for and
        `splice_deficit_blocks` measures.
        """
        if self.config.observe_only:
            # Enforced here rather than only by the `splice_max_blocks = 0`
            # that `from_env` forces: the guarantee belongs at the one place
            # that mutates the queue, or a config built any other way (a
            # test, the replay harness) would claim to be a baseline while
            # reordering.
            return
        k = self.config.splice_max_blocks
        if k <= 0 or not self._value_table:
            return

        # Re-stage: drop the record so every staged block is a candidate
        # again and one pass re-sorts the whole region by current score.
        # Without it the head is a stack of per-tick batches, newest in
        # front, and a block freed this tick outranks a worthless one staged
        # a minute ago — an inversion that only becomes visible once the
        # region is more than a batch deep.
        period_ms = self.config.splice_restage_period_ms
        if period_ms > 0 and (now - self._last_restage_monotonic) * 1000.0 >= (
            period_ms
        ):
            self._last_restage_monotonic = now
            self._spliced_scores.clear()
            self._spliced_ranks.clear()
            self.observer.counters.splice_restages_total += 1

        blocks = self.block_pool.blocks
        free_queue = self.block_pool.free_block_queue
        threshold = self.config.score_threshold

        # (score, -position, block_id): ascending score first, then higher
        # prefix position first. A node's prefix is only worth anything as a
        # contiguous run from position 0, so within one key the tail must go
        # before the head — freeing from the middle of a 1500-block prefix
        # reclaims one block and destroys the whole match (01 §6 Rule 3).
        candidates: list[tuple[float, int, int]] = []
        scored_blocks: dict[int, ScoreBreakdown] = {}

        for block_id, owners in self.index.tracked_block_owners():
            block = blocks[block_id]
            # A running node's blocks are never scored: ref_cnt > 0 keeps
            # them out of the free queue entirely (08 §5, invariant 8).
            if block.ref_cnt != 0 or block.is_null:
                continue
            if not free_queue.is_queued(block):
                continue
            # Already parked at the head by an earlier tick and not touched
            # since, so moving it again changes nothing. Measured on a real
            # run, omitting this check cost 2,054,400 relocations to produce
            # 47,363 evictions of spliced blocks — 43 moves per block that
            # actually went. The rest was the tick fighting itself: each
            # splice pulled a fresh worst-K to the front and pushed the
            # previous one back, so the queue never settled and LRU's
            # recency order was destroyed without anything replacing it.
            if block_id in self._spliced_scores:
                continue

            best: ScoreBreakdown | None = None
            best_position = -1
            for key in owners:
                breakdown = self._value_table.get(key)
                if breakdown is None:
                    continue
                # `max` over owners, not sum. A shared system preamble owned
                # by five nodes is worth what its most valuable owner makes
                # it worth; summing would make common prefixes unevictable
                # regardless of whether anything needs them (08 §4).
                if best is None or breakdown.score > best.score:
                    best = breakdown
                    entry = self.index.get_entry(key)
                    best_position = (
                        entry.positions.get(block_id, -1) if entry else -1
                    )
            if best is None:
                # Unscored: never selected, so it simply keeps its LRU
                # position — which is exactly what neutral should mean, with
                # no arithmetic (01 §6 Rule 2).
                continue
            if threshold is not None and best.score >= threshold:
                continue
            candidates.append((best.score, -best_position, block_id))
            scored_blocks[block_id] = best

        if not candidates:
            # Nothing left to stage — the steady state once the region is
            # deep enough. The gauges are still written, because a stale
            # deficit reading from an earlier tick is worse than none.
            self.observer.counters.splice_deficit_blocks = 0
            self.observer.counters.splice_staged_blocks = len(
                self._spliced_scores
            )
            return

        worst = heapq.nsmallest(k, candidates)

        # `appendleft_n` preserves order, so passing the candidates ascending
        # puts the worst-scoring block at the head, i.e. first out of
        # `popleft_n`. Getting this backwards evicts the *best* blocks first,
        # and the only symptom is an inverted hit rate — which is why the
        # score/outcome correlation in 07 §5.1 is not optional.
        ordered = []
        for rank, (_score, _neg_pos, block_id) in enumerate(worst):
            block = blocks[block_id]
            free_queue.remove(block)
            ordered.append(block)
            self._spliced_scores[block_id] = scored_blocks[block_id]
            self._spliced_ranks[block_id] = rank
        free_queue.appendleft_n(ordered)

        counters = self.observer.counters
        counters.splices_total += 1
        counters.blocks_spliced_total += len(ordered)
        # The window-1 gauge. Candidates the cap could not take this tick are
        # blocks the policy wanted to rank and left in raw LRU order, so a
        # sustained non-zero deficit means evictions are still being decided
        # by age past the staged region.
        counters.splice_deficit_blocks = len(candidates) - len(ordered)
        counters.splice_staged_blocks = len(self._spliced_scores)

    # -- reporting ---------------------------------------------------------

    def _snapshot_age_ms(self) -> float:
        snapshot = self._last_snapshot
        if snapshot.built_at_monotonic <= 0.0:
            return 0.0
        return (time.monotonic() - snapshot.built_at_monotonic) * 1000.0

    def stats(self) -> dict[str, float | int | bool]:
        out = dict(self.observer.counters.as_dict())
        total = self.block_pool.num_gpu_blocks
        free = self.block_pool.get_num_free_blocks()
        out["hbm_total_blocks"] = total
        out["hbm_free_blocks"] = free
        out["hbm_used_blocks"] = total - free
        out["hbm_usage"] = (total - free) / total if total else 0.0
        out["free_queue_len"] = self.block_pool.free_block_queue.num_free_blocks
        out.update(self.movement.as_dict())
        out.update(self.ttft.as_dict())
        out["index_keys"] = self.index.num_keys
        out["index_blocks"] = self.index.num_blocks
        out["index_speculative_keys"] = self.index.num_speculative_keys
        out["value_table_size"] = len(self._value_table)
        out["call_gap_ms"] = self.call_gap.mean_ms
        out["call_gap_p50_ms"] = self.call_gap.p50_ms
        out["call_gap_min_ms"] = self.call_gap.min_ms
        out["call_gap_n"] = self.call_gap.count
        out["key_gap_ms"] = self.key_gap.mean_ms
        out["key_gap_p50_ms"] = self.key_gap.p50_ms
        out["key_gap_min_ms"] = self.key_gap.min_ms
        out["key_gap_n"] = self.key_gap.count
        out["parsed_to_next_node_ms"] = self._ptn_mean_ms
        out["parsed_to_next_node_p50_ms"] = self._ptn_p50_ms
        out["parsed_to_next_node_min_ms"] = self._ptn_min_ms
        out["parsed_to_next_node_n"] = self._ptn_n
        out["prefetch_demand_mb"] = self._blocks_to_mb(self._demand_blocks)
        out["prefetch_demand_tokens"] = (
            self._demand_blocks * self.block_size_tokens
        )
        out["prefetch_blocks"] = self._demand_blocks
        out["prefetch_keys"] = self._demand_keys
        out["prefetch_unsized"] = self._demand_unsized
        # Phantom KV movement, priced per token on this model. `expected` is
        # every token the phantoms asked for, `moved` only the ones that
        # missed and had to be built.
        out["phantom_kv_expected_mb"] = self._tokens_to_mb(
            self.movement.phantom_query_tokens
        )
        out["phantom_kv_moved_mb"] = self._tokens_to_mb(
            self.movement.phantom_moved_tokens
        )
        out["block_size_tokens"] = self.block_size_tokens
        # The model's per-token KV footprint, so every MB figure above can be
        # rederived from the token counts without knowing the model.
        out["kv_bytes_per_token"] = self.kv_bytes_per_token
        # Which measurement window these numbers describe. Without it a
        # scraped sample cannot be told apart from one taken during warmup.
        out["metrics_epoch"] = self.metrics_epoch
        out["epoch_age_s"] = self.epoch_age_s
        out["observe_only"] = self.config.observe_only
        out["prefetch_origination_enabled"] = self._wants is not None
        if self._wants is not None:
            out["prefetch_wants_pending"] = self._wants.num_pending
            out["prefetch_wants_outstanding"] = self._wants.num_outstanding
        if self.snapshot_source is not None:
            for name, value in self.snapshot_source.stats().items():
                out[f"source_{name}"] = value
        return out
