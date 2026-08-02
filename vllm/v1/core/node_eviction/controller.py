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
from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.index import BlockOwnershipIndex
from vllm.v1.core.node_eviction.metrics import EvictionObserver
from vllm.v1.core.node_eviction.scoring import (
    apply_speculative_floor,
    build_value_table,
)
from vllm.v1.core.node_eviction.snapshot import SnapshotSource
from vllm.v1.core.node_eviction.types import (
    EMPTY_SNAPSHOT,
    ImportanceSnapshot,
    NodeKey,
    ScoreBreakdown,
)

logger = init_logger(__name__)

# Used when the workflow does not send `call_type`, and when the config
# disables call-type keying because the flattening turned out not to be
# deterministic for the same physical prefix (03 §1, requirement 2).
DEFAULT_CALL_TYPE = ""


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
    ) -> None:
        config.validate()
        self.config = config
        self.block_pool = block_pool
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
        self._evictions_since_tick = 0

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
        self.index.add_blocks(key, pairs, now, speculative=speculative)
        if speculative and existing is None:
            self.observer.counters.speculative_created += 1

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
        prefix_pos = None
        run_len = None
        if keys:
            entry = self.index.get_entry(keys[0])
            if entry is not None:
                prefix_pos = entry.positions.get(block_id)
                run_len = entry.run_len
        if breakdown is None and keys:
            breakdown = self._value_table.get(keys[0])

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
        )
        self.index.remove_block(block_id)
        self._evictions_since_tick += 1

    def on_reset_prefix_cache(self) -> None:
        if not self.enabled:
            return
        self.index.clear()
        self._value_table = {}
        self._spliced_scores.clear()
        self._spliced_ranks.clear()

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

        # Rule 1, an exact counter rather than an estimate. A fresh block is
        # exactly `block_hash is None`; using one costs nothing while
        # destroying a cached block costs its prefix. `appendleft` inserts at
        # the *absolute* head, so splicing a cached block in front of a fresh
        # one would throw away a prefix while a free unused block sat right
        # there. The previous attempt reverse-engineered this quantity from
        # aggregate counters and admitted in-code that getting it wrong
        # collapsed the cache to a single prompt (00 Part 2, weakness 3).
        num_free_fresh = self.block_pool.num_free_fresh
        if num_free_fresh > self.config.fresh_skip_threshold:
            counters.ticks_skipped_fresh += 1
            return

        self._rebuild_value_table(snapshot, now, now_ms)
        self._splice()

    def _maybe_gc(self, now: float) -> None:
        if (now - self._last_gc_monotonic) * 1000.0 < self.config.index_gc_period_ms:
            return
        self._last_gc_monotonic = now
        ttls = {
            key: row.time_to_next_call_ms
            for key, row in self._last_snapshot.rows.items()
        }
        self.index.gc(
            now,
            hard_drop_age=self.config.index_hard_drop_age_ms / 1000.0,
            speculative_ttls={k: v / 1000.0 for k, v in ttls.items()},
            speculative_ttl_multiple=(
                self.config.speculative_hard_drop_ttl_multiple
            ),
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
        # ever protect (max, not +). It ships with the splice, not later: the
        # splice is precisely what removes LRU's accidental protection of a
        # prefetched prefix (freed blocks go to the tail, and the tail is
        # evicted last), so without the floor this step can evict a prefix it
        # just paid ~11.4s to fetch (02 §5, build ordering constraint).
        for entry in self.index.entries():
            if not entry.speculative:
                continue
            row = snapshot.get(entry.key)
            ttl_ms = (
                row.time_to_next_call_ms
                if row is not None
                else self.config.speculative_default_ttl_ms
            )
            age_ms = (now - entry.created_at) * 1000.0
            base = table.get(entry.key)
            if base is None:
                # No forecast for a predicted node: the floor alone drives
                # it, which is what makes a falsified prediction the
                # preferred victim once the floor has decayed past zero.
                base = ScoreBreakdown(
                    score=self.config.speculative_floor_low,
                    prob=0.0,
                    ttnc_ms=ttl_ms,
                    decay=0.0,
                    e_miss_ms=0.0,
                    blocks=entry.num_blocks,
                )
            table[entry.key] = apply_speculative_floor(
                base, age_ms, ttl_ms, self.config
            )

        self._value_table = table

    def _splice(self) -> None:
        """Move the worst-scoring K free blocks to the head of the queue.

        Candidates come from the **index**, not from `get_all_free_blocks()`.
        That scan is O(F) — roughly 24k iterations of interpreted pointer
        chasing — and would dominate the K bound by about two orders of
        magnitude, so K would bound only the relink and not the tick. The
        index already holds every scored block and `ref_cnt == 0` tests
        freeness in O(1) (01 §6.4).
        """
        k = self.config.splice_max_blocks
        if k <= 0 or not self._value_table:
            return

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

    # -- reporting ---------------------------------------------------------

    def _snapshot_age_ms(self) -> float:
        snapshot = self._last_snapshot
        if snapshot.built_at_monotonic <= 0.0:
            return 0.0
        return (time.monotonic() - snapshot.built_at_monotonic) * 1000.0

    def stats(self) -> dict[str, float | int | bool]:
        out = dict(self.observer.counters.as_dict())
        out["index_keys"] = self.index.num_keys
        out["index_blocks"] = self.index.num_blocks
        out["index_speculative_keys"] = self.index.num_speculative_keys
        out["value_table_size"] = len(self._value_table)
        if self.snapshot_source is not None:
            for name, value in self.snapshot_source.stats().items():
                out[f"source_{name}"] = value
        return out
