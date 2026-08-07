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
    is_stale,
    score_key,
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
        if config.prefetch_wants_enabled and self.enabled:
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

        self.index.remove_block(block_id)
        self._evictions_since_tick += 1

    def on_reset_prefix_cache(self) -> None:
        if not self.enabled:
            return
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

        # Above the early returns below: HBM occupancy and the splice volume
        # are exactly what needs reporting when the tick is skipping, since
        # a tick that never reaches the splice is the failure mode the line
        # is meant to expose.
        self._maybe_log_hbm_summary(now)

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
        self._splice()

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
        those first; and "except this" is the speculative floor, which puts a
        freshly prefetched prefix above the whole score range (02 §5 part 3).
        """
        wants = self._wants
        if wants is None:
            return

        counters = self.observer.counters
        dropped_pending, dropped_outstanding = wants.expire(now)
        counters.prefetch_wants_expired += dropped_pending + dropped_outstanding

        cfg = self.config
        for key, row in snapshot.rows.items():
            if self.index.get_entry(key) is not None:
                # Already resident. Retiring the want here rather than on the
                # phantom's completion means it works no matter who put the
                # prefix there — the phantom, a real request, or a co-owning
                # node's identical preamble.
                if wants.note_satisfied(key):
                    counters.prefetch_wants_satisfied += 1
                continue
            if wants.is_tracked(key):
                continue
            if is_stale(row, now_ms, cfg.staleness_cutoff_ms):
                # A stale row must not buy prefill work. The eviction half
                # degrades to LRU on staleness; this half degrades to doing
                # nothing, which is the same conservative direction.
                continue
            if row.prob < cfg.prefetch_min_prob:
                continue
            if row.time_to_next_call_ms > cfg.prefetch_horizon_ms:
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
        top = sorted(
            self._evictions_by_key.items(), key=lambda kv: (-kv[1], kv[0])
        )[: self.config.hbm_summary_top_keys]
        top_str = ",".join(f"{label}={n}" for label, n in top) or "-"
        return (
            "kv_hbm variant=node_eviction "
            f"total={total} used={used} free={free} "
            f"usage={(used / total * 100.0) if total else 0.0:.1f}% "
            f"queue={pool.free_block_queue.num_free_blocks} "
            f"splices={c.splices_total} "
            f"spliced_blocks={c.blocks_spliced_total} "
            f"evicted={c.evictions_total} "
            f"evicted_by_score={c.evictions_by_score_total} "
            f"regret={c.regret_rate:.3f} "
            f"index_keys={self.index.num_keys} "
            f"index_blocks={self.index.num_blocks} "
            f"top_evicted={top_str}"
        )

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
        )
        if fingerprint == self._last_hbm_fingerprint:
            return None
        self._last_hbm_log_monotonic = now
        self._last_hbm_fingerprint = fingerprint
        summary = self.hbm_summary()
        logger.info("%s", summary)
        # Per-window, so the next line names who is evicting *now*.
        self._evictions_by_key.clear()
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
        total = self.block_pool.num_gpu_blocks
        free = self.block_pool.get_num_free_blocks()
        out["hbm_total_blocks"] = total
        out["hbm_free_blocks"] = free
        out["hbm_used_blocks"] = total - free
        out["hbm_usage"] = (total - free) / total if total else 0.0
        out["free_queue_len"] = self.block_pool.free_block_queue.num_free_blocks
        out["index_keys"] = self.index.num_keys
        out["index_blocks"] = self.index.num_blocks
        out["index_speculative_keys"] = self.index.num_speculative_keys
        out["value_table_size"] = len(self._value_table)
        out["prefetch_origination_enabled"] = self._wants is not None
        if self._wants is not None:
            out["prefetch_wants_pending"] = self._wants.num_pending
            out["prefetch_wants_outstanding"] = self._wants.num_outstanding
        if self.snapshot_source is not None:
            for name, value in self.snapshot_source.stats().items():
                out[f"source_{name}"] = value
        return out
