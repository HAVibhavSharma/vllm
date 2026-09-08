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
from collections.abc import Callable, Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.index import BlockOwnershipIndex
from vllm.v1.core.node_eviction.metrics import (
    CacheMovementTracker,
    EvictionObserver,
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


def _agent_id_of(request) -> str | None:
    """The prefetch registry key this request records under, if it carries one.

    Rides in `sampling_params.extra_args` like the eviction identity: the
    agent chat endpoint keeps `agent_id` on the inner request precisely so it
    survives into the engine (see
    `AgentChatCompletionRequest.to_chat_completion_request`).
    """
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    agent_id = extra_args.get("agent_id")
    return str(agent_id) if agent_id else None


def _is_prefetch_only(request) -> bool:
    """Discriminate a phantom prefetch from a real request.

    Mirrors `Scheduler._is_prefetch_only_request`; an index entry created by
    a phantom is a prediction rather than an observation and is stamped
    speculative (02 §5 part 1).
    """
    params = getattr(request, "kv_transfer_params", None)
    return bool(params and params.get("prefetch_only"))


# What `OpenAIServingChat` prefixes every `/v1/chat/completions` request id
# with (`chatcmpl-<base id>`, and `chatcmpl-<base id>_<i>` for the `n>1`
# sub-requests). Nothing else the server admits carries it: `/v1/completions`
# uses `cmpl-`, the embedding and scoring paths their own prefixes, and a
# phantom prefetch the `agent_prefetch:` shape.
_CHAT_COMPLETION_ID_PREFIX = "chatcmpl-"


def _is_chat_completion(request) -> bool:
    """Whether this request came in through `/v1/chat/completions`.

    Decided from the request id because that is the only trace of the
    entrypoint that survives into engine core — the `Request` carries no
    route. The prefix is set in one place (`OpenAIServingChat`), so it is as
    stable as anything available here, but it is a convention rather than a
    contract: a caller that submits straight to `engine_client.generate` with
    an id of its own is not counted, which is the intent.

    A phantom prefetch is excluded even when it borrows the shape. It never
    produces a token a user waited on, so its latency is not TTFT in the
    sense this measurement is asking about.
    """
    if _is_prefetch_only(request):
        return False
    request_id = getattr(request, "request_id", None)
    return isinstance(request_id, str) and request_id.startswith(
        _CHAT_COMPLETION_ID_PREFIX
    )


class NodeEvictionController:
    """Owns the index, the value table and the splice."""

    def __init__(
        self,
        block_pool,
        config: NodeEvictionConfig,
        snapshot_source: SnapshotSource | None = None,
        num_kv_cache_groups: int = 1,
        block_size_bytes: int = 0,
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
        self._last_restage_monotonic = 0.0
        self._evictions_since_tick = 0

        # Rate limit + change gate for the speculative-block summary line.
        self._last_prefetch_log_monotonic = 0.0
        self._last_prefetch_fingerprint: tuple[int, ...] | None = None

        # Measurement epoch. Bumped by `reset_measurement`, which the
        # benchmark harness calls over `POST /v1/kv_metrics/reset` once its
        # warmup has finished. Every `kv_hbm_ttft` line carries the epoch it was
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
        if request is not None:
            # Kept for this request's own `kv_hbm_ttft` line. Both figures are
            # taken here, while they are still the prompt's: `num_tokens` grows
            # with the decode, so reading it at teardown would report a
            # different denominator than the one the hit was measured against.
            request.hbm_query_tokens = num_tokens
            request.hbm_local_hit_tokens = num_hits

    def on_external_cache_query(
        self,
        num_tokens: int,
        num_local_hits: int,
        num_external_hits: int,
        request=None,
    ) -> None:
        """The second half of the same lookup, from `Scheduler.schedule`.

        Split from `on_cache_query` because the two tiers resolve at
        different points: the local hit is known inside
        `KVCacheManager.get_computed_blocks`, the external one only after
        `connector.get_num_new_matched_tokens` replies, several branches
        later. Counting both here rather than plumbing the external figure
        back into the manager keeps the connector out of the manager's
        signature.

        Must be gated by the caller on `Request.cold_tokens_counted` for the
        same reason `on_cache_query` is gated on `cache_query_counted` — the
        block it is called from re-runs on every step a request spends in the
        waiting queue.
        """
        if not self.enabled:
            return
        phantom = request is not None and _is_prefetch_only(request)
        self.movement.on_external_cache_query(
            num_tokens, num_local_hits, num_external_hits, phantom
        )
        if request is not None:
            # Overwrites what `on_cache_query` stashed rather than adding to
            # it: this hook sees the same lookup one tier later, so its local
            # figure is the authoritative one and its `num_tokens` is the same
            # denominator. A request that never reaches here (no connector
            # configured) keeps the local-only stash and reports
            # `external_hit_tokens=0`, which is what actually happened.
            request.hbm_query_tokens = num_tokens
            request.hbm_local_hit_tokens = num_local_hits
            request.hbm_external_hit_tokens = num_external_hits

    def on_request_finished(self, request) -> None:
        """`KVCacheManager.free` — the request is done with its blocks.

        Emits this request's own TTFT, on its own line. Not an average: the
        aggregate line carried only how many samples were taken, and the
        samples themselves live here, one per request, so that any aggregate
        can be computed afterwards over whichever subset of requests is
        actually being asked about.

        Called on preemption too, which is why the sample is gated on
        `ttft_recorded`: a preempted request keeps its original
        `first_token_ts`, so emitting it twice would weight slow requests by
        how often they were preempted.

        Only `/v1/chat/completions` requests are sampled, unless
        `ttft_chat_completions_only` is off — see that setting for why the
        other traffic is a different population.
        """
        if not self.enabled or request.ttft_recorded:
            return
        if self.config.ttft_chat_completions_only and not _is_chat_completion(
            request
        ):
            # Not a chat completion, so not a latency a user waited on:
            # phantom prefetches, cache-warming submissions and the raw
            # `/v1/completions` path all land here. Counting them mixes two
            # populations in one figure, and the warming traffic is exactly
            # the traffic the policy adds, so it would flatter itself.
            return
        first = getattr(request, "first_token_ts", None)
        if first is None:
            # Finished before producing a token — aborted, or a phantom with
            # nothing to emit. No prefill latency to attribute.
            return
        request.ttft_recorded = True
        ttft_ms = (first - request.arrival_time) * 1000.0
        if not self.ttft.record(ttft_ms):
            # Rejected as impossible (a non-monotonic clock). Not logged
            # either: a negative latency in the log is worse than a gap.
            return
        if self.config.ttft_per_request_log:
            logger.info("%s", self._ttft_line(request, ttft_ms))

    def _ttft_line(self, request, ttft_ms: float) -> str:
        """One request's TTFT, in `key=value` shape.

        The token breakdown travels with the latency because the latency
        alone cannot be read: 400 ms is a fast cold prefill or a slow warm
        one depending on how much of the prompt had to be computed, and
        separating those is the entire question an eviction policy is being
        asked. `preempted` is here for the same reason — a preempted request
        keeps its first `first_token_ts`, so its sample is real, but it was
        paid for by scheduling pressure rather than by a cache miss.
        """
        query = getattr(request, "hbm_query_tokens", None)
        local = getattr(request, "hbm_local_hit_tokens", None) or 0
        external = getattr(request, "hbm_external_hit_tokens", None) or 0
        if query is None:
            # No lookup was ever recorded for this request — it never reached
            # `get_computed_blocks`. Reporting -1 rather than 0 keeps it from
            # reading as a genuine all-cold prefill.
            query = local = external = -1
            cold = -1
        else:
            # Clamped for the same reason the aggregate is: when the two
            # tiers' spans overlap, the reported hits can exceed the prompt.
            cold = max(query - local - external, 0)
        return (
            f"kv_hbm_ttft variant={self._variant} "
            f"epoch={self.metrics_epoch} "
            f"req={request.request_id} "
            # The graph position this request came from. `node_key_for_request`
            # already keys eviction on (job, node, call_type); the agent id is
            # the same identity in the form the prefetch registry uses, so a
            # warm and the request it was for can be joined by string equality
            # instead of by reconstructing one from the other.
            f"agent={_agent_id_of(request) or '-'} "
            f"ttft_ms={ttft_ms:.1f} "
            f"query_tokens={query} "
            f"hit_tokens={local} "
            f"external_hit_tokens={external} "
            f"cold_tokens={cold} "
            f"preempted={getattr(request, 'num_preemptions', 0)} "
            # Phantom prefetches are excluded from every demand-side rate
            # line, so their latency is flagged rather than dropped: it is
            # real work the policy originated, and hiding it would make
            # origination look free.
            f"phantom={int(_is_prefetch_only(request))}"
        )

    def on_reset_prefix_cache(self) -> None:
        if not self.enabled:
            return
        self.movement.on_reset()
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

    def prefetch_summary(self) -> str | None:
        """One line on what prefetching bought, or None when the policy is off.

        Origination now lives entirely outside the engine: phantoms arrive
        over `POST /v1/agents/prefetch`, so the engine no longer knows what
        was *asked* for and can only report what *landed* — blocks stamped
        speculative, and whether a real request ever came for them.
        `waste` is the honest read on whether the caller's predictions are
        worth the prefills they cost.
        """
        if not self.enabled:
            return None
        c = self.observer.counters
        return (
            "node_eviction prefetch: "
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
        if not self.enabled or self.config.prefetch_summary_period_ms <= 0:
            return None
        period_s = self.config.prefetch_summary_period_ms / 1000.0
        if now - self._last_prefetch_log_monotonic < period_s:
            return None

        c = self.observer.counters
        fingerprint = (
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

        A `kv_hbm_reset` marker line names the boundary, and the discarded
        stats are returned so the caller can file them with its own results --
        which is where the cold phase's numbers survive now that the periodic
        accounting line is gone.
        """
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

        self.metrics_epoch += 1
        self._epoch_started_monotonic = time.monotonic()
        self._epoch_started_wall = time.time()
        self._epoch_label = label or "reset"

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
            # How many TTFT samples the discarded epoch held. Their latencies
            # are not summarised here — they were already logged one per
            # request, under the epoch number this marker is retiring, which
            # is what makes them separable after the fact.
            f"discarded_ttft_n={discarded.get('ttft_n', 0)} "
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
        # Which measurement window these numbers describe. Without it a
        # scraped sample cannot be told apart from one taken during warmup.
        out["metrics_epoch"] = self.metrics_epoch
        out["epoch_age_s"] = self.epoch_age_s
        out["observe_only"] = self.config.observe_only
        if self.snapshot_source is not None:
            for name, value in self.snapshot_source.stats().items():
                out[f"source_{name}"] = value
        return out
