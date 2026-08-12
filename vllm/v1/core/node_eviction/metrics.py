# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observability — plan/new-eviction/07-observability.md.

The principle: **ground truth is free, one frame later.** When a request
arrives and hits or misses the prefix cache, the engine learns whether the
block it evicted was actually needed. Nothing here requires new measurement,
only that decisions be written down.

This is not optional infrastructure. Almost every failure in this design is
silent — a policy with its sign inverted produces no error, no crash and no
log line, just a hit rate that looks like LRU's while burning a Redis
connection, a thread and a per-tick splice. The previous attempt shipped
unconditional `print()` to stderr and no counters.

Two artifacts, per §6:

- **always-on counters** — O(1) increments, answer "is something wrong?"
- **the decision log** — one JSONL line per evicted cached block, behind a
  flag because it is unbounded under pressure, answers "what exactly, and
  why?"
"""

import json
import math
from collections import deque
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.node_eviction.types import NodeKey, ScoreBreakdown

logger = init_logger(__name__)


@dataclass(slots=True)
class _RecentEviction:
    key: NodeKey
    score: float
    ts_ms: float
    e_miss_ms: float
    # Set once this eviction has been classified as "needed again", so
    # ageing it out of the buffer cannot also count it in the negative
    # class of the score/outcome correlation.
    counted: bool = False


@dataclass(slots=True)
class EvictionCounters:
    """The minimum always-on set (07 §6)."""

    policy_enabled: bool = False

    ticks_total: int = 0
    ticks_skipped_fresh: int = 0
    ticks_skipped_unchanged: int = 0
    splices_total: int = 0
    blocks_spliced_total: int = 0
    splice_restages_total: int = 0
    # Gauges, not totals: the state of the staged region at the last splice.
    # `splice_deficit_blocks` is the window-1 gauge — blocks the policy
    # wanted to rank and the per-tick cap could not take, which are therefore
    # left to be evicted by raw LRU age.
    splice_deficit_blocks: int = 0
    splice_staged_blocks: int = 0

    evictions_total: int = 0
    evictions_by_score_total: int = 0
    unscored_evictions_total: int = 0

    evicted_then_needed_total: int = 0
    evicted_then_needed_cost_ms: float = 0.0

    # Keys stamped speculative...
    speculative_created: int = 0
    speculative_confirmed: int = 0
    # ...and the blocks under them. The waste ratio needs the second: its
    # numerator counts blocks, and a key owns as many blocks as its prefix
    # is long.
    speculative_blocks_created: int = 0
    speculative_evicted_before_confirm: int = 0

    # Prefetch origination (02 §4, step 6).
    prefetch_wants_created: int = 0
    prefetch_wants_dropped: int = 0
    prefetch_wants_drained: int = 0
    prefetch_wants_satisfied: int = 0
    prefetch_wants_expired: int = 0
    prefetch_wants_pending: int = 0
    prefetch_wants_outstanding: int = 0

    snapshot_stale_seconds: float = 0.0
    snapshot_revision: int = 0
    index_keys: int = 0
    index_blocks: int = 0

    # Score/outcome correlation inputs (07 §5.1). Accumulated as running
    # sums so the correlation is a division, not a second pass over a log.
    needed_score_sum: float = 0.0
    needed_count: int = 0
    not_needed_score_sum: float = 0.0
    not_needed_count: int = 0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "policy_enabled": self.policy_enabled,
            "ticks_total": self.ticks_total,
            "ticks_skipped_fresh": self.ticks_skipped_fresh,
            "ticks_skipped_unchanged": self.ticks_skipped_unchanged,
            "splices_total": self.splices_total,
            "blocks_spliced_total": self.blocks_spliced_total,
            "splice_restages_total": self.splice_restages_total,
            "splice_deficit_blocks": self.splice_deficit_blocks,
            "splice_staged_blocks": self.splice_staged_blocks,
            "evictions_total": self.evictions_total,
            "evictions_by_score_total": self.evictions_by_score_total,
            "unscored_evictions_total": self.unscored_evictions_total,
            "unscored_eviction_ratio": self.unscored_eviction_ratio,
            "evicted_then_needed_total": self.evicted_then_needed_total,
            "evicted_then_needed_cost_ms": self.evicted_then_needed_cost_ms,
            "regret_rate": self.regret_rate,
            "speculative_created": self.speculative_created,
            "speculative_confirmed": self.speculative_confirmed,
            "speculative_blocks_created": self.speculative_blocks_created,
            "speculative_evicted_before_confirm": (
                self.speculative_evicted_before_confirm
            ),
            "speculative_waste": self.speculative_waste,
            "prefetch_wants_created": self.prefetch_wants_created,
            "prefetch_wants_dropped": self.prefetch_wants_dropped,
            "prefetch_wants_drained": self.prefetch_wants_drained,
            "prefetch_wants_satisfied": self.prefetch_wants_satisfied,
            "prefetch_wants_expired": self.prefetch_wants_expired,
            "prefetch_wants_pending": self.prefetch_wants_pending,
            "prefetch_wants_outstanding": self.prefetch_wants_outstanding,
            "prefetch_want_hit_rate": self.prefetch_want_hit_rate,
            "score_outcome_correlation": self.score_outcome_correlation,
            "snapshot_stale_seconds": self.snapshot_stale_seconds,
            "snapshot_revision": self.snapshot_revision,
            "index_keys": self.index_keys,
            "index_blocks": self.index_blocks,
        }

    @property
    def unscored_eviction_ratio(self) -> float:
        if self.evictions_total == 0:
            return 0.0
        return self.unscored_evictions_total / self.evictions_total

    @property
    def regret_rate(self) -> float:
        if self.evictions_by_score_total == 0:
            return 0.0
        return self.evicted_then_needed_total / self.evictions_by_score_total

    @property
    def speculative_waste(self) -> float:
        """`evicted_before_confirm / created` — this ratio *is* prefetch
        waste, and it is the only honest read on whether the forecast is
        worth anything. Without it, a policy that protects garbage for a full
        TTL looks identical to one that works (02 §5).

        Both sides are counted in **blocks**. 02 §5 writes the denominator as
        "created", which reads naturally as the number of predictions, but
        the numerator can only be per-block: eviction happens one block at a
        time. Dividing blocks by keys made a single fully wasted 50-block
        prefix report 5000% waste, so the fraction is taken over blocks and
        the key counts are reported separately.
        """
        if self.speculative_blocks_created == 0:
            return 0.0
        return (
            self.speculative_evicted_before_confirm
            / self.speculative_blocks_created
        )

    @property
    def prefetch_want_hit_rate(self) -> float:
        """How often a want was answered before it aged out.

        Distinct from `speculative_waste`, which asks whether a *landed*
        prefetch was used. This asks the prior question — whether the
        instruction reached HBM at all — and separates "the forecast was
        wrong" from "the phantom never ran" (02 §4).
        """
        answered = self.prefetch_wants_satisfied
        total = answered + self.prefetch_wants_expired
        if total == 0:
            return 0.0
        return answered / total

    @property
    def score_outcome_correlation(self) -> float:
        """**The one number** (07 §5.1).

        Mean score of blocks whose key was requested again within the
        horizon, minus the mean score of those that were not, normalised by
        their sum. If blocks that were needed again scored *lower* than
        blocks that were not, the policy is anti-correlated — actively worse
        than LRU — and this goes negative.

        A near-zero value says the formula carries no signal. A negative one
        says ship the off switch today.
        """
        if self.needed_count == 0 or self.not_needed_count == 0:
            return 0.0
        needed_mean = self.needed_score_sum / self.needed_count
        not_needed_mean = self.not_needed_score_sum / self.not_needed_count
        denom = abs(needed_mean) + abs(not_needed_mean)
        if denom == 0.0:
            return 0.0
        return (needed_mean - not_needed_mean) / denom


class CacheMovementTracker:
    """Prefix-cache effectiveness, and the KV volume eviction forced back in.

    Two numbers, and the second is the one an eviction policy is actually
    judged on.

    **Hit rate** answers "did the cache work". It is necessary but not
    sufficient: a bigger cache raises it too, so on its own it cannot
    separate a better policy from more HBM.

    **Rematerialisation** answers "how much work did we redo". A block is
    counted when its hash is cached *again* after having been evicted — the
    block existed, we threw it away, we rebuilt it. That is movement the
    policy caused and a better policy avoids.

    Deliberately not the miss count. Misses include every prefix the server
    has never seen, which no eviction policy can do anything about; counting
    them would credit a policy for a cold start and bury the signal under
    first-contact traffic. Rematerialisation is zero for a first-time prefix
    by construction.

    The evicted-hash ring is what bounds the memory. A block evicted more
    than one full pool turnover ago is not something the policy is still
    responsible for, so the window defaults to the pool size.

    **Phantom prefetches are counted separately, never in the headline
    rates.** A phantom queries the prefix cache like any other request, so
    left alone it lands in both numerator and denominator — and not with a
    consistent sign: its first prefill is a near-total miss that deflates the
    rate, while a phantom for an already-resident prefix is a large hit no
    user ever experienced. Since the whole point of the line is a field-by-
    field diff against an LRU baseline that originates no phantoms, mixing
    them breaks the comparison it exists to serve. `phantom_hit_rate` keeps
    the number visible, because a phantom that misses is a full prefill
    serialized ahead of real traffic (12 §5.6).
    """

    def __init__(self, window_blocks: int, block_size_bytes: int = 0) -> None:
        self.window_blocks = max(int(window_blocks), 0)
        self.block_size_bytes = block_size_bytes

        # Ring of hashes that were cached and then evicted. `set` for the
        # O(1) membership test on the caching path, `deque` for the eviction
        # order the ring needs; they are kept in step.
        self._evicted: deque = deque()
        self._evicted_set: set = set()

        self._reset_totals()

    def _reset_totals(self) -> None:
        """Zero everything measured, leaving the configuration alone.

        Shared by `__init__` and `on_reset` so a counter added to one cannot
        silently go missing from the other — the failure mode there is a
        number that survives a cache wipe and quietly describes a cache that
        no longer exists.
        """
        self.remat_blocks = 0
        self.blocks_cached = 0
        self.hit_tokens = 0
        self.query_tokens = 0
        # Reset when a summary line is emitted, so the line can show the
        # rate *now* next to the rate since boot. A cumulative hit rate over
        # a long run is dominated by whatever the workload did first.
        self.window_hit_tokens = 0
        self.window_query_tokens = 0
        # Phantom prefetch traffic, kept out of every rate above.
        self.phantom_hit_tokens = 0
        self.phantom_query_tokens = 0

    def on_cache_query(
        self,
        num_tokens: int,
        num_hits: int,
        phantom: bool = False,
    ) -> None:
        if phantom:
            # Deliberately the only bucket a phantom touches: it is warming
            # work the policy originated, not demand the cache served.
            self.phantom_query_tokens += num_tokens
            self.phantom_hit_tokens += num_hits
            return
        self.query_tokens += num_tokens
        self.hit_tokens += num_hits
        self.window_query_tokens += num_tokens
        self.window_hit_tokens += num_hits

    def on_block_cached(self, block_hash) -> None:
        self.blocks_cached += 1
        if block_hash is None:
            return
        if block_hash in self._evicted_set:
            self.remat_blocks += 1
            # Discard rather than leave it: the hash is resident again, so a
            # *second* remat requires a second eviction. Leaving it would
            # count every subsequent cache of the same prefix as movement.
            self._evicted_set.discard(block_hash)

    def on_block_evicted(self, block_hash) -> None:
        if block_hash is None or self.window_blocks == 0:
            return
        # The set is the membership test; the deque only supplies eviction
        # order. A hash that was counted and discarded from the set can be
        # appended again, leaving a stale duplicate in the deque — it pops
        # early and shortens the window slightly for that one hash. The error
        # is bounded and conservative (it can only *under*-count movement),
        # which is the right direction for a number used to claim an
        # improvement.
        if block_hash not in self._evicted_set:
            self._evicted.append(block_hash)
            self._evicted_set.add(block_hash)
        while len(self._evicted) > self.window_blocks:
            self._evicted_set.discard(self._evicted.popleft())

    def on_reset(self) -> None:
        # Every hash in the pool was just invalidated, so nothing in the ring
        # can be rematerialised — a later cache of the same prefix is new
        # work, not redone work.
        self._evicted.clear()
        self._evicted_set.clear()
        # And the rates go with it. `reset_prefix_cache` is only ever invoked
        # explicitly, which in practice means "start a clean measurement", so
        # tokens counted against the old cache must not carry into the new
        # one. This mirrors upstream: `CachingMetrics.observe` calls its own
        # `reset()` on `stats.reset` (`v1/metrics/stats.py:68`), so leaving
        # these standing would also put the two numbers permanently out of
        # step after the first reset.
        self._reset_totals()

    @property
    def hit_rate(self) -> float:
        """Demand-side hit rate, one query counted per request.

        **Deliberately not comparable to vLLM's `Prefix cache hit rate`
        line.** That one counts every call to `get_computed_blocks`, and the
        scheduler makes one per step for as long as a request sits in the
        waiting queue — so its denominator is weighted by queueing delay,
        which eviction pressure itself sets. Measured on the 2026-08-09 run
        that came to 779M query tokens against ~440k real prompt tokens, a
        factor of ~1771. The caller counts each request once
        (`Request.cache_query_counted`); upstream's counter is left alone so
        it stays comparable to stock vLLM. Expect the two numbers to differ,
        and expect this one to be the meaningful one.
        """
        if self.query_tokens == 0:
            return 0.0
        return self.hit_tokens / self.query_tokens

    @property
    def window_hit_rate(self) -> float:
        if self.window_query_tokens == 0:
            return 0.0
        return self.window_hit_tokens / self.window_query_tokens

    @property
    def phantom_hit_rate(self) -> float:
        """How often origination fetched something already there.

        High is not good here: it means the residency test admitted a want
        for a prefix HBM still held, and the phantom bought nothing. Low with
        a large `phantom_query_tokens` is the expensive case — real prefills
        the policy chose to run. Either way it belongs next to the real rate,
        not inside it.
        """
        if self.phantom_query_tokens == 0:
            return 0.0
        return self.phantom_hit_tokens / self.phantom_query_tokens

    @property
    def remat_mb(self) -> float:
        """The movement number, in MB of KV actually rebuilt.

        Zero when the block size is unknown — reported rather than guessed,
        because a fabricated byte count is worse than an absent one when the
        whole point is comparing two runs.
        """
        return self.remat_blocks * self.block_size_bytes / 1e6

    @property
    def remat_ratio(self) -> float:
        """Fraction of all caching work that was redoing work."""
        if self.blocks_cached == 0:
            return 0.0
        return self.remat_blocks / self.blocks_cached

    def reset_window(self) -> None:
        self.window_hit_tokens = 0
        self.window_query_tokens = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "hit_rate": self.hit_rate,
            "hit_rate_window": self.window_hit_rate,
            "hit_tokens": self.hit_tokens,
            "query_tokens": self.query_tokens,
            "phantom_hit_rate": self.phantom_hit_rate,
            "phantom_hit_tokens": self.phantom_hit_tokens,
            "phantom_query_tokens": self.phantom_query_tokens,
            "remat_blocks": self.remat_blocks,
            "remat_mb": self.remat_mb,
            "remat_ratio": self.remat_ratio,
            "blocks_cached_total": self.blocks_cached,
            "block_size_bytes": self.block_size_bytes,
        }


class TTFTTracker:
    """Engine-side time to first token, in ms.

    `Request.first_token_ts - Request.arrival_time`, both stamped inside
    engine core. This is deliberately *not* the front end's TTFT: that one
    also carries front-end queueing and detokenization, neither of which an
    eviction policy can move, which dilutes exactly the effect being
    measured. What is left here — scheduler queueing plus prefill — is the
    part a cache miss actually pays for.

    It belongs on the `kv_hbm` line because hit rate alone cannot settle the
    question. A policy can raise hit rate and still lose on latency if the
    blocks it kept were cheap to rebuild and the ones it dropped were not;
    `remat_mb` says how much KV was rebuilt, this says what it cost.

    The mean is cumulative and exact. Percentiles come from a bounded ring of
    recent samples, because retaining every sample for an exact percentile is
    unbounded under load and the tail is what matters anyway.
    """

    def __init__(self, ring_size: int = 4096) -> None:
        self.ring_size = max(int(ring_size), 0)
        self._recent: deque = deque(maxlen=self.ring_size or 1)
        self.count = 0
        self.total_ms = 0.0
        self.window_count = 0
        self.window_total_ms = 0.0

    def record(self, ttft_ms: float) -> None:
        if ttft_ms < 0.0:
            # A non-monotonic wall clock can produce this. Dropping is right:
            # a negative latency in the mean is worse than a missing sample.
            return
        self.count += 1
        self.total_ms += ttft_ms
        self.window_count += 1
        self.window_total_ms += ttft_ms
        if self.ring_size:
            self._recent.append(ttft_ms)

    @property
    def mean_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    @property
    def window_mean_ms(self) -> float:
        if self.window_count == 0:
            return 0.0
        return self.window_total_ms / self.window_count

    @property
    def p95_ms(self) -> float:
        """Over the retained ring, not the whole run."""
        if not self._recent:
            return 0.0
        ordered = sorted(self._recent)
        # Nearest-rank: the smallest sample at or above the 95th percentile.
        idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
        return ordered[max(idx, 0)]

    def reset_window(self) -> None:
        self.window_count = 0
        self.window_total_ms = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ttft_ms": self.mean_ms,
            "ttft_window_ms": self.window_mean_ms,
            "ttft_p95_ms": self.p95_ms,
            "ttft_n": self.count,
        }


class EvictionObserver:
    """Counters, the regret ring buffer, and the optional decision log."""

    def __init__(
        self,
        decision_log_path: str | None = None,
        regret_window_ms: float = 120_000.0,
        regret_buffer_size: int = 4096,
    ) -> None:
        self.counters = EvictionCounters()
        self.regret_window_ms = regret_window_ms
        # Bounded ring buffer of recent evictions. On request admission the
        # key is already in hand, so a hit against this is an O(1) lookup —
        # which turns regret from an offline join into a number the server
        # reports (07 §6).
        self._recent: deque[_RecentEviction] = deque(maxlen=regret_buffer_size)
        self._recent_by_key: dict[NodeKey, list[_RecentEviction]] = {}
        self._log_file = None
        self._log_path = decision_log_path
        if decision_log_path:
            try:
                self._log_file = open(decision_log_path, "a", buffering=1)
            except OSError:
                logger.exception(
                    "Could not open node eviction decision log %s",
                    decision_log_path,
                )

    # -- decisions ---------------------------------------------------------

    def record_eviction(
        self,
        block_id: int,
        keys: list[NodeKey],
        breakdown: ScoreBreakdown | None,
        now_ms: float,
        prefix_pos: int | None = None,
        run_len: int | None = None,
        rank_in_splice: int | None = None,
        snapshot_age_ms: float = 0.0,
        num_free_fresh: int = 0,
        was_spliced: bool = False,
        speculative: bool | None = None,
    ) -> None:
        """One evicted *cached* block. Fresh blocks carry no information.

        `speculative` is the caller's authoritative read of provenance, from
        the ownership index. It falls back to the score breakdown only for
        callers that have no index in hand (the replay harness, tests).
        """
        counters = self.counters
        counters.evictions_total += 1
        if breakdown is None or not breakdown.scored:
            counters.unscored_evictions_total += 1
        if was_spliced:
            counters.evictions_by_score_total += 1
        if speculative is None:
            speculative = breakdown is not None and breakdown.speculative
        if speculative:
            counters.speculative_evicted_before_confirm += 1

        if keys and breakdown is not None and breakdown.scored:
            record = _RecentEviction(
                key=keys[0],
                score=breakdown.score,
                ts_ms=now_ms,
                e_miss_ms=breakdown.e_miss_ms,
            )
            self._push_recent(record)

        if self._log_file is None:
            return
        line = {
            "ts": now_ms,
            "block_id": block_id,
            "keys": [list(k) for k in keys],
            "score": breakdown.score if breakdown is not None else None,
            "terms": breakdown.as_terms() if breakdown is not None else None,
            "speculative": speculative,
            "floor_applied": (
                breakdown.floor_applied if breakdown is not None else None
            ),
            "prefix_pos": prefix_pos,
            "run_len": run_len,
            "rank_in_splice": rank_in_splice,
            # Separates a *wrong* forecast from a merely *late* one. They
            # produce identical evictions and need opposite fixes, and the
            # natural instinct is to blame the formula — the expensive place
            # to look (07 §7).
            "snapshot_age_ms": snapshot_age_ms,
            "num_free_fresh": num_free_fresh,
            "policy_on": self.counters.policy_enabled,
        }
        try:
            self._log_file.write(json.dumps(line) + "\n")
        except (OSError, ValueError):
            logger.debug("Node eviction decision log write failed", exc_info=True)

    def _push_recent(self, record: _RecentEviction) -> None:
        if len(self._recent) == self._recent.maxlen:
            evicted = self._recent[0]
            bucket = self._recent_by_key.get(evicted.key)
            if bucket:
                try:
                    bucket.remove(evicted)
                except ValueError:
                    pass
                if not bucket:
                    del self._recent_by_key[evicted.key]
        self._recent.append(record)
        self._recent_by_key.setdefault(record.key, []).append(record)

    # -- outcomes ----------------------------------------------------------

    def record_admission(self, key: NodeKey, now_ms: float) -> None:
        """A request for `key` arrived. Anything we evicted for that key
        inside the horizon is regret, and every recent eviction feeds the
        score/outcome correlation."""
        bucket = self._recent_by_key.get(key)
        if not bucket:
            return
        counters = self.counters
        survivors: list[_RecentEviction] = []
        for record in bucket:
            if record.counted:
                continue
            if now_ms - record.ts_ms <= self.regret_window_ms:
                record.counted = True
                counters.evicted_then_needed_total += 1
                counters.evicted_then_needed_cost_ms += record.e_miss_ms
                counters.needed_score_sum += record.score
                counters.needed_count += 1
            else:
                survivors.append(record)
        if survivors:
            self._recent_by_key[key] = survivors
        else:
            self._recent_by_key.pop(key, None)

    def expire(self, now_ms: float) -> None:
        """Age out evictions that were never needed again. Called from the
        tick, so the correlation's negative class is populated without any
        offline join."""
        counters = self.counters
        while self._recent and now_ms - self._recent[0].ts_ms > self.regret_window_ms:
            record = self._recent.popleft()
            bucket = self._recent_by_key.get(record.key)
            if bucket is not None:
                try:
                    bucket.remove(record)
                except ValueError:
                    pass
                if not bucket:
                    self._recent_by_key.pop(record.key, None)
            if not record.counted:
                record.counted = True
                counters.not_needed_score_sum += record.score
                counters.not_needed_count += 1

    def close(self) -> None:
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()
        self._log_file = None
