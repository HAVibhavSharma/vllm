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
