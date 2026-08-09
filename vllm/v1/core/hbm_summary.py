# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HBM block accounting for the LRU baseline.

This exists to be diffed against the node-aware eviction branch. That branch
logs one `kv_hbm ...` line per period from its controller; without the same
line here, comparing the two runs means reading two different sets of numbers
produced by two different code paths and hoping they mean the same thing.

So the line emitted here is byte-compatible in shape with the policy branch's:
same key order, same units, `variant=baseline`, and `splices=0` /
`spliced_blocks=0` because LRU never reorders the free queue. A field that
only the policy can move is reported as zero rather than omitted, so a naive
field-by-field diff still lines up.

The tracking is deliberately thin. It answers four questions:

- how many blocks HBM has, and how many are in use (`total`/`used`/`free`)
- how many sit in the free queue as eviction candidates (`queue`)
- how many cached blocks were destroyed (`evicted`)
- **which `job_id:node` owned them** (`top_evicted`)

The last one is the reason this is not just `get_usage()`. "We evicted 12k
blocks" is not actionable; "9k of them belonged to job7:research" says the
pressure has a source.
"""

import math
import os
import time
from collections import deque

from vllm.logger import init_logger

logger = init_logger(__name__)

# Mirrors `node_eviction/controller.py::node_key_for_request` on the policy
# branch: identity rides in `sampling_params.extra_args` and already crosses
# the front-end -> engine-core boundary, so nothing new is plumbed for it.
_JOB_ID_FIELD = "job_id"
_NODE_FIELD = "langgraph_node"

UNTRACKED = "<untracked>"


def node_label_for_request(request) -> str | None:
    """`"{job_id}:{langgraph_node}"`, or None when the request has no identity.

    A request without identity is not an error: benchmark traffic and warmup
    have none, and their blocks are simply counted under `<untracked>`.
    """
    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is None:
        return None
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return None
    job_id = extra_args.get(_JOB_ID_FIELD)
    node = extra_args.get(_NODE_FIELD)
    if job_id is None or node is None:
        return None
    return f"{job_id}:{node}"


class HBMSummaryLogger:
    """Per-period HBM accounting, attributed to the node that owned the block.

    Costs one dict write per newly cached block and one dict pop per eviction
    — both on paths that are already iterating those blocks — plus a clock
    read per scheduler step.
    """

    def __init__(
        self,
        block_pool,
        period_ms: float,
        top_keys: int = 5,
        remat_window_blocks: int = 0,
        block_size_bytes: int = 0,
    ) -> None:
        self.block_pool = block_pool
        self.period_ms = period_ms
        self.top_keys = top_keys
        self.block_size_bytes = block_size_bytes
        # `0` sizes the ring to the pool: one full turnover is the horizon
        # over which an eviction is still this policy's doing.
        self.remat_window_blocks = (
            remat_window_blocks
            if remat_window_blocks > 0
            else getattr(block_pool, "num_gpu_blocks", 0)
        )

        # block_id -> owning node label. Bounded by the pool: an entry is
        # added when a block is given a hash and removed when that hash is
        # stripped, which are the only two transitions that exist.
        self._owner: dict[int, str] = {}
        # Per window, cleared when the line is emitted, so a finished job
        # stops being named instead of pinning its job_id forever.
        self._evictions_by_key: dict[str, int] = {}

        # Hashes that were cached and then evicted. `set` for the O(1)
        # membership test on the caching path, `deque` for the eviction order
        # the ring needs; they are kept in step.
        self._evicted_hashes: deque = deque()
        self._evicted_hash_set: set = set()

        # Not reset by `reset_prefix_cache`, matching the policy branch, where
        # the eviction counters live on `EvictionObserver` and survive it.
        self.evictions_total = 0

        # Engine-side TTFT, in ms: `first_token_ts - arrival_time`, both
        # stamped inside engine core. Deliberately not the front end's TTFT —
        # that also carries front-end queueing and detokenization, neither of
        # which an eviction policy can move, which dilutes the effect being
        # measured. Mirrors `node_eviction/metrics.py::TTFTTracker`; the mean
        # is exact and cumulative, the percentile comes from a bounded ring
        # because retaining every sample is unbounded under load.
        self._ttft_recent: deque = deque(maxlen=4096)
        self.ttft_count = 0
        self.ttft_total_ms = 0.0
        self.ttft_window_count = 0
        self.ttft_window_total_ms = 0.0

        self._reset_totals()

        self._last_log_monotonic = 0.0
        self._last_fingerprint: tuple[int, ...] | None = None

    def _reset_totals(self) -> None:
        """Zero everything the prefix cache's contents can invalidate.

        Shared by `__init__` and `on_reset_prefix_cache` so a counter added to
        one cannot go missing from the other. Mirrors
        `node_eviction/metrics.py::CacheMovementTracker._reset_totals` on the
        policy branch — if the two diverge here, the first run after a reset
        compares one branch's fresh numbers against the other's carried-over
        ones, which looks exactly like a policy effect.
        """
        self.blocks_cached_total = 0
        self.remat_blocks = 0
        self.hit_tokens = 0
        self.query_tokens = 0
        # First-scheduling queries only. This is the subset vLLM's own
        # "Prefix cache hit rate" counts, so it is what reconciles this line
        # against the engine's log; without it the two disagree by a factor
        # that grows with the preemption rate and nobody can tell which is
        # broken.
        self.hit_tokens_fresh = 0
        self.query_tokens_fresh = 0
        # Reset when a line is emitted: a cumulative hit rate over a long run
        # is dominated by whatever the workload did first.
        self.window_hit_tokens = 0
        self.window_query_tokens = 0

    def configure(self, block_size_bytes: int) -> None:
        """Told the KV page size after construction.

        `BlockPool` builds this and does not know the byte size of a block —
        only `KVCacheManager` holds the `KVCacheConfig`. Left at 0 the MB
        figure reports 0 rather than a guess: a fabricated byte count is
        worse than an absent one when the point is comparing two runs.
        """
        self.block_size_bytes = block_size_bytes

    @classmethod
    def maybe_build(cls, block_pool) -> "HBMSummaryLogger | None":
        """Off when `VLLM_HBM_SUMMARY_PERIOD_MS=0`; 30s otherwise.

        Read straight from the environment rather than added to `envs.py`:
        this is comparison instrumentation on a baseline clone, and keeping
        it out of the config surface keeps the diff against upstream small.
        """
        raw = os.environ.get("VLLM_HBM_SUMMARY_PERIOD_MS", "30000")
        try:
            period_ms = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid VLLM_HBM_SUMMARY_PERIOD_MS=%r; using 30000", raw
            )
            period_ms = 30_000.0
        if period_ms <= 0:
            return None
        try:
            top_keys = int(os.environ.get("VLLM_HBM_SUMMARY_TOP_KEYS", "5"))
        except ValueError:
            top_keys = 5
        try:
            window = int(os.environ.get("VLLM_HBM_REMAT_WINDOW_BLOCKS", "0"))
        except ValueError:
            window = 0
        return cls(
            block_pool,
            period_ms,
            max(top_keys, 0),
            remat_window_blocks=max(window, 0),
        )

    # -- hooks from BlockPool ---------------------------------------------

    def on_blocks_cached(self, request, blocks) -> None:
        label = node_label_for_request(request) or UNTRACKED
        for block in blocks:
            if block.is_null:
                continue
            self._owner[block.block_id] = label
            self.blocks_cached_total += 1
            self._note_cached(block.block_hash)

    def on_block_evicted(self, block_id: int) -> None:
        label = self._owner.pop(block_id, UNTRACKED)
        self._evictions_by_key[label] = self._evictions_by_key.get(label, 0) + 1
        self.evictions_total += 1
        # Read off the block rather than passed in: `BlockPool` calls this
        # *before* `reset_hash()`, so the hash is still there.
        self._note_evicted(self._hash_of(block_id))

    def on_cache_query(
        self, num_tokens: int, num_hits: int, preempted: bool = False
    ) -> None:
        """One prefix-cache lookup, from `KVCacheManager.get_computed_blocks`.

        Counted here rather than read off `PrefixCacheStats` because that
        object is drained by whoever polls the metrics loggers — sharing it
        would make this hit rate depend on whether anything else was
        scraping, and on `log_stats` being on at all.
        """
        self.query_tokens += num_tokens
        self.hit_tokens += num_hits
        self.window_query_tokens += num_tokens
        self.window_hit_tokens += num_hits
        if not preempted:
            self.query_tokens_fresh += num_tokens
            self.hit_tokens_fresh += num_hits

    def on_request_finished(self, request) -> None:
        """`KVCacheManager.free` — the request is done with its blocks.

        Called on preemption too, which is why the sample is gated on
        `ttft_recorded`: a preempted request keeps its original
        `first_token_ts`, so counting it twice would weight slow requests by
        how often they were preempted.
        """
        if request.ttft_recorded:
            return
        first = getattr(request, "first_token_ts", None)
        if first is None:
            # Finished before producing a token — aborted. No prefill latency
            # to attribute.
            return
        request.ttft_recorded = True
        ttft_ms = (first - request.arrival_time) * 1000.0
        if ttft_ms < 0.0:
            # A non-monotonic wall clock can produce this. Dropping is right:
            # a negative latency in the mean is worse than a missing sample.
            return
        self.ttft_count += 1
        self.ttft_total_ms += ttft_ms
        self.ttft_window_count += 1
        self.ttft_window_total_ms += ttft_ms
        self._ttft_recent.append(ttft_ms)

    @property
    def ttft_mean_ms(self) -> float:
        if self.ttft_count == 0:
            return 0.0
        return self.ttft_total_ms / self.ttft_count

    @property
    def ttft_window_mean_ms(self) -> float:
        if self.ttft_window_count == 0:
            return 0.0
        return self.ttft_window_total_ms / self.ttft_window_count

    @property
    def ttft_p95_ms(self) -> float:
        """Over the retained ring, not the whole run."""
        if not self._ttft_recent:
            return 0.0
        ordered = sorted(self._ttft_recent)
        # Nearest-rank: the smallest sample at or above the 95th percentile.
        idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
        return ordered[max(idx, 0)]

    def on_reset_prefix_cache(self) -> None:
        # Every hash in the pool was just invalidated, so every claim here is
        # stale. Not counted as evictions: that would attribute an operator
        # action to whichever job happened to be resident.
        self._owner.clear()
        # Nothing in the ring can be rematerialised either — a later cache of
        # the same prefix is new work, not redone work.
        self._evicted_hashes.clear()
        self._evicted_hash_set.clear()
        # And the rates go with it: those tokens described a cache that no
        # longer exists. `reset_prefix_cache` is only ever invoked explicitly,
        # which in practice means "start a clean measurement". Upstream agrees
        # — `CachingMetrics.observe` resets its own aggregation on
        # `stats.reset` (`v1/metrics/stats.py`).
        self._reset_totals()

    # -- rematerialisation --------------------------------------------------
    #
    # A block is counted as movement when its hash is cached *again* after
    # having been evicted: the block existed, we threw it away, we rebuilt
    # it. That is work the eviction forced, and a better policy avoids it.
    #
    # Deliberately not the miss count. Misses include every prefix the server
    # has never seen, which no eviction policy can do anything about;
    # counting them would credit a policy for a cold start. Rematerialisation
    # is zero for a first-time prefix by construction.

    def _hash_of(self, block_id: int):
        blocks = getattr(self.block_pool, "blocks", None)
        if blocks is None or block_id >= len(blocks):
            return None
        return blocks[block_id].block_hash

    def _note_cached(self, block_hash) -> None:
        if block_hash is None:
            return
        if block_hash in self._evicted_hash_set:
            self.remat_blocks += 1
            # Discard rather than leave it: the hash is resident again, so a
            # *second* remat requires a second eviction. Leaving it would
            # count every subsequent cache of the same prefix as movement.
            self._evicted_hash_set.discard(block_hash)

    def _note_evicted(self, block_hash) -> None:
        if block_hash is None or self.remat_window_blocks == 0:
            return
        # The set is the membership test; the deque only supplies eviction
        # order. A hash that was counted and discarded from the set can be
        # appended again, leaving a stale duplicate in the deque — it pops
        # early and shortens the window slightly for that one hash. The error
        # is bounded and conservative (it can only *under*-count movement),
        # which is the right direction for a number used to claim an
        # improvement.
        if block_hash not in self._evicted_hash_set:
            self._evicted_hashes.append(block_hash)
            self._evicted_hash_set.add(block_hash)
        while len(self._evicted_hashes) > self.remat_window_blocks:
            self._evicted_hash_set.discard(self._evicted_hashes.popleft())

    @property
    def hit_rate(self) -> float:
        if self.query_tokens == 0:
            return 0.0
        return self.hit_tokens / self.query_tokens

    @property
    def hit_rate_fresh(self) -> float:
        """Demand-side hit rate: first scheduling of each request only.

        A preempted request re-queries with a `num_tokens` grown to include
        what it already generated, so counting those inflates the denominator
        with work no cache was going to serve. `PrefixCacheStats` routes them
        to `preempted_*` and `CachingMetrics.observe` ignores those; this
        mirrors it.

        **Deliberately no longer comparable to vLLM's `Prefix cache hit rate`
        line.** That one counts every call to `get_computed_blocks`, and the
        scheduler makes one per step for as long as a request sits in the
        waiting queue, so its denominator is weighted by queueing delay. The
        caller now counts each request once (`Request.cache_query_counted`);
        upstream's counter is left alone so it stays comparable to stock
        vLLM. The policy branch applies the identical guard.
        """
        if self.query_tokens_fresh == 0:
            return 0.0
        return self.hit_tokens_fresh / self.query_tokens_fresh

    @property
    def window_hit_rate(self) -> float:
        if self.window_query_tokens == 0:
            return 0.0
        return self.window_hit_tokens / self.window_query_tokens

    @property
    def remat_mb(self) -> float:
        """The movement number, in MB of KV actually rebuilt."""
        return self.remat_blocks * self.block_size_bytes / 1e6

    @property
    def remat_ratio(self) -> float:
        """Fraction of all caching work that was redoing work."""
        if self.blocks_cached_total == 0:
            return 0.0
        return self.remat_blocks / self.blocks_cached_total

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        pool = self.block_pool
        total = pool.num_gpu_blocks
        free = pool.get_num_free_blocks()
        used = total - free
        top = sorted(
            self._evictions_by_key.items(), key=lambda kv: (-kv[1], kv[0])
        )[: self.top_keys]
        top_str = ",".join(f"{label}={n}" for label, n in top) or "-"
        return (
            "kv_hbm variant=baseline "
            f"total={total} used={used} free={free} "
            f"usage={(used / total * 100.0) if total else 0.0:.1f}% "
            f"queue={pool.free_block_queue.num_free_blocks} "
            f"splices=0 "
            f"spliced_blocks=0 "
            # LRU stages nothing and can therefore never fall short of what it
            # wanted to stage. Emitted as zeros rather than omitted so the
            # field-by-field diff still lines up.
            f"staged=0 "
            f"deficit=0 "
            f"evicted={self.evictions_total} "
            f"evicted_by_score=0 "
            f"regret=0.000 "
            f"hit_rate={self.hit_rate:.4f} "
            f"hit_rate_fresh={self.hit_rate_fresh:.4f} "
            f"hit_rate_win={self.window_hit_rate:.4f} "
            # Engine-side TTFT — what a miss cost, next to how often it
            # happened. Hit rate alone cannot separate a policy that kept
            # cheap-to-rebuild blocks from one that kept expensive ones.
            f"ttft_ms={self.ttft_mean_ms:.1f} "
            f"ttft_win_ms={self.ttft_window_mean_ms:.1f} "
            f"ttft_p95_ms={self.ttft_p95_ms:.1f} "
            f"ttft_n={self.ttft_count} "
            f"hit_tokens={self.hit_tokens} "
            f"query_tokens={self.query_tokens} "
            f"query_tokens_fresh={self.query_tokens_fresh} "
            # This branch originates no prefetch, so there is no phantom
            # traffic to separate out. Emitted as zeros for the same reason as
            # `staged`/`deficit`: on the policy branch these carry the prefill
            # that origination bought, and a missing field there reads as a
            # parse failure rather than an absence.
            f"phantom_hit_rate=0.0000 "
            f"phantom_query_tokens=0 "
            f"remat_blocks={self.remat_blocks} "
            f"remat_mb={self.remat_mb:.1f} "
            f"remat_ratio={self.remat_ratio:.4f} "
            f"blocks_cached={self.blocks_cached_total} "
            f"index_keys={len(set(self._owner.values()))} "
            f"index_blocks={len(self._owner)} "
            f"top_evicted={top_str}"
        )

    def maybe_log(self, now: float | None = None) -> str | None:
        """Rate-limited *and* change-gated, so an idle server logs nothing.

        Without the change gate this prints an identical line every period
        forever, which trains everyone to filter it out — and the one time it
        matters is the time it changed.

        Returns the line it logged, or None. The return value exists so the
        gating is testable without capturing log output: `vllm`'s root logger
        sets `propagate=False`, so pytest's `caplog` sees nothing and a test
        written against it would pass vacuously.
        """
        now = time.monotonic() if now is None else now
        if now - self._last_log_monotonic < self.period_ms / 1000.0:
            return None

        # Occupancy is deliberately not in the fingerprint: it moves by a
        # block on every step, which would defeat the gate entirely.
        fingerprint = (
            self.evictions_total,
            self.blocks_cached_total,
            self.query_tokens,
            self.remat_blocks,
            # A window where only latency moved is still worth a line.
            self.ttft_count,
        )
        if fingerprint == self._last_fingerprint:
            return None
        self._last_log_monotonic = now
        self._last_fingerprint = fingerprint
        line = self.summary()
        logger.info("%s", line)
        # Per window, so the next line names who is evicting *now* and shows
        # the hit rate *now*.
        self._evictions_by_key.clear()
        self.window_hit_tokens = 0
        self.window_query_tokens = 0
        self.ttft_window_count = 0
        self.ttft_window_total_ms = 0.0
        return line

    def stats(self) -> dict[str, float | int]:
        pool = self.block_pool
        total = pool.num_gpu_blocks
        free = pool.get_num_free_blocks()
        return {
            "hbm_total_blocks": total,
            "hbm_free_blocks": free,
            "hbm_used_blocks": total - free,
            "hbm_usage": (total - free) / total if total else 0.0,
            "free_queue_len": pool.free_block_queue.num_free_blocks,
            "splices_total": 0,
            "blocks_spliced_total": 0,
            "splice_staged_blocks": 0,
            "splice_deficit_blocks": 0,
            "evictions_total": self.evictions_total,
            "hit_rate": self.hit_rate,
            "hit_rate_window": self.window_hit_rate,
            "hit_rate_fresh": self.hit_rate_fresh,
            "ttft_ms": self.ttft_mean_ms,
            "ttft_window_ms": self.ttft_window_mean_ms,
            "ttft_p95_ms": self.ttft_p95_ms,
            "ttft_n": self.ttft_count,
            "hit_tokens": self.hit_tokens,
            "query_tokens": self.query_tokens,
            "hit_tokens_fresh": self.hit_tokens_fresh,
            "query_tokens_fresh": self.query_tokens_fresh,
            "phantom_hit_rate": 0.0,
            "phantom_hit_tokens": 0,
            "phantom_query_tokens": 0,
            "remat_blocks": self.remat_blocks,
            "remat_mb": self.remat_mb,
            "remat_ratio": self.remat_ratio,
            "blocks_cached_total": self.blocks_cached_total,
            "block_size_bytes": self.block_size_bytes,
            "index_keys": len(set(self._owner.values())),
            "index_blocks": len(self._owner),
        }
