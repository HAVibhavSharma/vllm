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

import os
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Mirrors `node_eviction/controller.py::node_key_for_request` on the policy
# branch: identity rides in `sampling_params.extra_args` and already crosses
# the front-end -> engine-core boundary, so nothing new is plumbed for it.
_JOB_ID_FIELD = "job_id"
_NODE_FIELD = "langgraph_node"

UNTRACKED = "<untracked>"


def is_prefetch_only(request) -> bool:
    """A phantom prefetch, not demand.

    `POST /v1/agents/prefetch` submits one request per prefix carrying
    `kv_transfer_params={"prefetch_only": True}` (`agent_prefetch/
    submitter.py`), and the scheduler discriminates on exactly this
    (`Scheduler._is_prefetch_only_request`). This branch does not *originate*
    phantoms the way the policy branch does, but it serves the endpoint, so
    the traffic exists here too and has to be kept out of the headline rate.
    """
    params = getattr(request, "kv_transfer_params", None)
    return bool(params and params.get("prefetch_only"))


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
        ttft_per_request_log: bool = True,
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
        # measured.
        #
        # No mean, no median, no percentile: every sample is emitted on its
        # own `kv_hbm_ttft` line as it is taken, mirroring
        # `node_eviction/metrics.py::TTFTTracker`. An aggregate computed here
        # would be fixed at write time to a window nobody chose and a
        # population nobody selected, and TTFT is heavy-tailed enough that
        # this routinely decides the answer. With the samples in the log, any
        # aggregate can be computed afterwards over exactly the requests being
        # asked about. Only the count survives, because the `kv_hbm` line's
        # rates are unreadable without it and because a window in which only
        # latency moved still has to get past the change gate.
        self.ttft_count = 0
        self.ttft_window_count = 0
        self.ttft_per_request_log = ttft_per_request_log

        self._reset_totals()

        self._last_log_monotonic = 0.0
        self._last_fingerprint: tuple[int, ...] | None = None

        # Measurement epoch. Bumped by `reset_measurement`, which a benchmark
        # harness calls over `POST /v1/kv_metrics/reset` once its warmup has
        # finished. Every `kv_hbm` line carries the epoch it was measured in,
        # so a warmup line can never be mistaken for a measured one — they are
        # otherwise identical in shape. Mirrors `node_eviction/controller.py`
        # on the policy branch, field for field.
        self.metrics_epoch = 0
        self._epoch_started_monotonic = time.monotonic()
        self._epoch_started_wall = time.time()
        self._epoch_label = "boot"

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
        # Phantom prefetch traffic, kept out of every rate above.
        self.phantom_hit_tokens = 0
        self.phantom_query_tokens = 0

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
        # One INFO line per request instead of one per window, which is real
        # cost under load. On by default because it is now the only place TTFT
        # is reported; `ttft_n` on the `kv_hbm` line does not depend on it.
        per_request = os.environ.get("VLLM_HBM_TTFT_PER_REQUEST", "1") not in (
            "0",
            "false",
            "False",
        )
        return cls(
            block_pool,
            period_ms,
            max(top_keys, 0),
            remat_window_blocks=max(window, 0),
            ttft_per_request_log=per_request,
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
        self,
        num_tokens: int,
        num_hits: int,
        preempted: bool = False,
        phantom: bool = False,
        request=None,
    ) -> None:
        """One prefix-cache lookup, from `KVCacheManager.get_computed_blocks`.

        Counted here rather than read off `PrefixCacheStats` because that
        object is drained by whoever polls the metrics loggers — sharing it
        would make this hit rate depend on whether anything else was
        scraping, and on `log_stats` being on at all.

        A phantom (`/v1/agents/prefetch`) touches only the phantom buckets.
        Left in the headline rate it lands in both numerator and denominator,
        and not with a consistent sign: a phantom's first prefill is a
        near-total miss that deflates the rate, while a phantom for an
        already-resident prefix is a large hit no user ever experienced.
        Since the line exists to be diffed field-by-field against arms that
        may originate a different number of phantoms, either direction shows
        up as a policy difference that never happened.
        """
        if request is not None:
            # Stashed for this request's own `kv_hbm_ttft` line, phantom or
            # not: a phantom's latency is real work, and the line flags it
            # rather than dropping it. Taken here, while both figures are
            # still the prompt's — `num_tokens` grows with the decode, so
            # reading it at teardown would report a different denominator
            # than the one the hit was measured against.
            request.hbm_query_tokens = num_tokens
            request.hbm_local_hit_tokens = num_hits
        if phantom:
            self.phantom_query_tokens += num_tokens
            self.phantom_hit_tokens += num_hits
            return
        self.query_tokens += num_tokens
        self.hit_tokens += num_hits
        self.window_query_tokens += num_tokens
        self.window_hit_tokens += num_hits
        if not preempted:
            self.query_tokens_fresh += num_tokens
            self.hit_tokens_fresh += num_hits

    def on_request_finished(self, request) -> None:
        """`KVCacheManager.free` — the request is done with its blocks.

        Emits this request's own TTFT, on its own line. Not an average: the
        `kv_hbm` line carries only how many samples were taken, and the
        samples themselves live here, one per request, so any aggregate can be
        computed afterwards over whichever subset of requests is actually
        being asked about.

        Called on preemption too, which is why the sample is gated on
        `ttft_recorded`: a preempted request keeps its original
        `first_token_ts`, so emitting it twice would weight slow requests by
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
            # a negative latency in the log is worse than a missing sample,
            # and it must not be counted either — `ttft_n` would then stand
            # over a sample nothing can be found for.
            return
        self.ttft_count += 1
        self.ttft_window_count += 1
        if self.ttft_per_request_log:
            logger.info("%s", self._ttft_line(request, ttft_ms))

    def _ttft_line(self, request, ttft_ms: float) -> str:
        """One request's TTFT, in the same key=value shape as `kv_hbm`.

        The token breakdown travels with the latency because the latency alone
        cannot be read: 400 ms is a fast cold prefill or a slow warm one
        depending on how much of the prompt had to be computed, and separating
        those is the whole question. `preempted` is here for the same reason —
        a preempted request keeps its first `first_token_ts`, so its sample is
        real, but it was paid for by scheduling pressure rather than a miss.

        Identical in shape to the policy branch's line, field for field, for
        the same reason the `kv_hbm` line is: the arms are meant to be diffed.
        `external_hit_tokens` is always 0 here — this branch has no hook that
        sees the connector's answer — and is emitted rather than omitted so a
        field-by-field diff still lines up.
        """
        query = getattr(request, "hbm_query_tokens", None)
        local = getattr(request, "hbm_local_hit_tokens", None) or 0
        if query is None:
            # No lookup was ever recorded for this request — it never reached
            # `get_computed_blocks`. Reporting -1 rather than 0 keeps it from
            # reading as a genuine all-cold prefill.
            query = local = -1
            cold = -1
        else:
            cold = max(query - local, 0)
        return (
            "kv_hbm_ttft variant=baseline "
            f"epoch={self.metrics_epoch} "
            f"req={getattr(request, 'request_id', '-')} "
            f"ttft_ms={ttft_ms:.1f} "
            f"query_tokens={query} "
            f"hit_tokens={local} "
            f"external_hit_tokens=0 "
            f"cold_tokens={cold} "
            f"preempted={getattr(request, 'num_preemptions', 0)} "
            # Phantom prefetches are out of every rate on the `kv_hbm` line,
            # so their latency is flagged rather than dropped: it is real work
            # warming bought, and hiding it would make warming look free.
            f"phantom={int(is_prefetch_only(request))}"
        )

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
    def phantom_hit_rate(self) -> float:
        """How often warming fetched something already resident.

        High is not good: it means the prefetch bought nothing. Low with a
        large `phantom_query_tokens` is the expensive case — real prefills
        submitted ahead of user traffic. Either way it belongs next to the
        real rate, not inside it.
        """
        if self.phantom_query_tokens == 0:
            return 0.0
        return self.phantom_hit_tokens / self.phantom_query_tokens

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
            # How many TTFT samples stand behind this epoch — not what they
            # were. The latencies are on their own `kv_hbm_ttft` lines, one
            # per request, because an average computed here is fixed at write
            # time to a window nobody chose and TTFT is heavy-tailed enough
            # that such an average routinely describes no request in it.
            f"ttft_n={self.ttft_count} "
            f"ttft_win_n={self.ttft_window_count} "
            f"hit_tokens={self.hit_tokens} "
            f"query_tokens={self.query_tokens} "
            f"query_tokens_fresh={self.query_tokens_fresh} "
            # Phantom prefetch traffic (`/v1/agents/prefetch`), excluded from
            # every rate above and reported here so the prefill that warming
            # bought is visible next to the hit rate it was meant to raise,
            # rather than hidden inside it.
            f"phantom_hit_rate={self.phantom_hit_rate:.4f} "
            f"phantom_query_tokens={self.phantom_query_tokens} "
            f"remat_blocks={self.remat_blocks} "
            f"remat_mb={self.remat_mb:.1f} "
            f"remat_ratio={self.remat_ratio:.4f} "
            f"blocks_cached={self.blocks_cached_total} "
            f"index_keys={len(set(self._owner.values()))} "
            f"index_blocks={len(self._owner)} "
            # Which measurement epoch these numbers belong to, and how long it
            # has been running. `epoch=0` is everything before the harness
            # said its warmup was done; the run to report on is the highest
            # epoch present. Ahead of `top_evicted` because that field is
            # free-form (it contains `=` and `,`) and has to stay last.
            f"epoch={self.metrics_epoch} "
            f"epoch_age_s={self.epoch_age_s:.1f} "
            f"top_evicted={top_str}"
        )

    @property
    def epoch_age_s(self) -> float:
        """Seconds since the current measurement epoch began.

        The denominator for anything rate-like read off the line, and the
        reason an end-to-end figure need not include the warmup: the epoch
        clock restarts when the harness posts its reset.
        """
        return max(time.monotonic() - self._epoch_started_monotonic, 0.0)

    def reset_measurement(
        self,
        label: str = "",
        before_reset: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        """Discard everything measured so far and start a new epoch.

        The workload calls this over `POST /v1/kv_metrics/reset` once its
        warmup has populated the caches. A warmup pass corrupts every headline
        number three separate ways: the `kv_hbm_ttft` samples carry cold
        prefills no measured request paid, and nothing but the epoch stamp
        separates them; the hit rate is cumulative, so a
        cold pass drags it down for the rest of the run; and the elapsed
        window behind any rate starts at server boot rather than at the first
        measured request.

        `before_reset` is the hook the caller uses to flush HBM at the same
        boundary (`EngineCore.reset_kv_metrics(flush_hbm=True)`). It runs
        *after* the cold-phase numbers have been captured and *before* the
        counters are zeroed, which is the only ordering that works: flushing
        first fires `on_reset_prefix_cache`, which zeroes the token counters
        itself, and the marker line would then report a cold phase that looks
        like it never ran.

        With no hook the KV cache is left alone — a warmup exists to fill it —
        and the rematerialisation ring survives, because a block evicted
        during the warmup and rebuilt afterwards is real rebuild work paid
        inside the measured window. With one, the resident blocks go (and
        `on_reset_prefix_cache` drops the ring, correctly: an operator wipe is
        not policy eviction) but a KV connector's store does not, so the warm
        phase starts with an empty HBM cache in front of a populated CPU tier.

        Byte-compatible with the policy branch's
        `NodeEvictionController.reset_measurement`, for the same reason the
        `kv_hbm` line is: the two runs are meant to be diffed.
        """
        # Ungated: `maybe_log` would suppress this if the period had not
        # elapsed or nothing had changed, and the one line guaranteed to
        # matter is the last one before the boundary.
        discarded_line = self.summary()
        logger.info("%s", discarded_line)

        discarded = self.stats()
        previous_epoch = self.metrics_epoch
        epoch_age_s = self.epoch_age_s

        # Whatever the hook does to the cache — including the counter writes
        # its own hooks make — is undone by the zeroing below, so the new
        # epoch starts at zero either way.
        hbm_flushed = "skipped"
        if before_reset is not None:
            hbm_flushed = "true" if before_reset() else "false"

        self._reset_totals()
        self.evictions_total = 0
        self._evictions_by_key.clear()
        # The samples themselves are already in the log, stamped with the
        # epoch they were taken in; this is what stops the next epoch's first
        # line from claiming a count that belongs to the warmup.
        self.ttft_count = 0
        self.ttft_window_count = 0

        self.metrics_epoch += 1
        self._epoch_started_monotonic = time.monotonic()
        self._epoch_started_wall = time.time()
        self._epoch_label = label or "reset"

        # So the first line of the new epoch is not withheld by the rate limit
        # or the change gate — with the counters at zero the fingerprint would
        # otherwise have to move before anything printed.
        self._last_log_monotonic = 0.0
        self._last_fingerprint = None

        marker = (
            f"kv_hbm_reset variant=baseline "
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
            # request, under the epoch this marker is retiring, which is what
            # makes them separable afterwards.
            f"discarded_ttft_n={discarded['ttft_n']} "
            f"discarded_hit_rate={discarded['hit_rate']:.4f} "
            f"discarded_query_tokens={discarded['query_tokens']} "
            f"discarded_evicted={discarded['evictions_total']}"
        )
        logger.info("%s", marker)

        return {
            "ok": True,
            "variant": "baseline",
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
            "ttft_n": self.ttft_count,
            "ttft_window_n": self.ttft_window_count,
            "hit_tokens": self.hit_tokens,
            "query_tokens": self.query_tokens,
            "hit_tokens_fresh": self.hit_tokens_fresh,
            "query_tokens_fresh": self.query_tokens_fresh,
            "phantom_hit_rate": self.phantom_hit_rate,
            "phantom_hit_tokens": self.phantom_hit_tokens,
            "phantom_query_tokens": self.phantom_query_tokens,
            "remat_blocks": self.remat_blocks,
            "remat_mb": self.remat_mb,
            "remat_ratio": self.remat_ratio,
            "blocks_cached_total": self.blocks_cached_total,
            "block_size_bytes": self.block_size_bytes,
            "index_keys": len(set(self._owner.values())),
            "index_blocks": len(self._owner),
            # Which measurement window these numbers describe. Without it a
            # scraped sample cannot be told apart from one taken during warmup.
            "metrics_epoch": self.metrics_epoch,
            "epoch_age_s": self.epoch_age_s,
        }
