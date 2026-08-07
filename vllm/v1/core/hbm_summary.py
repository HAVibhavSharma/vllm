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
    ) -> None:
        self.block_pool = block_pool
        self.period_ms = period_ms
        self.top_keys = top_keys

        # block_id -> owning node label. Bounded by the pool: an entry is
        # added when a block is given a hash and removed when that hash is
        # stripped, which are the only two transitions that exist.
        self._owner: dict[int, str] = {}
        # Per window, cleared when the line is emitted, so a finished job
        # stops being named instead of pinning its job_id forever.
        self._evictions_by_key: dict[str, int] = {}

        self.evictions_total = 0
        self.blocks_cached_total = 0

        self._last_log_monotonic = 0.0
        self._last_fingerprint: tuple[int, ...] | None = None

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
        return cls(block_pool, period_ms, max(top_keys, 0))

    # -- hooks from BlockPool ---------------------------------------------

    def on_blocks_cached(self, request, blocks) -> None:
        label = node_label_for_request(request) or UNTRACKED
        for block in blocks:
            if block.is_null:
                continue
            self._owner[block.block_id] = label
            self.blocks_cached_total += 1

    def on_block_evicted(self, block_id: int) -> None:
        label = self._owner.pop(block_id, UNTRACKED)
        self._evictions_by_key[label] = self._evictions_by_key.get(label, 0) + 1
        self.evictions_total += 1

    def on_reset_prefix_cache(self) -> None:
        # Every hash in the pool was just invalidated, so every claim here is
        # stale. Not counted as evictions: that would attribute an operator
        # action to whichever job happened to be resident.
        self._owner.clear()

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
            f"evicted={self.evictions_total} "
            f"evicted_by_score=0 "
            f"regret=0.000 "
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
        fingerprint = (self.evictions_total, self.blocks_cached_total)
        if fingerprint == self._last_fingerprint:
            return None
        self._last_log_monotonic = now
        self._last_fingerprint = fingerprint
        line = self.summary()
        logger.info("%s", line)
        self._evictions_by_key.clear()
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
            "evictions_total": self.evictions_total,
            "index_keys": len(set(self._owner.values())),
            "index_blocks": len(self._owner),
        }
