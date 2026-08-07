# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the node-aware KV eviction policy.

Every constant here is `[open]` in the design docs — none of them has been
validated against a real trace. They live in one dataclass, loadable from a
JSON file, precisely so the offline replay harness can sweep them without
touching code (08 §6, §7).
"""

import json
import os
from dataclasses import asdict, dataclass, fields

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class NodeEvictionConfig:
    """Tunables for the policy. Defaults are the design docs' starting
    guesses, not measured values."""

    # --- Enablement -------------------------------------------------------
    enabled: bool = False
    """Master switch. When False literally nothing in this package runs and
    the free queue behaves exactly as upstream."""

    # --- Identity (03 §1) -------------------------------------------------
    use_call_type: bool = True
    """Key the index at `(job_id, node, call_type)`. Set False if the
    workflow's flattened `call_type` is not deterministic for what is
    physically the same prefix: an unstable label re-keys the same blocks
    every turn, every key holds a fragment, and no key accumulates enough
    history to be scored — which is worse than no policy at all. With this
    off the index keys at `(job_id, node)` and multi-owner `max` subsumes
    max-over-call-types."""

    # --- Scoring (08 §2) --------------------------------------------------
    tau_ms: float = 30_000.0
    """Time constant of `decay(t) = tau / (tau + t)`. Sets how far ahead the
    policy plans. 30s is a guess (08 §6)."""

    delta_l1_ms: float = 180.0
    """Cost of a miss that LMCache L1 can still serve: GPU hit ~200ms vs L1
    reconstruct ~380ms. From one old benchmark, on one model (00 Part 2)."""

    delta_cold_ms: float = 11_400.0
    """Cost of a miss that must pay full prefill. 60x delta_l1_ms, which is
    why p_cold = 2% already doubles expected eviction cost (01 §5)."""

    staleness_cutoff_ms: float = 120_000.0
    """`PROB` rows older than this are treated as unscored, so a missed
    pub/sub message degrades to LRU rather than acting on stale data."""

    # --- Speculative floor (02 §5) ----------------------------------------
    speculative_floor_high: float = 1e9
    """Starting height of the decaying floor on prefetched (speculative)
    entries. Must sit above the top of the normal score range so a freshly
    prefetched prefix is protected against everything."""

    speculative_floor_low: float = -1.0
    """Where the floor lands after TTL. Below every real score, so a
    falsified prediction becomes the *preferred* victim rather than merely
    losing its protection."""

    speculative_default_ttl_ms: float = 30_000.0
    """TTL used when the forecast carries no `time_to_next_call` for the
    predicted node."""

    speculative_hard_drop_ttl_multiple: float = 4.0
    """Speculative entries are dropped from the index entirely at
    TTL x this, which bounds index growth on the speculative side."""

    # --- Prefetch origination (02 §4, step 6) -----------------------------
    prefetch_wants_enabled: bool = False
    """Produce a want-list at all. Separate from `enabled` on purpose:
    reordering the free queue is free, while originating a prefetch costs a
    real prefill whenever LMCache misses the prefix. Driven by
    `VLLM_NODE_EVICTION_PREFETCH_DRAIN`, the same switch the front-end
    drainer reads, so the two halves cannot be turned on independently."""

    prefetch_min_prob: float = 0.5
    """Only forecast rows at least this likely become wants. The diagram's
    "admit/prefetch if the need for the cache is imminent" has two axes; this
    is the *will it happen* one."""

    prefetch_horizon_ms: float = 60_000.0
    """...and this is the *when* axis. A node predicted 10 minutes out is
    real but not imminent, and warming it now just evicts something that is
    needed sooner."""

    prefetch_max_outstanding: int = 8
    """Ceiling on wants pending + in flight. This is the actual bound on how
    much prefill work speculation can create, so it is deliberately small."""

    prefetch_max_per_drain: int = 4
    """Ceiling on how many wants one front-end poll may take. Each want
    fans out to *every* prefix the registry holds for its agent, so this is
    not the phantom count."""

    prefetch_want_ttl_ms: float = 30_000.0
    """A want nobody drained within this window is dropped as stale intent
    rather than acted on late."""

    prefetch_resubmit_backoff_ms: float = 60_000.0
    """How long a drained want blocks a re-offer for the same key. Bounds
    the retry rate when a phantom is dropped or its prefix never lands."""

    prefetch_summary_period_ms: float = 30_000.0
    """How often the tick may log one INFO line of prefetch state. Rate
    limited *and* change-gated, so an idle server stays silent. Set to 0 to
    turn the line off; the counters remain readable via
    `KVCacheManager.get_node_eviction_stats()` either way.

    This exists because every per-event prefetch log is either too rare to
    prove anything (startup) or too frequent for INFO (one per phantom), and
    the numbers that actually separate "the forecast is wrong" from "the
    phantom never ran" are ratios over time, not events."""

    prefetch_agent_namespace: str = "langgraph"
    """`agent_id` is `{namespace}:{node_name}` — the join with the LangGraph
    fork's `derive_agent_id()`. Resolved engine-side so the front-end drainer
    needs no knowledge of the naming convention."""

    # --- Tick and splice (01 §6 Rule 4) -----------------------------------
    tick_period_ms: float = 250.0
    """Wall-clock cadence of the re-score. Step count is not used because
    step duration varies too much."""

    splice_max_blocks: int = 256
    """K: how many blocks may be moved to the head per tick. The policy
    converges over several ticks; approximate order is fine for a
    heuristic."""

    fresh_skip_threshold: int = 0
    """Skip the splice while more than this many *fresh* (unhashed) blocks sit
    at the **head** of the free queue. Rule 1: an unused block must be consumed
    before any cached block is destroyed, and `appendleft` inserts ahead of
    everything.

    Measured at the head, not over the whole queue. A block freed without a
    hash is appended to the *tail* and is still "fresh", so the whole-queue
    count has a permanent floor in any steady workload — on a real run it never
    dropped below 21, which made the splice skip every tick and the policy
    degrade silently to LRU. Only blocks at the head can be jumped by
    `appendleft`, so only those are the ones Rule 1 is about.

    0 means "skip only while the very next block to be popped is fresh", which
    is the rule stated exactly."""

    score_threshold: float | None = None
    """If set, only blocks scoring below this are spliced. `None` means
    unconditional worst-K. Which of the two is right is explicitly
    unresolved (08 §6)."""

    # --- Treating a sentinel row as an absent one (OFF by default) --------
    #
    # These exist because 80% of scored evictions in the 2026-08-07 run ran
    # on the prediction engine's extreme values rather than on a graded
    # forecast (`prob=0.01, ttnc=1h` on 58% — its end-of-job floors;
    # `prob=1.0, ttnc=60s` on 22% — its reach estimate saturating at the
    # clamp). With the numerator constant the score collapses to
    # `constant / blocks`, i.e. "evict from whichever key holds the most".
    #
    # **They default to off, because the measurement did not support turning
    # them on.** Re-checked with same-millisecond duplicate records excluded,
    # blocks evicted under the low-end sentinel were needed again only 24.3%
    # of the time, against 54.7% for the saturated class and 26.1%/49.3%
    # across the bottom/top score deciles. The floors are *good* decisions —
    # they are the engine correctly saying a finished job's blocks are free
    # to go — and gating them out would neuter the one thing the forecast
    # currently does well.
    #
    # Kept, disabled, for the case they were written for: a source that
    # publishes a true blackout sentinel (every key at the floor because it
    # has no idea, not because the jobs ended). Turn them on only with the
    # come-back rate per forecast class in hand.

    uninformative_prob_at_or_below: float = -1.0
    """A row whose `prob` is at or below this is treated as absent, so its
    key stays unscored and keeps LRU order (01 §6 Rule 2).

    Negative disables the gate, which is the default. `0.0` still gates,
    because a published probability of exactly zero is a floor, not a
    prediction."""

    uninformative_ttnc_at_or_above_ms: float = 0.0
    """A row predicting the next call this far out is treated as absent.

    `0` or negative disables the gate, which is the default."""

    # --- Index lifetime (01 §5 Age) ---------------------------------------
    index_hard_drop_age_ms: float = 1_800_000.0
    """Entries untouched for this long are dropped so a silent job cannot
    grow the index for the lifetime of the server."""

    index_gc_period_ms: float = 60_000.0
    """How often to sweep for the above."""

    # --- Observability (07) -----------------------------------------------
    decision_log_path: str | None = None
    """When set, one JSONL line per evicted *cached* block. Unbounded under
    pressure, hence flag-gated; the counters in §6 are always on."""

    regret_window_ms: float = 120_000.0
    """Horizon H for the always-on regret counter: an eviction whose key is
    requested again within this window is counted as regret."""

    regret_buffer_size: int = 4096
    """Bound on the ring buffer backing that counter."""

    hbm_summary_period_ms: float = 30_000.0
    """How often the tick may log one INFO line of HBM block accounting:
    how many blocks exist, how many are in use, how many sit in the free
    queue, how many the splice reshuffled, and which node keys the evictions
    came from. Rate limited *and* change-gated like the prefetch line, so an
    idle server stays silent. Set to 0 to turn the line off.

    The line is emitted in the same `kv_hbm ...` key=value shape by the LRU
    baseline, so the two runs diff directly. Without it there is no way to
    tell a policy that reshuffled nothing from one that reshuffled
    constantly — both look identical in the hit rate until the workload
    changes."""

    remat_window_blocks: int = 0
    """How many recently-evicted block hashes to remember when counting
    rematerialisation — a block cached again after being evicted, i.e. work
    the eviction forced us to redo. `0` sizes the ring to the pool, which is
    the meaningful horizon: a block evicted more than one full turnover ago
    is not something this policy is still answerable for.

    This is the number the policy is judged on. Hit rate alone cannot
    separate a better policy from more HBM, and raw miss counts credit a
    policy for cold starts it had no part in."""

    hbm_summary_top_keys: int = 5
    """How many `job_id:node` keys the HBM line names as eviction sources.
    Counted per window and reset after each line, which is also what bounds
    the memory: a finished job stops appearing instead of accumulating."""

    # --- Redis transport (01 §2) ------------------------------------------
    redis_url: str | None = None
    """e.g. redis://localhost:6379/0. When unset the policy runs with
    whatever snapshot source it was given (a fixture, or none)."""

    redis_prob_pattern: str = "PROB|*"
    redis_history_pattern: str = "HISTORY|*"
    redis_info_pattern: str = "INFO|*"
    """Key/channel patterns. Whether the channel is the key itself or one
    global channel is question 3 in 03 §7 and is not yet answered; these
    default to the per-key form."""

    redis_payload_is_document: bool = True
    """True: a published message carries the full JSON document. False: it
    is a change notification and the thread must GET the key (03 §3.3)."""

    redis_message_timeout_s: float = 1.0
    redis_reseed_period_s: float = 60.0
    redis_reconnect_backoff_s: float = 1.0
    redis_reconnect_backoff_max_s: float = 30.0

    @classmethod
    def from_env(cls) -> "NodeEvictionConfig":
        """Build from `VLLM_NODE_EVICTION_*`, with an optional JSON file
        supplying the rest.

        Precedence: JSON file < explicit env var. The file is where the
        replay harness writes a swept parameter set.
        """
        from vllm import envs

        cfg = cls()

        config_path = envs.VLLM_NODE_EVICTION_CONFIG
        if config_path:
            try:
                with open(config_path) as f:
                    overrides = json.load(f)
            except (OSError, ValueError):
                logger.exception(
                    "Failed to read node eviction config %s; using defaults",
                    config_path,
                )
            else:
                cfg = cls.from_dict(overrides)

        if envs.VLLM_NODE_EVICTION_POLICY is not None:
            cfg.enabled = bool(envs.VLLM_NODE_EVICTION_POLICY)
        if envs.VLLM_NODE_EVICTION_REDIS_URL:
            cfg.redis_url = envs.VLLM_NODE_EVICTION_REDIS_URL
        if envs.VLLM_NODE_EVICTION_DECISION_LOG:
            cfg.decision_log_path = envs.VLLM_NODE_EVICTION_DECISION_LOG
        # Assigned unconditionally, unlike the tunables above: the front-end
        # drainer reads this same env var, and a JSON file that could switch
        # the engine half on by itself would accumulate wants nobody submits
        # — which reads as a broken forecast, not a config mistake.
        cfg.prefetch_wants_enabled = bool(envs.VLLM_NODE_EVICTION_PREFETCH_DRAIN)

        return cfg

    @classmethod
    def from_dict(cls, values: dict) -> "NodeEvictionConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            logger.warning(
                "Ignoring unknown node eviction config keys: %s",
                ", ".join(sorted(unknown)),
            )
        return cls(**{k: v for k, v in values.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        """Fail loudly on a configuration that cannot produce a sane
        ranking, rather than silently ranking everything the same."""
        if self.tau_ms <= 0:
            raise ValueError("tau_ms must be > 0")
        if self.splice_max_blocks < 0:
            raise ValueError("splice_max_blocks must be >= 0")
        if self.tick_period_ms < 0:
            raise ValueError("tick_period_ms must be >= 0")
        if self.delta_cold_ms < self.delta_l1_ms:
            raise ValueError(
                "delta_cold_ms must be >= delta_l1_ms; a cold prefill cannot "
                "be cheaper than an L1 reconstruct"
            )
        if self.speculative_floor_high < self.delta_cold_ms:
            # Invariant 6 in 08 §5: scores are bounded above by delta_cold,
            # so a floor below it would not actually protect anything.
            raise ValueError(
                "speculative_floor_high must exceed delta_cold_ms so the "
                "floor sits above the whole score range"
            )
        if self.speculative_floor_low >= self.speculative_floor_high:
            raise ValueError("speculative floor must decay downwards")
        if not 0.0 <= self.prefetch_min_prob <= 1.0:
            raise ValueError("prefetch_min_prob must be a probability in [0, 1]")
        if self.prefetch_max_outstanding < 0:
            raise ValueError("prefetch_max_outstanding must be >= 0")
        if self.prefetch_max_per_drain < 0:
            raise ValueError("prefetch_max_per_drain must be >= 0")
        if self.hbm_summary_top_keys < 0:
            raise ValueError("hbm_summary_top_keys must be >= 0")
        if self.remat_window_blocks < 0:
            raise ValueError("remat_window_blocks must be >= 0")
        if self.uninformative_prob_at_or_below > 1.0:
            # A gate at prob > 1 would make every row uninformative and the
            # policy permanently inert, which is a silent failure rather than
            # a loud one.
            raise ValueError(
                "uninformative_prob_at_or_below must be <= 1.0; a gate above "
                "the probability range would make every key unscored"
            )
        if self.prefetch_wants_enabled and not self.prefetch_agent_namespace:
            # An empty namespace yields agent_id ":research", which matches
            # nothing in the registry — every want would fan out to zero
            # phantoms and the failure would be invisible.
            raise ValueError(
                "prefetch_agent_namespace must be non-empty when prefetch "
                "origination is on"
            )


def load_config_from_file(path: str | os.PathLike) -> NodeEvictionConfig:
    with open(path) as f:
        return NodeEvictionConfig.from_dict(json.load(f))
