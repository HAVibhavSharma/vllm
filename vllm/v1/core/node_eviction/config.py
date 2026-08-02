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

    # --- Tick and splice (01 §6 Rule 4) -----------------------------------
    tick_period_ms: float = 250.0
    """Wall-clock cadence of the re-score. Step count is not used because
    step duration varies too much."""

    splice_max_blocks: int = 256
    """K: how many blocks may be moved to the head per tick. The policy
    converges over several ticks; approximate order is fine for a
    heuristic."""

    fresh_skip_threshold: int = 0
    """Skip the splice entirely while more than this many *fresh* (unhashed)
    blocks are queued. Rule 1: an unused block must be consumed before any
    cached block is destroyed, and `appendleft` inserts ahead of everything."""

    score_threshold: float | None = None
    """If set, only blocks scoring below this are spliced. `None` means
    unconditional worst-K. Which of the two is right is explicitly
    unresolved (08 §6)."""

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


def load_config_from_file(path: str | os.PathLike) -> NodeEvictionConfig:
    with open(path) as f:
        return NodeEvictionConfig.from_dict(json.load(f))
