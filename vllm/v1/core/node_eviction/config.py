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

    observe_only: bool = False
    """Run the bookkeeping and the reporting, but never reorder the queue.

    The A/B baseline arm. `VLLM_NODE_EVICTION_POLICY=0` gives *upstream*, not
    a measurable baseline: with no controller there is no `kv_hbm_ttft` line, no
    hit-rate or remat tracking, and nothing to diff the policy arm against
    field by field — which is what that line exists for.

    Set by `VLLM_NODE_EVICTION_OBSERVE=1`, which forces `enabled` on (so the
    hooks fire) and `splice_max_blocks` to 0 (so `_splice` returns on its
    first line and eviction stays pure LRU `popleft`).

    The value table is still rebuilt each tick, so the decision log records
    what the policy *would* have done without it doing anything — the
    counterfactual, on the same trace.

    Not the same as `enabled=False`: observing costs real work in the hot
    path (index maintenance per cached block, the evicted-hash ring, the
    fresh-block counter on the free queue). That is why it is opt-in rather
    than what flag-off does — "off" has to stay byte-identical to upstream or
    the baseline measures the wrong thing."""

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
    policy plans. 30s is a guess (08 §6).

    Only consulted for a time-free `prob`; see `respect_prob_horizon`."""

    respect_prob_horizon: bool = True
    """Skip `decay(ttnc)` for a row whose publisher declared `prob_horizon`.

    Such a row carries `prob = P(fires within the next N calls)`, which
    already encodes the time preference `decay` exists to supply, so applying
    both discounts time twice. Set False to restore the multiply — the point
    of the switch is that the double discount can be measured against this
    rather than assumed. It has no effect on a publisher that does not send
    `prob_horizon`, which is the default on the forecast side too."""

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
    speculative_floor_high: float = 0.0
    """Height of the floor on prefetched (speculative) entries, or **0 to
    turn the floor off**, which is now the default.

    When on it must sit above `delta_cold_ms`, the top of the normal score
    range, so a prefetched prefix is protected against everything.

    **Off by default.** At 1e9 a prefetched key outranked every real key by
    ~87,000x, and because `_splice` takes `max` over a block's owners, one
    phantom whose prefix shared a system preamble lifted that preamble out
    of the candidate set for every node sharing it. The policy's evictions
    were then concentrated entirely on real, confirmed blocks — the floor
    was buying protection for predictions by spending it on observations.
    Nothing demoted a wrong prediction either (`scoring.speculative_floor`),
    so with a bimodal forecast the waste compounded.

    With the floor off a speculative entry is scored like any other: from
    its forecast row if it has one, and not at all if it does not, in which
    case it keeps its LRU position (Rule 2). Provenance is untouched —
    `entry.speculative` is still stamped and `speculative_waste` still
    measures whether the prefetch half is worth running, which is the number
    to watch after this change.

    Set back to a value above `delta_cold_ms` (1e9 was the old default) to
    A/B against the protected behaviour."""

    prefetch_summary_period_ms: float = 30_000.0
    """How often the tick may log one INFO line of speculative-block state.
    Rate limited *and* change-gated, so an idle server stays silent. Set to 0
    to turn the line off; the counters remain readable via
    `KVCacheManager.get_node_eviction_stats()` either way.

    Origination itself lives outside the engine — phantoms arrive over
    `POST /v1/agents/prefetch` — so what this line reports is what landed and
    whether it was used, not what was asked for."""

    # --- Tick and splice (01 §6 Rule 4) -----------------------------------
    tick_period_ms: float = 250.0
    """Wall-clock cadence of the re-score. Step count is not used because
    step duration varies too much."""

    splice_max_blocks: int = 4096
    """Cap on blocks relocated per tick — a **catch-up rate limit**, not a
    per-tick workload.

    This used to be K, the fixed number moved every tick, and it was the
    whole of window 1. The splice only reorders the free queue;
    `get_new_blocks` stays a plain `popleft_n`. So the policy governs exactly
    the prefix of the queue it has managed to order, and everything past that
    is evicted in raw LRU order with the scores having no say. At K=32
    against ~30 evictions per tick (measured), the ordered region was two
    blocks deep, and a single request needing thousands of blocks — up to
    `max_model_len / block_size`, 7,500 on the reference deployment — popped
    through it instantly and took the rest by age.

    The reason K was small is `splice_max_blocks=256` producing 2,054,400
    relocations for 73,134 evictions (12 §2). But that was the tick fighting
    itself: without the already-spliced skip (12 §3.2) every tick re-moved
    the same blocks, 43 times each. With the skip, a block is relocated at
    most once per re-stage, the candidate set is only the blocks that are
    free, scored and *not yet staged*, and steady-state work is therefore
    bounded by the number of newly-freed blocks (~30/tick), not by this cap.
    The cap binds only when there is a genuine deficit — i.e. right after a
    burst, which is exactly when catching up quickly is the point.

    Watch `splice_deficit_blocks`: sustained non-zero means the ordered
    region is not keeping up and window 1 is still open."""

    splice_restage_period_ms: float = 5_000.0
    """How often to re-rank the whole staged region from scratch.

    Necessary once the region is deep. `appendleft_n` puts each tick's batch
    *in front of* the previous one, so the head is a stack of batches —
    newest first, each internally sorted, globally unsorted. At two blocks
    deep that is invisible; at thousands it inverts the policy, because a
    block freed this tick lands ahead of a genuinely worthless block staged a
    minute ago. Clearing the staging record lets one pass re-sort everything
    by current score, which also picks up forecast changes since a block was
    staged.

    Costs one full relink of the staged region per period. 0 disables
    re-staging, which restores the batch-stack behaviour."""

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

    remat_window_blocks: int = 0
    """How many recently-evicted block hashes to remember when counting
    rematerialisation — a block cached again after being evicted, i.e. work
    the eviction forced us to redo. `0` sizes the ring to the pool, which is
    the meaningful horizon: a block evicted more than one full turnover ago
    is not something this policy is still answerable for.

    This is the number the policy is judged on. Hit rate alone cannot
    separate a better policy from more HBM, and raw miss counts credit a
    policy for cold starts it had no part in."""

    ttft_per_request_log: bool = True
    """One INFO `kv_hbm_ttft` line per request that produced a token, giving
    that request's engine-side TTFT and the token breakdown behind it.

    On by default because it is the *only* place TTFT is reported. A mean or a
    percentile computed in-process is fixed at write time to a window nobody
    chose, and TTFT is heavy-tailed enough that such a figure routinely
    describes no request that produced it. Raw samples can be aggregated
    afterwards over exactly the requests being asked about — a warm subset, a
    single job, the tail — which no in-process summary can be re-cut into.

    The cost is one line per request, which is real under load. Set False
    when running with the policy in production rather than in an
    experiment."""

    ttft_chat_completions_only: bool = True
    """Take a TTFT sample only from requests that arrived on
    `/v1/chat/completions` (identified by the `chatcmpl-` request-id prefix
    the OpenAI front end sets), and never from a phantom prefetch.

    On by default because everything else the engine admits answers a
    different question. A phantom prefetch, an APC warming submission or a
    raw `/v1/completions` call has no user waiting on its first token, and
    the warming traffic is *originated by the policy being measured* — folded
    into one figure it lets a policy improve its own TTFT by issuing more of
    the requests it is cheap at. Set False to sample every request that
    produces a token, e.g. when the run is not driven through the chat API at
    all.

    Applies to the per-request `kv_hbm_ttft` lines and to the `ttft_n`
    counter reported by `stats()`, so the two stay one population."""

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

        if envs.VLLM_NODE_EVICTION_OBSERVE:
            # Last, so it wins over both the JSON file and the switches
            # above. Observe mode is a claim about the *whole* run — "this
            # arm changes nothing" — and a file that could re-enable the
            # splice underneath it would produce a baseline that silently
            # is not one.
            cfg.observe_only = True
            cfg.enabled = True
            cfg.splice_max_blocks = 0

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
        if self.splice_restage_period_ms < 0:
            raise ValueError("splice_restage_period_ms must be >= 0")
        if self.tick_period_ms < 0:
            raise ValueError("tick_period_ms must be >= 0")
        if self.delta_cold_ms < self.delta_l1_ms:
            raise ValueError(
                "delta_cold_ms must be >= delta_l1_ms; a cold prefill cannot "
                "be cheaper than an L1 reconstruct"
            )
        if self.speculative_floor_high < 0.0:
            raise ValueError("speculative_floor_high must be >= 0")
        if 0.0 < self.speculative_floor_high < self.delta_cold_ms:
            # Invariant 6 in 08 §5: scores are bounded above by delta_cold,
            # so a floor below it would not actually protect anything. 0 is
            # exempt because it means "no floor" rather than "a floor that
            # does nothing" — the difference matters, since a floor between
            # 0 and delta_cold protects some keys and not others, which is
            # a silent partial policy rather than a switch.
            raise ValueError(
                "speculative_floor_high must be 0 (off) or exceed "
                "delta_cold_ms so the floor sits above the whole score range"
            )
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


def load_config_from_file(path: str | os.PathLike) -> NodeEvictionConfig:
    with open(path) as f:
        return NodeEvictionConfig.from_dict(json.load(f))
