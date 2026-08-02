# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The v1 scoring function — plan/new-eviction/08-scoring.md.

The quantity is a **value density**: expected milliseconds of latency saved
per block of HBM held.

    score(key) = prob(key) * decay(ttnc) * E_miss(key) / blocks(key)
    decay(t)   = tau / (tau + t)
    E_miss     = p_l1 * delta_l1 + p_cold * delta_cold

Not "how likely is this to be reused" (ignores cost and size) and not "how
expensive was it to build" (ignores whether it will ever be reused). A score
in ms/block is comparable across nodes and jobs and against a wall-clock
budget; a unitless "importance" is not, and it makes every constant
unfalsifiable.

Nothing in this module reads a clock. The tick timestamp is passed in, which
is what makes the offline replay harness reproducible (08 §5, invariant 7).
"""

from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.types import (
    ImportanceRow,
    ImportanceSnapshot,
    NodeKey,
    ScoreBreakdown,
)

# Scores must never be exactly 0 for a scored block: 0 would make freshly
# built prefixes the top eviction candidates and destroy them before their
# first reuse, which is strictly worse than LRU (08 §5, invariant 5).
_MIN_SCORE = 1e-12


def decay(ttnc_ms: float, tau_ms: float) -> float:
    """`tau / (tau + t)` — the time discount.

    Chosen over `exp(-t/tau)` for its fatter tail: an 11.4s prefix rebuild is
    still worth avoiding at t = 2min, and an exponential writes that off far
    too fast.
    """
    if ttnc_ms <= 0.0:
        return 1.0
    return tau_ms / (tau_ms + ttnc_ms)


def expected_miss_cost(row: ImportanceRow, config: NodeEvictionConfig) -> float:
    """`p_l1 * delta_l1 + p_cold * delta_cold`.

    Under the current "assume L1 resident" position this is a constant, and
    constants do not change a ranking — so v1 effectively computes
    `prob * decay / blocks`. It is written in full anyway so that publishing
    a measured `p_cold` (04 §5.4) changes a data file rather than this code.
    """
    delta_l1 = row.delta_l1_ms if row.delta_l1_ms is not None else config.delta_l1_ms
    delta_cold = (
        row.delta_cold_ms if row.delta_cold_ms is not None else config.delta_cold_ms
    )
    p_cold = max(0.0, row.p_cold)
    p_l1 = max(0.0, row.p_l1)
    total = p_l1 + p_cold
    if total <= 0.0:
        # No hit-class distribution published: fall back to the assumption
        # rather than to a zero cost, which would rank the prefix worthless.
        return delta_l1
    if total > 1.0:
        # Normalise rather than reject: a p_gpu component may have been
        # included, and a saturating distribution should not inflate cost.
        p_l1 /= total
        p_cold /= total
    return p_l1 * delta_l1 + p_cold * delta_cold


def is_stale(row: ImportanceRow, now_wall_ms: float, cutoff_ms: float) -> bool:
    """Whether a forecast row is too old to act on.

    This is the backstop that makes a silently missed pub/sub message degrade
    to LRU instead of to a confident wrong answer (01 §2).
    """
    if cutoff_ms <= 0.0:
        return False
    if row.update_ts_ms <= 0.0:
        # No timestamp published at all. Treating this as fresh would defeat
        # the gate entirely, so treat it as stale.
        return True
    return (now_wall_ms - row.update_ts_ms) > cutoff_ms


def score_key(
    row: ImportanceRow,
    num_blocks: int,
    config: NodeEvictionConfig,
) -> ScoreBreakdown:
    """Score one `(job, node, call_type)` key.

    `num_blocks` is the prefix length currently held by that key, taken from
    the index at tick time (free, no extra bookkeeping).
    """
    if num_blocks <= 0:
        return ScoreBreakdown(
            score=_MIN_SCORE,
            prob=row.prob,
            ttnc_ms=row.time_to_next_call_ms,
            decay=0.0,
            e_miss_ms=0.0,
            blocks=0,
        )

    d = decay(row.time_to_next_call_ms, config.tau_ms)
    e_miss = expected_miss_cost(row, config)
    prob = min(max(row.prob, 0.0), 1.0)

    # Dividing by blocks is what turns a preference into a density. It is
    # legitimate *because* eviction is tail-first: a prefix truncated from
    # the tail is still a valid match for its first N blocks, so retained
    # value is roughly linear in blocks retained and the marginal value of a
    # block is E_miss / blocks. Without tail-first ordering this term would
    # be a category error (08 §2).
    raw = prob * d * e_miss / num_blocks

    return ScoreBreakdown(
        score=max(raw, _MIN_SCORE),
        prob=prob,
        ttnc_ms=row.time_to_next_call_ms,
        decay=d,
        e_miss_ms=e_miss,
        blocks=num_blocks,
    )


def speculative_floor(
    age_ms: float, ttl_ms: float, config: NodeEvictionConfig
) -> float:
    """The decaying soft pin on prefetched entries (02 §5 part 3).

    Starts above the top of the normal score range and decays to *below* it
    over the prediction's own deadline, so:

    - a correct prediction is protected until the block is used, and
    - a falsified one becomes the preferred victim rather than merely losing
      protection.

    A hard `ref_cnt` pin was rejected: it needs a leak-proof release path, and
    one lost release strands a block for the process lifetime. A hard pin
    turns a scoring bug into a leaked-block bug.
    """
    if ttl_ms <= 0.0:
        ttl_ms = config.speculative_default_ttl_ms
    if ttl_ms <= 0.0:
        return config.speculative_floor_low
    frac = min(max(age_ms / ttl_ms, 0.0), 1.0)
    high = config.speculative_floor_high
    low = config.speculative_floor_low
    return high + (low - high) * frac


def apply_speculative_floor(
    base: ScoreBreakdown,
    age_ms: float,
    ttl_ms: float,
    config: NodeEvictionConfig,
) -> ScoreBreakdown:
    """`score = max(base, floor(age))`, for speculative entries only.

    `max`, not `+`: the floor only ever protects and never inflates, so a
    prefetch cannot contaminate the ranking it was supposed to serve. Once
    the entry is confirmed the floor is gone and the block is scored
    honestly.
    """
    floor = speculative_floor(age_ms, ttl_ms, config)
    return ScoreBreakdown(
        score=max(base.score, floor),
        prob=base.prob,
        ttnc_ms=base.ttnc_ms,
        decay=base.decay,
        e_miss_ms=base.e_miss_ms,
        blocks=base.blocks,
        speculative=True,
        floor_applied=floor,
        scored=True,
    )


def build_value_table(
    snapshot: ImportanceSnapshot,
    key_block_counts: dict[NodeKey, int],
    config: NodeEvictionConfig,
    now_wall_ms: float,
) -> dict[NodeKey, ScoreBreakdown]:
    """Score every tracked key once per tick.

    This is the "tick computes, request reads" split (02 §2): the result is a
    plain dict, and the per-request path is a single lookup in it. Keys with
    no forecast row, or a stale one, are simply absent — an unscored block is
    never selected by the splice and therefore keeps its LRU position, which
    is exactly what a neutral score should mean, with no arithmetic (08 §4).
    """
    table: dict[NodeKey, ScoreBreakdown] = {}
    for key, num_blocks in key_block_counts.items():
        row = snapshot.get(key)
        if row is None or is_stale(row, now_wall_ms, config.staleness_cutoff_ms):
            continue
        table[key] = score_key(row, num_blocks, config)
    return table
