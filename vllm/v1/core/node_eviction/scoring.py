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


def is_uninformative(row: ImportanceRow, config: NodeEvictionConfig) -> bool:
    """Whether a row that *arrived* still carries no forecast.

    The companion to `is_stale`, and the harder case. A stale row is
    detectably old; a sentinel row is fresh, well-formed, and wrong — the
    prediction engine's way of saying "no idea" is a value, not an absence,
    so it flows through scoring and comes out as a confident near-zero.

    That matters because of what the score does when its numerator is
    constant. `score = prob * decay * E_miss / blocks`: hold the first three
    fixed across every key and the ranking is exactly `1/blocks`, i.e. "evict
    from whichever key holds the most".

    **Both gates are off by default** — see the config. On the one workload
    where this was measured the low-end sentinel turned out to mark genuinely
    dead keys (24.3% needed again, against 54.7% for the saturated class), so
    treating it as absent would have discarded real signal. Enable only for a
    source that publishes a true blackout, and only with per-class come-back
    rates in hand.
    """
    prob_gate = config.uninformative_prob_at_or_below
    if prob_gate >= 0.0 and row.prob <= prob_gate:
        return True
    ttnc_gate = config.uninformative_ttnc_at_or_above_ms
    if ttnc_gate > 0.0 and row.time_to_next_call_ms >= ttnc_gate:
        return True
    return False


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


def speculative_floor(config: NodeEvictionConfig) -> float:
    """The soft pin on prefetched entries (02 §5 part 3).

    Constant, and it does not expire. It sits above the top of the normal
    score range, so a prefetched prefix is never selected by the splice until
    a real prefix hit confirms it and the floor stops applying.

    **This used to decay to `speculative_floor_low` over the prediction's own
    deadline**, on the theory that a falsified prediction should become the
    preferred victim rather than merely lose protection. Removed: the
    deadline came from `time_to_next_call_ms`, which is 60s or 3600s and
    nothing between (12 §2), so the decay was not tracking the prediction's
    horizon — it was reading one bit of a broken forecast and, on the 60s
    arm, turning the protection into a countdown to *first-out* status. A
    prefetch whose call landed at 61s paid a prefill and then bought itself
    the worst possible eviction rank.

    The consequence to be aware of: nothing now demotes a wrong prediction.
    An unconfirmed entry keeps the floor until the index GC drops it at
    `index_hard_drop_age_ms`, and while it holds the floor the splice will
    choose *real* blocks over it. Confirmation (`on_prefix_hit`) is the only
    release, so the waste ratio is the number that matters.

    Still a soft pin, not a `ref_cnt` pin: a permanently high score only
    means "never volunteered for eviction". The block stays in the free queue
    at its LRU position and ordinary `popleft` can still take it, so this
    cannot strand a block for the process lifetime the way a hard pin with a
    lost release would.
    """
    return config.speculative_floor_high


def apply_speculative_floor(
    base: ScoreBreakdown,
    config: NodeEvictionConfig,
) -> ScoreBreakdown:
    """`score = max(base, floor)`, for speculative entries only.

    `max`, not `+`: the floor only ever protects and never inflates, so a
    prefetch cannot contaminate the ranking it was supposed to serve. Once
    the entry is confirmed the floor is gone and the block is scored
    honestly.
    """
    floor = speculative_floor(config)
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
    no forecast row, a stale one, or one carrying no signal are simply absent
    — an unscored block is never selected by the splice and therefore keeps
    its LRU position, which is exactly what a neutral score should mean, with
    no arithmetic (08 §4).

    All three exclusions are the same rule: **act only on a forecast that
    says something.** Missing, expired and don't-know are the same state as
    far as this policy is concerned, and treating the third differently from
    the first two is what made a real run lose to LRU.
    """
    table: dict[NodeKey, ScoreBreakdown] = {}
    for key, num_blocks in key_block_counts.items():
        row = snapshot.get(key)
        if row is None or is_stale(row, now_wall_ms, config.staleness_cutoff_ms):
            continue
        if is_uninformative(row, config):
            continue
        table[key] = score_key(row, num_blocks, config)
    return table
