# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The eight invariants from 08 §5.

These are properties of the formula, checkable before any trace exists.
Several of them encode failures the previous attempt actually shipped.
"""

import pytest

from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.scoring import (
    apply_speculative_floor,
    build_value_table,
    decay,
    expected_miss_cost,
    is_stale,
    score_key,
    speculative_floor,
)
from vllm.v1.core.node_eviction.types import (
    ImportanceRow,
    ImportanceSnapshot,
    NodeKey,
)

CONFIG = NodeEvictionConfig()
KEY = NodeKey("run-42", "research", "tavily:summary")


def row(**kwargs) -> ImportanceRow:
    base = {"prob": 0.5, "time_to_next_call_ms": 10_000.0, "update_ts_ms": 1.0}
    base.update(kwargs)
    return ImportanceRow(**base)


# -- invariant 1: strictly increasing in prob ---------------------------


def test_increasing_in_prob():
    low = score_key(row(prob=0.2), 100, CONFIG).score
    high = score_key(row(prob=0.9), 100, CONFIG).score
    assert high > low


# -- invariant 2: strictly decreasing in time_to_next_call --------------


def test_decreasing_in_time_to_next_call():
    soon = score_key(row(time_to_next_call_ms=8_000.0), 100, CONFIG).score
    later = score_key(row(time_to_next_call_ms=60_000.0), 100, CONFIG).score
    assert soon > later


def test_the_walkthrough_separation_comes_from_timing_not_prob():
    """06 §3.3: both nodes will certainly run again, so `prob` barely
    separates them. At tau = 30s the discount is 0.79 for a node due in 8s
    and 0.33 for one due in 60s — a 2.4x gap from timing alone."""
    assert decay(8_000.0, 30_000.0) == pytest.approx(0.789, abs=0.01)
    assert decay(60_000.0, 30_000.0) == pytest.approx(0.333, abs=0.01)

    research = score_key(row(prob=0.92, time_to_next_call_ms=8_000.0), 1500, CONFIG)
    supervisor = score_key(
        row(prob=0.85, time_to_next_call_ms=60_000.0), 1600, CONFIG
    )
    assert research.score > supervisor.score


# -- invariant 3: strictly decreasing in blocks (it is a density) -------


def test_decreasing_in_blocks():
    small = score_key(row(), 100, CONFIG).score
    large = score_key(row(), 1600, CONFIG).score
    assert small > large


# -- invariant 4: increasing in p_cold ----------------------------------


def test_increasing_in_p_cold():
    """Expensive misses are worth avoiding. At a 60x cost ratio, p_cold = 2%
    already doubles expected eviction cost (01 §5)."""
    warm = score_key(row(p_l1=1.0, p_cold=0.0), 100, CONFIG).score
    cold = score_key(row(p_l1=0.98, p_cold=0.02), 100, CONFIG).score
    assert cold > warm
    assert cold > 1.9 * warm


def test_e_miss_defaults_to_the_l1_assumption():
    assert expected_miss_cost(ImportanceRow(), CONFIG) == CONFIG.delta_l1_ms
    # No distribution published at all: fall back to the assumption rather
    # than to a zero cost, which would rank the prefix worthless.
    assert (
        expected_miss_cost(ImportanceRow(p_l1=0.0, p_cold=0.0), CONFIG)
        == CONFIG.delta_l1_ms
    )


def test_e_miss_is_constant_under_the_l1_assumption():
    """v1 collapses to prob * decay / blocks: a constant does not change a
    ranking (08 §3)."""
    a = expected_miss_cost(row(), CONFIG)
    b = expected_miss_cost(row(prob=0.9, time_to_next_call_ms=1.0), CONFIG)
    assert a == b


# -- invariant 5: never 0 for a scored block ----------------------------


def test_never_zero():
    """0 makes freshly built prefixes the top eviction candidates and
    destroys them before first reuse — strictly worse than LRU."""
    assert score_key(row(prob=0.0), 100, CONFIG).score > 0.0
    assert score_key(row(prob=0.0), 1_000_000, CONFIG).score > 0.0
    assert score_key(row(), 0, CONFIG).score > 0.0


# -- invariant 6: bounded above by delta_cold ---------------------------


def test_bounded_above_by_delta_cold():
    """So the speculative floor can be placed above the whole range."""
    worst_case = score_key(
        row(prob=1.0, time_to_next_call_ms=0.0, p_l1=0.0, p_cold=1.0),
        1,
        CONFIG,
    )
    assert worst_case.score <= CONFIG.delta_cold_ms
    assert CONFIG.speculative_floor_high > CONFIG.delta_cold_ms


# -- invariant 7: deterministic, no clock reads -------------------------


def test_deterministic():
    """Reading `time.monotonic()` inside the score rather than passing a tick
    timestamp in makes replay non-reproducible and the harness useless."""
    r = row()
    scores = {score_key(r, 100, CONFIG).score for _ in range(50)}
    assert len(scores) == 1


# -- invariant 8: a running node's blocks are never scored --------------
# Structural: ref_cnt > 0 keeps them out of the free queue entirely, so the
# splice never sees them. Covered in test_controller.py.


# -- decay ---------------------------------------------------------------


def test_decay_shape():
    assert decay(0.0, 30_000.0) == 1.0
    assert decay(30_000.0, 30_000.0) == pytest.approx(0.5)
    # Fatter tail than an exponential: an 11.4s rebuild is still worth
    # avoiding at t = 2min.
    assert decay(120_000.0, 30_000.0) > 0.15


def test_decay_is_monotone():
    previous = 2.0
    for t in range(0, 200_000, 5_000):
        current = decay(float(t), 30_000.0)
        assert current < previous
        previous = current


# -- staleness gate ------------------------------------------------------


def test_staleness_gate():
    """The backstop that makes a missed pub/sub message degrade to LRU
    rather than act on stale data (01 §2)."""
    fresh = ImportanceRow(update_ts_ms=1_000_000.0)
    assert not is_stale(fresh, 1_000_100.0, 120_000.0)
    assert is_stale(fresh, 1_500_000.0, 120_000.0)


def test_row_without_a_timestamp_is_stale():
    """A row without update_ts defeats the gate entirely, so treat it as
    stale rather than as fresh (05 §5.4)."""
    assert is_stale(ImportanceRow(update_ts_ms=0.0), 1000.0, 120_000.0)


# -- the speculative floor (02 §5) --------------------------------------


def test_floor_starts_above_the_score_range_and_decays_below_it():
    at_start = speculative_floor(0.0, 10_000.0, CONFIG)
    at_ttl = speculative_floor(10_000.0, 10_000.0, CONFIG)
    assert at_start > CONFIG.delta_cold_ms
    # Decays past neutral, so a falsified prediction becomes the *preferred*
    # victim rather than merely losing protection.
    assert at_ttl < 0.0
    assert at_ttl == pytest.approx(CONFIG.speculative_floor_low)


def test_floor_decays_monotonically():
    previous = float("inf")
    for age in range(0, 11_000, 1_000):
        current = speculative_floor(float(age), 10_000.0, CONFIG)
        assert current < previous
        previous = current


def test_floor_only_ever_protects():
    """`max`, not `+`: the floor never inflates a score, so a prefetch cannot
    contaminate the ranking it was supposed to serve."""
    base = score_key(row(), 100, CONFIG)
    # Well past TTL, so the floor is at its low value and loses to the base.
    floored = apply_speculative_floor(base, 100_000.0, 10_000.0, CONFIG)
    assert floored.score == base.score
    assert floored.speculative

    # Fresh prefetch: the floor dominates.
    protected = apply_speculative_floor(base, 0.0, 10_000.0, CONFIG)
    assert protected.score > base.score


def test_floor_falls_back_to_the_default_ttl():
    assert speculative_floor(0.0, 0.0, CONFIG) == pytest.approx(
        CONFIG.speculative_floor_high
    )


# -- the value table -----------------------------------------------------


def test_value_table_skips_unscored_and_stale_keys():
    other = NodeKey("run-42", "supervisor", "plan")
    stale = NodeKey("run-42", "writer", "draft")
    snapshot = ImportanceSnapshot(
        rows={
            KEY: ImportanceRow(prob=0.9, update_ts_ms=1_000_000.0),
            stale: ImportanceRow(prob=0.9, update_ts_ms=1.0),
        }
    )
    table = build_value_table(
        snapshot,
        {KEY: 100, other: 100, stale: 100},
        CONFIG,
        now_wall_ms=1_000_100.0,
    )
    assert KEY in table
    # No forecast row at all -> absent -> never selected -> keeps its LRU
    # position, which is what neutral means with no arithmetic (08 §4).
    assert other not in table
    assert stale not in table
