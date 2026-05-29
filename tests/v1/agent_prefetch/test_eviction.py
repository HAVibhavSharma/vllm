# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the agent-aware early-eviction policy.

These tests exercise the policy in isolation -- no engine, no scheduler,
no real block pool. They cover the public contract:

* probability aggregation across live requests
* threshold-based low-probability agent selection
* block tagging / untagging with the request-id and explicit forms
* LRU-ish ordering of returned evictable blocks
* clear / unregister side effects
* defensive empty-state behavior
"""

from __future__ import annotations

import time

import pytest

from vllm.v1.agent_prefetch import (
    DEFAULT_EVICTION_THRESHOLD,
    DEFAULT_EVICTION_WINDOW,
    DEFAULT_PROBABILITY_TTL_SECONDS,
    AgentEvictionPolicy,
)


@pytest.fixture
def policy() -> AgentEvictionPolicy:
    """A fresh policy instance per test -- avoids cross-test leakage
    from the module-level singleton."""
    return AgentEvictionPolicy()


# ---- registration ------------------------------------------------------


def test_register_request_records_probabilities(policy):
    policy.register_request(
        request_id="r1",
        agent_id="A",
        agent_probabilities={"A": 0.8, "B": 0.1},
        window=4,
        threshold=0.5,
    )
    assert policy.aggregated_probability("A") == pytest.approx(0.8)
    assert policy.aggregated_probability("B") == pytest.approx(0.1)
    # Unknown agents fall through to 0.
    assert policy.aggregated_probability("C") == 0.0


def test_register_request_with_no_agent_id_is_noop(policy):
    policy.register_request(
        request_id="r1",
        agent_id="",
        agent_probabilities={"A": 0.9},
    )
    # Nothing should have been registered.
    assert policy.stats()["active_requests"] == 0
    assert policy.aggregated_probability("A") == 0.0


def test_register_request_default_threshold_and_window(policy):
    policy.register_request(request_id="r1", agent_id="A")
    assert policy.effective_threshold() == DEFAULT_EVICTION_THRESHOLD
    # We don't expose window directly, but registering with no window
    # uses the default. Re-registering should be idempotent.
    policy.register_request(request_id="r1", agent_id="A", window=None)
    assert policy.effective_threshold() == DEFAULT_EVICTION_THRESHOLD


def test_unregister_request_removes_from_aggregation(policy):
    policy.register_request(
        request_id="r1", agent_id="A", agent_probabilities={"X": 0.9}
    )
    policy.register_request(
        request_id="r2", agent_id="B", agent_probabilities={"X": 0.2}
    )
    assert policy.aggregated_probability("X") == pytest.approx(0.9)

    policy.unregister_request("r1")
    assert policy.aggregated_probability("X") == pytest.approx(0.2)


# ---- aggregation -------------------------------------------------------


def test_aggregated_probability_is_max_across_requests(policy):
    policy.register_request(
        request_id="r1", agent_id="caller1", agent_probabilities={"X": 0.3}
    )
    policy.register_request(
        request_id="r2", agent_id="caller2", agent_probabilities={"X": 0.7}
    )
    policy.register_request(
        request_id="r3", agent_id="caller3", agent_probabilities={"X": 0.5}
    )
    # Max == 0.7
    assert policy.aggregated_probability("X") == pytest.approx(0.7)


def test_effective_threshold_uses_min_across_requests(policy):
    policy.register_request(
        request_id="r1", agent_id="A", threshold=0.4
    )
    policy.register_request(
        request_id="r2", agent_id="B", threshold=0.7
    )
    # min => 0.4. The most lenient cutoff wins so we never evict a
    # block one of the live requests still considers borderline.
    assert policy.effective_threshold() == pytest.approx(0.4)


def test_effective_threshold_default_when_empty(policy):
    assert policy.effective_threshold() == DEFAULT_EVICTION_THRESHOLD


# ---- block tagging -----------------------------------------------------


def test_tag_block_via_request_id(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9},
        threshold=0.5,
    )
    for bid in (1, 2, 3):
        policy.tag_block(block_id=bid, request_id="r1")

    # A is above threshold => no early eviction candidates.
    assert policy.evictable_blocks(num_needed=5) == []


def test_tag_block_via_request_id_unknown_request_is_noop(policy):
    policy.tag_block(block_id=42, request_id="not-registered")
    assert policy.stats()["tagged_blocks"] == 0


def test_tag_block_explicit_does_not_require_live_request(policy):
    policy.tag_block_explicit(block_id=7, agent_id="ghost")
    # No live request mentions "ghost", so aggregated prob is 0 < 0.5,
    # and threshold defaults to 0.5 when no live request is present.
    # But evictable_blocks returns empty when there are no live
    # requests (we only evict early when at least one request is
    # actively voting).
    assert policy.evictable_blocks(num_needed=5) == []

    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9},  # ghost not mentioned -> 0.0
        threshold=0.5,
    )
    assert 7 in policy.evictable_blocks(num_needed=5)


def test_tag_block_reassign_moves_block_between_agents(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9, "B": 0.1},  # B is low-prob
        threshold=0.5,
    )
    policy.tag_block_explicit(block_id=1, agent_id="A")
    assert policy.evictable_blocks(num_needed=5) == []

    # Reassign block 1 to agent B (the low-prob agent).
    policy.tag_block_explicit(block_id=1, agent_id="B")
    assert policy.evictable_blocks(num_needed=5) == [1]


def test_untag_block_drops_block_from_index(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"low": 0.1},
        threshold=0.5,
    )
    policy.tag_block_explicit(block_id=10, agent_id="low")
    policy.tag_block_explicit(block_id=11, agent_id="low")
    assert set(policy.evictable_blocks(num_needed=5)) == {10, 11}

    policy.untag_block(block_id=10)
    assert policy.evictable_blocks(num_needed=5) == [11]


def test_clear_all_blocks(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"low": 0.0},
        threshold=0.5,
    )
    policy.tag_block_explicit(block_id=1, agent_id="low")
    policy.tag_block_explicit(block_id=2, agent_id="low")
    assert policy.stats()["tagged_blocks"] == 2

    policy.clear_all_blocks()
    assert policy.stats()["tagged_blocks"] == 0
    assert policy.evictable_blocks(num_needed=5) == []


# ---- evictable_blocks: low-probability filtering -----------------------


def test_evictable_blocks_returns_only_low_probability_agents(policy):
    # A above threshold (safe), B and C below threshold (evictable).
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"A": 0.9, "B": 0.2, "C": 0.0},
        threshold=0.5,
    )
    policy.tag_block_explicit(block_id=100, agent_id="A")
    policy.tag_block_explicit(block_id=200, agent_id="B")
    policy.tag_block_explicit(block_id=300, agent_id="C")

    chosen = set(policy.evictable_blocks(num_needed=5))
    assert chosen == {200, 300}


def test_evictable_blocks_respects_num_needed_cap(policy):
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"A": 0.1},
        threshold=0.5,
    )
    for bid in range(10):
        policy.tag_block_explicit(block_id=bid, agent_id="A")
    picks = policy.evictable_blocks(num_needed=3)
    assert len(picks) == 3
    assert set(picks).issubset(set(range(10)))


def test_evictable_blocks_empty_when_no_live_request(policy):
    # Block ownership exists but no live request is voting.
    policy.tag_block_explicit(block_id=1, agent_id="A")
    assert policy.evictable_blocks(num_needed=5) == []


def test_evictable_blocks_empty_when_all_agents_above_threshold(policy):
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"A": 0.9, "B": 0.8},
        threshold=0.5,
    )
    policy.tag_block_explicit(block_id=1, agent_id="A")
    policy.tag_block_explicit(block_id=2, agent_id="B")
    assert policy.evictable_blocks(num_needed=5) == []


def test_evictable_blocks_returns_lru_ordering_within_agent(policy):
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"A": 0.1},
        threshold=0.5,
    )
    # tag in order 5, 3, 8 -> insertion order preserved -> oldest
    # first.
    policy.tag_block_explicit(block_id=5, agent_id="A")
    policy.tag_block_explicit(block_id=3, agent_id="A")
    policy.tag_block_explicit(block_id=8, agent_id="A")
    picks = policy.evictable_blocks(num_needed=10)
    assert picks == [5, 3, 8]


# ---- aggregation edge cases --------------------------------------------


def test_aggregation_after_one_request_finishes(policy):
    policy.register_request(
        request_id="r1", agent_id="A", agent_probabilities={"X": 0.9}
    )
    policy.register_request(
        request_id="r2", agent_id="B", agent_probabilities={"X": 0.1}
    )
    policy.tag_block_explicit(block_id=1, agent_id="X")

    # While r1 votes high for X, block 1 is safe.
    assert policy.evictable_blocks(num_needed=5) == []

    # r1 finishes; the remaining vote on X is from r2 (0.1 < 0.5) so
    # block 1 becomes a candidate.
    policy.unregister_request("r1")
    assert policy.evictable_blocks(num_needed=5) == [1]


def test_window_default_in_active_info(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.1},
    )
    info = policy._active["r1"]  # noqa: SLF001 (internal sanity check)
    assert info.window == DEFAULT_EVICTION_WINDOW


# ---- TTL on probability votes ------------------------------------------


def test_ttl_default_constant_is_positive():
    assert DEFAULT_PROBABILITY_TTL_SECONDS > 0


def test_ttl_none_disables_expiry(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"X": 0.9},
        ttl_seconds=None,
    )
    policy.tag_block_explicit(1, agent_id="X")
    time.sleep(0.1)
    # Vote should still count.
    assert policy.aggregated_probability("X") == pytest.approx(0.9)
    assert policy.evictable_blocks(5) == []


def test_ttl_zero_disables_expiry(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"X": 0.9},
        ttl_seconds=0,
    )
    time.sleep(0.1)
    assert policy.aggregated_probability("X") == pytest.approx(0.9)


def test_ttl_vote_expires_and_block_becomes_evictable(policy):
    # r1 votes high for X (0.9) with a very short TTL.
    policy.register_request(
        request_id="r1", agent_id="caller1",
        agent_probabilities={"X": 0.9},
        threshold=0.5,
        ttl_seconds=0.05,
    )
    # r2 keeps X at 0.1 with no TTL.
    policy.register_request(
        request_id="r2", agent_id="caller2",
        agent_probabilities={"X": 0.1},
        threshold=0.5,
        ttl_seconds=None,
    )
    policy.tag_block_explicit(100, agent_id="X")
    # Before r1's TTL elapses, X is safe (max vote = 0.9).
    assert policy.evictable_blocks(5) == []

    time.sleep(0.1)  # past r1's TTL
    # r1's vote is stale -> only r2's 0.1 counts -> X is below
    # threshold -> block 100 becomes evictable.
    assert policy.evictable_blocks(5) == [100]


def test_ttl_sweep_drops_request_from_active(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9},
        ttl_seconds=0.05,
    )
    assert policy.stats()["active_requests"] == 1
    time.sleep(0.1)
    # Any query path triggers a sweep.
    policy.aggregated_probability("A")
    assert policy.stats()["active_requests"] == 0


def test_ttl_sweep_drops_request_to_agent_mapping(policy):
    """After sweep, ``tag_block`` via request_id is a no-op because the
    request's agent association has been dropped."""
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9},
        ttl_seconds=0.05,
    )
    time.sleep(0.1)
    # Trigger a sweep.
    policy.aggregated_probability("A")
    # Now ``tag_block`` should silently no-op for the swept request.
    policy.tag_block(block_id=42, request_id="r1")
    assert policy.stats()["tagged_blocks"] == 0


def test_ttl_only_affects_votes_not_block_tags(policy):
    """Block-ownership tags survive vote expiry: only the *vote* ages
    out, not the agent->block index."""
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"X": 0.9},
        ttl_seconds=0.05,
    )
    policy.tag_block_explicit(7, agent_id="X")
    assert policy.stats()["tagged_blocks"] == 1

    time.sleep(0.1)
    policy.aggregated_probability("X")  # sweep
    # Vote gone but block 7 still tagged to X.
    assert policy.stats()["tagged_blocks"] == 1


def test_ttl_per_request_independent(policy):
    """Different requests can carry different TTLs; one expiring
    doesn't drag down the other."""
    policy.register_request(
        request_id="r_short", agent_id="A",
        agent_probabilities={"X": 0.8},
        ttl_seconds=0.05,
    )
    policy.register_request(
        request_id="r_long", agent_id="B",
        agent_probabilities={"X": 0.6},
        ttl_seconds=10.0,
    )
    time.sleep(0.1)
    # r_short expired; r_long still alive at 0.6.
    assert policy.aggregated_probability("X") == pytest.approx(0.6)
    assert policy.stats()["active_requests"] == 1
