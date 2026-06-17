# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the agent-aware eviction policy.

These tests exercise the policy in isolation -- no engine, no scheduler,
no real block pool. They cover the public contract:

* probability aggregation across live requests (max)
* block tagging / untagging with the request-id and explicit forms
* lowest-probability-first ordering of returned evictable blocks
* clear / unregister side effects
* defensive empty-state behavior
"""

from __future__ import annotations

import pytest

from vllm.v1.agent_prefetch import AgentEvictionPolicy


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


def test_self_vote_implicit_one(policy):
    """A live request implicitly self-votes 1.0 for its own agent
    even when it doesn't appear in the probabilities map."""
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"B": 0.2},  # A intentionally missing
    )
    assert policy.aggregated_probability("A") == pytest.approx(1.0)


# ---- block tagging -----------------------------------------------------


def test_tag_block_via_request_id(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9},
    )
    for bid in (1, 2, 3):
        policy.tag_block(block_id=bid, request_id="r1")
    # A is the only active agent and self-voted 1.0 -> excluded from
    # the eviction ranking -> nothing to evict.
    assert policy.evictable_blocks(num_needed=5) == []


def test_tag_block_via_request_id_unknown_request_is_noop(policy):
    policy.tag_block(block_id=42, request_id="not-registered")
    assert policy.stats()["tagged_blocks"] == 0


def test_tag_block_explicit_does_not_require_live_request(policy):
    policy.tag_block_explicit(block_id=7, agent_id="ghost")
    # No live request -> no eviction candidates at all.
    assert policy.evictable_blocks(num_needed=5) == []

    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9},  # ghost unmentioned -> 0.0
    )
    assert 7 in policy.evictable_blocks(num_needed=5)


def test_tag_block_reassign_moves_block_between_agents(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9, "B": 0.1},
    )
    policy.tag_block_explicit(block_id=1, agent_id="A")
    # A is the active agent -> excluded -> no eviction candidates.
    assert policy.evictable_blocks(num_needed=5) == []

    # Reassign block 1 to agent B (non-active, low-prob).
    policy.tag_block_explicit(block_id=1, agent_id="B")
    assert policy.evictable_blocks(num_needed=5) == [1]


def test_untag_block_drops_block_from_index(policy):
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"low": 0.1},
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
    )
    policy.tag_block_explicit(block_id=1, agent_id="low")
    policy.tag_block_explicit(block_id=2, agent_id="low")
    assert policy.stats()["tagged_blocks"] == 2

    policy.clear_all_blocks()
    assert policy.stats()["tagged_blocks"] == 0
    assert policy.evictable_blocks(num_needed=5) == []


# ---- evictable_blocks: lowest-probability-first ranking ----------------


def test_evictable_blocks_drains_lowest_probability_first(policy):
    """All non-active agents are evictable. Lowest probability gets
    drained first, then we walk up the ranking."""
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"A": 0.9, "B": 0.2, "C": 0.0},
    )
    policy.tag_block_explicit(block_id=100, agent_id="A")
    policy.tag_block_explicit(block_id=200, agent_id="B")
    policy.tag_block_explicit(block_id=300, agent_id="C")

    # With enough headroom, every non-active agent gets drained.
    picks = policy.evictable_blocks(num_needed=5)
    assert set(picks) == {100, 200, 300}
    # Ordering: C (0.0) before B (0.2) before A (0.9).
    assert picks.index(300) < picks.index(200) < picks.index(100)


def test_evictable_blocks_respects_num_needed_cap(policy):
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"A": 0.1},
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


def test_evictable_blocks_excludes_active_agents(policy):
    """Currently-active agents (the ones owning live requests) are
    never offered up, regardless of their probability."""
    policy.register_request(
        request_id="r1", agent_id="A",
        agent_probabilities={"A": 0.9, "B": 0.8},
    )
    policy.tag_block_explicit(block_id=1, agent_id="A")
    policy.tag_block_explicit(block_id=2, agent_id="B")
    # A is the active agent -> excluded. B is non-active so its block
    # is fair game even though B's vote is high.
    assert policy.evictable_blocks(num_needed=5) == [2]


def test_evictable_blocks_returns_lru_ordering_within_agent(policy):
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"A": 0.1},
    )
    # tag in order 5, 3, 8 -> insertion order preserved -> oldest first.
    policy.tag_block_explicit(block_id=5, agent_id="A")
    policy.tag_block_explicit(block_id=3, agent_id="A")
    policy.tag_block_explicit(block_id=8, agent_id="A")
    picks = policy.evictable_blocks(num_needed=10)
    assert picks == [5, 3, 8]


def test_evictable_blocks_walks_up_ranking_when_lowest_is_short(policy):
    """If the lowest-probability agent doesn't have enough blocks, the
    walk continues to the next-lowest."""
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"low": 0.1, "mid": 0.4, "high": 0.8},
    )
    policy.tag_block_explicit(block_id=1, agent_id="low")
    policy.tag_block_explicit(block_id=2, agent_id="mid")
    policy.tag_block_explicit(block_id=3, agent_id="mid")
    policy.tag_block_explicit(block_id=4, agent_id="high")

    picks = policy.evictable_blocks(num_needed=3)
    # low (1 block) drained, then mid (2 blocks) drained -> 1, 2, 3.
    assert picks == [1, 2, 3]


# ---- aggregation edge cases --------------------------------------------


def test_aggregation_after_one_request_finishes(policy):
    policy.register_request(
        request_id="r1", agent_id="A", agent_probabilities={"X": 0.9}
    )
    policy.register_request(
        request_id="r2", agent_id="B", agent_probabilities={"X": 0.1}
    )
    policy.tag_block_explicit(block_id=1, agent_id="X")

    # X has probability 0.9 from r1 but is non-active so still
    # evictable -- the new design has no threshold gate.
    assert policy.evictable_blocks(num_needed=5) == [1]

    # r1 finishes; X's max vote drops to 0.1.
    policy.unregister_request("r1")
    assert policy.evictable_blocks(num_needed=5) == [1]


def test_unregister_keeps_block_tags(policy):
    """Block-ownership tags survive request finish so cached work from
    a finished turn can be reused / evicted later."""
    policy.register_request(
        request_id="r1", agent_id="caller",
        agent_probabilities={"X": 0.9},
    )
    policy.tag_block_explicit(7, agent_id="X")
    assert policy.stats()["tagged_blocks"] == 1

    policy.unregister_request("r1")
    # Tag survives.
    assert policy.stats()["tagged_blocks"] == 1
