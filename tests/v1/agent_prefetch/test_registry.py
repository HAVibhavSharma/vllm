# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import pytest

from vllm.v1.agent_prefetch.hashing import compute_prefix_hash
from vllm.v1.agent_prefetch.registry import (
    AgentPrefixRegistry,
    PrefixDescriptor,
)


def _make_desc(token_ids: list[int], salt: str = "s") -> PrefixDescriptor:
    return PrefixDescriptor(
        token_ids=tuple(token_ids),
        prefix_hash=compute_prefix_hash("m", salt, token_ids),
        cache_salt=salt,
    )


# ---------------------------------------------------------------------------
# PrefixDescriptor
# ---------------------------------------------------------------------------


def test_descriptor_rejects_empty_tokens():
    with pytest.raises(ValueError):
        PrefixDescriptor(token_ids=(), prefix_hash=b"x", cache_salt="s")


def test_descriptor_rejects_empty_hash():
    with pytest.raises(ValueError):
        PrefixDescriptor(token_ids=(1,), prefix_hash=b"", cache_salt="s")


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_registry_rejects_invalid_caps():
    with pytest.raises(ValueError):
        AgentPrefixRegistry(default_top_k=-1)
    with pytest.raises(ValueError):
        AgentPrefixRegistry(max_agents=0)
    with pytest.raises(ValueError):
        AgentPrefixRegistry(max_per_agent=0)
    with pytest.raises(ValueError):
        AgentPrefixRegistry(default_top_k=100, max_per_agent=10)


# ---------------------------------------------------------------------------
# record + top_k basic
# ---------------------------------------------------------------------------


def test_record_and_top_k_returns_newest_first():
    reg = AgentPrefixRegistry()
    d1 = _make_desc([1, 2, 3, 4])
    d2 = _make_desc([5, 6, 7, 8])
    d3 = _make_desc([9, 10, 11, 12])
    reg.record("agent", d1)
    reg.record("agent", d2)
    reg.record("agent", d3)

    top = reg.top_k("agent", k=3)
    assert [d.token_ids for d in top] == [
        d3.token_ids,
        d2.token_ids,
        d1.token_ids,
    ]


def test_top_k_caps_at_available_entries():
    reg = AgentPrefixRegistry()
    reg.record("agent", _make_desc([1, 2]))
    assert len(reg.top_k("agent", k=10)) == 1


def test_top_k_uses_default_when_k_none():
    reg = AgentPrefixRegistry(default_top_k=2)
    for i in range(5):
        reg.record("a", _make_desc([i, i + 1]))
    assert len(reg.top_k("a")) == 2


def test_top_k_unknown_agent_returns_empty():
    reg = AgentPrefixRegistry()
    assert reg.top_k("missing") == []


def test_top_k_with_zero_k_returns_empty():
    reg = AgentPrefixRegistry()
    reg.record("agent", _make_desc([1, 2]))
    assert reg.top_k("agent", k=0) == []


def test_top_k_is_side_effect_free():
    """Reading must not promote LRU order. Otherwise prefetch fanout
    would itself shuffle the recency we rely on."""
    reg = AgentPrefixRegistry()
    d1 = _make_desc([1])
    d2 = _make_desc([2])
    reg.record("a", d1)
    reg.record("a", d2)
    # Pull twice; order must be stable.
    first = [d.token_ids for d in reg.top_k("a", k=2)]
    second = [d.token_ids for d in reg.top_k("a", k=2)]
    assert first == second == [(2,), (1,)]


# ---------------------------------------------------------------------------
# LRU semantics
# ---------------------------------------------------------------------------


def test_per_agent_lru_eviction():
    reg = AgentPrefixRegistry(default_top_k=3, max_per_agent=3)
    descs = [_make_desc([i, i + 1]) for i in range(5)]
    for d in descs:
        reg.record("a", d)
    # The two oldest (descs[0], descs[1]) should be gone.
    assert reg.agent_size("a") == 3
    top = reg.top_k("a", k=3)
    assert [d.token_ids for d in top] == [
        descs[4].token_ids,
        descs[3].token_ids,
        descs[2].token_ids,
    ]


def test_repeated_record_promotes_existing_entry():
    reg = AgentPrefixRegistry(default_top_k=3, max_per_agent=3)
    d1 = _make_desc([1, 2])
    d2 = _make_desc([3, 4])
    d3 = _make_desc([5, 6])
    reg.record("a", d1)
    reg.record("a", d2)
    reg.record("a", d3)
    # Re-record d1 -> it should now be MRU.
    reg.record("a", d1)
    top = reg.top_k("a", k=3)
    assert [d.token_ids for d in top] == [
        d1.token_ids,
        d3.token_ids,
        d2.token_ids,
    ]
    # Adding a 4th distinct entry should evict d2 (LRU), not d1.
    d4 = _make_desc([7, 8])
    reg.record("a", d4)
    top = reg.top_k("a", k=3)
    assert d2.token_ids not in [d.token_ids for d in top]


def test_cross_agent_lru_eviction():
    reg = AgentPrefixRegistry(default_top_k=1, max_agents=2, max_per_agent=4)
    reg.record("a", _make_desc([1]))
    reg.record("b", _make_desc([2]))
    reg.record("c", _make_desc([3]))
    # Agent "a" was LRU and should be gone.
    assert reg.num_agents() == 2
    assert reg.top_k("a") == []
    assert reg.top_k("b") != []
    assert reg.top_k("c") != []


def test_recording_for_existing_agent_promotes_it():
    reg = AgentPrefixRegistry(default_top_k=1, max_agents=2, max_per_agent=4)
    reg.record("a", _make_desc([1]))
    reg.record("b", _make_desc([2]))
    # Touch "a" again -> "b" becomes LRU.
    reg.record("a", _make_desc([3]))
    reg.record("c", _make_desc([4]))
    # "b" should have been evicted, not "a".
    assert reg.top_k("b") == []
    assert reg.top_k("a") != []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_evict_agent():
    reg = AgentPrefixRegistry()
    reg.record("a", _make_desc([1]))
    assert reg.evict_agent("a")
    assert reg.top_k("a") == []
    assert not reg.evict_agent("a")


def test_record_rejects_empty_agent_id():
    reg = AgentPrefixRegistry()
    with pytest.raises(ValueError):
        reg.record("", _make_desc([1]))


def test_stats_shape():
    reg = AgentPrefixRegistry(default_top_k=5, max_agents=100, max_per_agent=10)
    reg.record("a", _make_desc([1]))
    reg.record("a", _make_desc([2]))
    reg.record("b", _make_desc([3]))
    s = reg.stats()
    assert s["num_agents"] == 2
    assert s["total_descriptors"] == 3
    assert s["max_agents"] == 100
    assert s["max_per_agent"] == 10
    assert s["default_top_k"] == 5


def test_thread_safety_smoke():
    """Hammer the registry from multiple threads. The point is to
    catch obvious lock-related corruption, not to be a stress test."""
    reg = AgentPrefixRegistry(default_top_k=8, max_per_agent=8)
    num_threads = 8
    iters_per_thread = 200

    def worker(tid: int) -> None:
        for i in range(iters_per_thread):
            reg.record(
                f"agent-{tid % 3}",
                _make_desc([tid, i, i + 1]),
            )
            reg.top_k(f"agent-{tid % 3}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No exception means no race-induced corruption made it to the
    # surface. The invariant we can cheaply assert: every agent map is
    # bounded by max_per_agent.
    for agent_id in ("agent-0", "agent-1", "agent-2"):
        assert reg.agent_size(agent_id) <= 8
