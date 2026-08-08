# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recording prefixes from ordinary `/v1/chat/completions` traffic.

This is the link that was missing end to end. The first real run produced 96
wants, drained all 96, and submitted zero phantoms, because the registry is
otherwise only written by `/v1/agents/*` and the workload never called those.
Nothing here needs an engine or a server — the registry is a plain dict and
the app state is an attribute bag.
"""

from types import SimpleNamespace

from vllm.v1.agent_prefetch import auto_register
from vllm.v1.agent_prefetch.auto_register import (
    agent_id_for_extra_args,
    maybe_record_chat_prefix,
)
from vllm.v1.agent_prefetch.hashing import DEFAULT_CHUNK_SIZE
from vllm.v1.agent_prefetch.registry import AgentPrefixRegistry

MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
# Two full chunks plus a partial one, so alignment has something to drop.
TOKENS = list(range(DEFAULT_CHUNK_SIZE * 2 + 3))
XARGS = {
    "job_id": "run-42",
    "langgraph_node": "researcher",
    "call_type": "tavily:summary",
}


def state_with_registry(**kwargs):
    return SimpleNamespace(
        agent_prefetch_registry=AgentPrefixRegistry(**kwargs)
    )


def record(app_state, tokens=TOKENS, xargs=XARGS, cache_salt=None):
    return maybe_record_chat_prefix(
        app_state,
        extra_args=xargs,
        model_name=MODEL,
        prompt_token_ids=tokens,
        cache_salt=cache_salt,
    )


# -- the join with engine core -------------------------------------------


def test_agent_id_matches_what_engine_core_asks_for(monkeypatch):
    """`PrefetchWant.agent_id` is `f"{namespace}:{key.node}"`. A mismatch
    here is invisible: the drainer looks up an agent stored under a different
    name and logs "no registered prefixes yet" for the life of the server."""
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    assert agent_id_for_extra_args(XARGS) == "langgraph:researcher"


def test_a_request_without_a_node_is_not_agent_traffic(monkeypatch):
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    assert agent_id_for_extra_args(None) is None
    assert agent_id_for_extra_args({"job_id": "run-42"}) is None
    assert agent_id_for_extra_args({"langgraph_node": ""}) is None


# -- recording ------------------------------------------------------------


def test_a_plain_chat_request_populates_the_registry(monkeypatch):
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    state = state_with_registry()

    assert record(state) is True

    descriptors = state.agent_prefetch_registry.get_all("langgraph:researcher")
    assert len(descriptors) == 1
    # Chunk-aligned: the 3-token tail is dropped because LMCache only stores
    # complete chunks.
    assert len(descriptors[0].token_ids) == DEFAULT_CHUNK_SIZE * 2


def test_recording_is_a_no_op_when_origination_is_off(monkeypatch):
    """No registry on the app state means the drainer never started, which
    means nothing will ever read what we record. Paying the hash on every
    chat request for that would be a pure serving-path tax."""
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    assert record(SimpleNamespace()) is False


def test_a_non_agent_request_is_ignored(monkeypatch):
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    state = state_with_registry()
    assert record(state, xargs=None) is False
    assert state.agent_prefetch_registry.num_agents() == 0


def test_a_prompt_shorter_than_one_chunk_is_not_recorded(monkeypatch):
    """`chunk_align` floors to a multiple of the chunk size, and LMCache
    stores nothing below that — a zero-length descriptor would fan out to a
    phantom that warms nothing."""
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    state = state_with_registry()
    assert record(state, tokens=list(range(DEFAULT_CHUNK_SIZE - 1))) is False


def test_the_recorded_salt_is_the_requests_own(monkeypatch):
    """Not a synthetic `agent::` one. The phantom passes this to LMCache, so
    a salt the real request never carried would key its chunks somewhere the
    real request cannot find them."""
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    state = state_with_registry()
    record(state, cache_salt="tenant-7")
    desc = state.agent_prefetch_registry.get_all("langgraph:researcher")[0]
    assert desc.cache_salt == "tenant-7"


def test_the_same_prefix_twice_is_one_descriptor(monkeypatch):
    """Keyed by prefix hash, so a node called repeatedly with an unchanged
    preamble does not grow the registry."""
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    state = state_with_registry()
    record(state)
    record(state)
    assert state.agent_prefetch_registry.agent_size("langgraph:researcher") == 1


def test_growth_is_bounded_per_agent(monkeypatch):
    """One descriptor per turn per node, unbounded over a long run, is why
    the registry now carries a per-agent cap. LRU, so the newest prefixes —
    the ones a repeat call can match — are the survivors."""
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")
    state = state_with_registry(default_top_k=2, max_per_agent=4)
    for i in range(10):
        record(state, tokens=list(range(DEFAULT_CHUNK_SIZE * (i + 1))))
    assert state.agent_prefetch_registry.agent_size("langgraph:researcher") == 4


def test_a_broken_registry_does_not_break_the_chat_request(monkeypatch):
    """This runs inline on the serving hot path. Prefetch bookkeeping must
    never turn into a failed completion."""
    monkeypatch.setattr(auto_register, "_namespace", "langgraph")

    class ExplodingRegistry:
        def record(self, *args, **kwargs):
            raise RuntimeError("boom")

    state = SimpleNamespace(agent_prefetch_registry=ExplodingRegistry())
    assert record(state) is False
