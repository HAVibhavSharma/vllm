# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/v1/agents/prefetch` must announce the identity its blocks index under.

The endpoint used to call `PhantomPrefetchSubmitter.submit` without an
`identity`, so every prefix it warmed was cached and then dropped on the floor
by the policy: `node_key_for_request` returns None for an anonymous request,
`on_blocks_cached` returns early on `key is None`, and the blocks keep plain
LRU order. Warmed, unscored, first out under pressure.

These tests pin the two halves of the fix — that a complete identity reaches
the submitter, and that a partial one is refused rather than half-attached.
"""

import asyncio
from types import SimpleNamespace

from vllm.entrypoints.openai.agent_chat.api_router import (
    _fan_out_prefetches,
    _prefetch_identity,
)
from vllm.entrypoints.openai.agent_chat.protocol import AgentPrefetchRequest
from vllm.v1.core.node_eviction.controller import node_key_for_request
from vllm.v1.core.node_eviction.types import NodeKey


class RecordingSubmitter:
    def __init__(self) -> None:
        self.submissions: list[dict] = []

    async def submit(self, **kwargs):
        self.submissions.append(kwargs)
        return None  # no task: nothing here awaits the engine


class OnePrefixRegistry:
    def top_k(self, agent_id, k):
        return [
            SimpleNamespace(
                token_ids=(1, 2, 3),
                prefix_hash=b"\xaa" * 8,
                cache_salt=f"agent::{agent_id}",
            )
        ]

    def get_all(self, agent_id):
        return self.top_k(agent_id, None)


def make_request(**overrides) -> AgentPrefetchRequest:
    body = {
        "agent_id": "langgraph:research_supervisor:supervisor_tools:researcher",
        "job_id": "1",
        "langgraph_node": "researcher",
        "call_type": "researcher",
    }
    body.update(overrides)
    return AgentPrefetchRequest(**body)


def test_the_identity_is_the_key_the_real_request_will_present():
    """The join the prefetch exists to serve. `agent_id` carries a graph
    path; the real request sends the bare runtime node. Indexing under the
    path would be worse than not indexing — it looks like attribution and
    joins nothing."""
    identity = _prefetch_identity(make_request())

    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=identity)
    )
    assert node_key_for_request(request) == NodeKey("1", "researcher", "researcher")


def test_agent_id_rides_along_for_attribution():
    """The policy ignores it, but `FileStatLogger`'s per-request CSV keys on
    it, and a phantom indistinguishable from real traffic there makes the
    prefetch arm unauditable after the run."""
    identity = _prefetch_identity(make_request())
    assert identity["agent_id"] == (
        "langgraph:research_supervisor:supervisor_tools:researcher"
    )


def test_a_missing_call_type_still_indexes():
    """`call_type` is optional — `node_key_for_request` substitutes the empty
    string, which is exactly the `use_call_type=False` key."""
    identity = _prefetch_identity(make_request(call_type=None))
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=identity)
    )
    assert node_key_for_request(request) == NodeKey("1", "researcher", "")


def test_a_partial_identity_is_refused_whole():
    """Mirrors `node_key_for_request`, which returns None if either `job_id`
    or `langgraph_node` is missing. Attaching half of it would put fields in
    `extra_args` that read as attribution while indexing nothing."""
    assert _prefetch_identity(make_request(job_id=None)) is None
    assert _prefetch_identity(make_request(langgraph_node=None)) is None
    assert _prefetch_identity(make_request(job_id=None, langgraph_node=None)) is None


def test_the_identity_reaches_the_submitter():
    """The plumbing itself: `_fan_out_prefetches` used to drop it on the
    floor by not passing the argument at all."""
    submitter = RecordingSubmitter()
    identity = _prefetch_identity(make_request())

    asyncio.run(
        _fan_out_prefetches(
            registry=OnePrefixRegistry(),
            submitter=submitter,
            agent_id="langgraph:researcher",
            k=1,
            wait=False,
            identity=identity,
        )
    )

    assert len(submitter.submissions) == 1
    assert submitter.submissions[0]["identity"] == identity


def test_an_absent_identity_is_passed_as_none_not_invented():
    """A caller that sends no identity gets the old unscored behaviour, not a
    guess assembled from `agent_id` — a wrong key is not a safe default."""
    submitter = RecordingSubmitter()

    asyncio.run(
        _fan_out_prefetches(
            registry=OnePrefixRegistry(),
            submitter=submitter,
            agent_id="langgraph:researcher",
            k=1,
            wait=False,
            identity=_prefetch_identity(make_request(job_id=None)),
        )
    )

    assert submitter.submissions[0]["identity"] is None
