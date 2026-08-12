# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The front-end half of prefetch origination (02 §4 option a).

No engine, no event loop of its own beyond `asyncio.run`: the drainer's whole
job is to turn a want dict from engine core into phantom submissions, and
both ends of that are fakeable.
"""

import asyncio
from types import SimpleNamespace

from vllm.v1.agent_prefetch.drain import PrefetchWantDrainer
from vllm.v1.agent_prefetch.registry import AgentPrefixRegistry, PrefixDescriptor
from vllm.v1.agent_prefetch.submitter import PhantomPrefetchSubmitter
from vllm.v1.core.node_eviction.controller import node_key_for_request
from vllm.v1.core.node_eviction.types import NodeKey

WANT = {
    "job_id": "run-42",
    "node": "research",
    "call_type": "tavily:summary",
    "agent_id": "langgraph:research",
    "score": 177.6,
    "prob": 0.99,
    "time_to_next_call_ms": 100.0,
}


class FakeEngineClient:
    """Returns one batch of wants, then nothing — like a real engine whose
    want-list is emptied by the first drain."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = 0

    async def drain_prefetch_wants(self, max_items=4):
        self.calls += 1
        if not self._batches:
            return []
        return self._batches.pop(0)


class RecordingSubmitter:
    def __init__(self, returns_task=True):
        self.submissions = []
        self._returns_task = returns_task

    async def submit(self, **kwargs):
        self.submissions.append(kwargs)
        return object() if self._returns_task else None


def registry_with_prefix(agent_id="langgraph:research", n=1):
    registry = AgentPrefixRegistry()
    for i in range(n):
        registry.record(
            agent_id,
            PrefixDescriptor(
                token_ids=(1, 2, 3, i),
                prefix_hash=b"\xaa" * 8 + bytes([i]),
                cache_salt=f"agent::{agent_id}",
            ),
        )
    return registry


def drain_once(engine_client, registry, submitter, **kwargs):
    drainer = PrefetchWantDrainer(
        engine_client, registry, submitter, **kwargs
    )
    return asyncio.run(drainer.drain_once())


# -- fan-out -------------------------------------------------------------


def test_a_want_fans_out_over_every_prefix_the_agent_has_used():
    """One want is one *node*, and a node may have several recorded prefixes.
    The registry, not the want, decides how many phantoms that is."""
    submitter = RecordingSubmitter()
    submitted = drain_once(
        FakeEngineClient([[WANT]]), registry_with_prefix(n=3), submitter
    )
    assert submitted == 3
    assert len(submitter.submissions) == 3
    assert {s["agent_id"] for s in submitter.submissions} == {"langgraph:research"}


def test_a_want_for_an_unknown_agent_submits_nothing():
    """Engine core can want a node this process has never served. Inventing a
    prefix for it would prefetch the wrong bytes."""
    submitter = RecordingSubmitter()
    submitted = drain_once(
        FakeEngineClient([[WANT]]),
        registry_with_prefix(agent_id="langgraph:writer"),
        submitter,
    )
    assert submitted == 0
    assert submitter.submissions == []


def test_an_engine_without_the_utility_method_is_not_an_error():
    """An older engine, or one where the policy is off, must degrade to no
    prefetching rather than to an exception on a background poll."""
    submitter = RecordingSubmitter()
    submitted = drain_once(SimpleNamespace(), registry_with_prefix(), submitter)
    assert submitted == 0


def test_the_phantom_carries_the_identity_the_policy_predicted():
    """Without this the phantom is anonymous: `node_key_for_request` returns
    None, `on_blocks_cached` returns early, and the blocks it fetched are
    never stamped speculative — leaving the decaying floor (02 §5) inert for
    exactly the blocks it exists to protect."""
    submitter = RecordingSubmitter()
    drain_once(FakeEngineClient([[WANT]]), registry_with_prefix(), submitter)

    identity = submitter.submissions[0]["identity"]
    assert identity == {
        "job_id": "run-42",
        "langgraph_node": "research",
        "call_type": "tavily:summary",
    }


def test_the_identity_round_trips_into_the_key_the_engine_asked_for():
    """The join that closes the loop: the want's key must be the key the
    phantom's blocks are indexed under, or the prefetch protects nothing."""
    captured = {}

    class CapturingEngineClient:
        async def generate(self, prompt, params, request_id):
            captured["params"] = params
            return
            yield  # pragma: no cover - makes this an async generator

    async def scenario():
        submitter = PhantomPrefetchSubmitter(CapturingEngineClient())
        task = await submitter.submit(
            agent_id="langgraph:research",
            token_ids=[1, 2, 3],
            prefix_hash=b"\xaa" * 8,
            cache_salt="agent::langgraph:research",
            identity={
                "job_id": "run-42",
                "langgraph_node": "research",
                "call_type": "tavily:summary",
            },
        )
        assert task is not None
        await task

    asyncio.run(scenario())

    params = captured["params"]
    # The prefetch_only marker must survive alongside the identity, or the
    # scheduler will not finalize the phantom and the index will not stamp it.
    assert params.extra_args["kv_transfer_params"]["prefetch_only"] is True

    request = SimpleNamespace(
        sampling_params=params,
        kv_transfer_params=params.extra_args["kv_transfer_params"],
    )
    assert node_key_for_request(request) == NodeKey(
        "run-42", "research", "tavily:summary"
    )


def test_an_anonymous_phantom_is_invisible_to_the_policy():
    """`submit` without an identity still yields an unscored phantom.

    Not a wish: a caller that cannot name the node the prefix is for has
    nothing safe to send, and inventing a key from `agent_id` would index the
    blocks under a name no real request presents. `/v1/agents/prefetch` now
    supplies the identity when the caller gives it one — see
    `test_prefetch_identity.py` — so this is the deliberate fallback, not the
    endpoint's behaviour."""
    captured = {}

    class CapturingEngineClient:
        async def generate(self, prompt, params, request_id):
            captured["params"] = params
            return
            yield  # pragma: no cover - makes this an async generator

    async def scenario():
        submitter = PhantomPrefetchSubmitter(CapturingEngineClient())
        task = await submitter.submit(
            agent_id="langgraph:research",
            token_ids=[1, 2, 3],
            prefix_hash=b"\xaa" * 8,
            cache_salt="agent::langgraph:research",
        )
        await task

    asyncio.run(scenario())

    request = SimpleNamespace(
        sampling_params=captured["params"], kv_transfer_params=None
    )
    assert node_key_for_request(request) is None


# -- resilience ----------------------------------------------------------


def test_a_failing_poll_does_not_kill_the_loop():
    """A drain that raises mid-shutdown must not silently stop prefetching
    for the rest of the process — that looks identical to a forecast that
    predicted nothing."""

    class ExplodingEngineClient:
        def __init__(self):
            self.calls = 0

        async def drain_prefetch_wants(self, max_items=4):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("engine busy")
            return [WANT]

    engine = ExplodingEngineClient()
    submitter = RecordingSubmitter()

    async def scenario():
        drainer = PrefetchWantDrainer(
            engine, registry_with_prefix(), submitter, interval_s=0.05
        )
        drainer.start()
        await asyncio.sleep(0.25)
        await drainer.stop()
        return drainer.stats()

    stats = asyncio.run(scenario())
    assert stats["errors"] >= 1
    assert engine.calls >= 2
    assert submitter.submissions, "the loop stopped after the first failure"


def test_stop_is_idempotent_and_safe_before_start():
    drainer = PrefetchWantDrainer(
        SimpleNamespace(), registry_with_prefix(), RecordingSubmitter()
    )
    asyncio.run(drainer.stop())
    asyncio.run(drainer.stop())
