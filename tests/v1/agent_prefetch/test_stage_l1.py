# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`stage_l1` rides from `/v1/agents/prefetch` to the scheduler.

langgraph's lookahead from a ReAct node's first turn asks for it: the phantom
reads the prefix from LMCache's L2 into L1 only (`Scheduler._stage_phantoms`),
and the same node's ordinary warm, sent as the tool loop ends, releases it.
That warm carries the same prefix, so the staged phantom needs an id of its
own or the submitter would skip the warm as already in flight.
"""

import asyncio
from types import SimpleNamespace

from vllm.entrypoints.openai.agent_chat.api_router import _fan_out_prefetches
from vllm.entrypoints.openai.agent_chat.protocol import AgentPrefetchRequest
from vllm.v1.agent_prefetch.submitter import PhantomPrefetchSubmitter

AGENT = "langgraph:2:swe_architect:extract_implementation_plan"


class RecordingSubmitter:
    def __init__(self) -> None:
        self.submissions: list[dict] = []

    async def submit(self, **kwargs):
        self.submissions.append(kwargs)
        return None


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


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def generate(self, prompt, params, request_id, priority=0):
        self.calls.append((request_id, params.extra_args))
        if False:
            yield None


def test_off_unless_asked():
    assert AgentPrefetchRequest(agent_id=AGENT).stage_l1 is False


def test_the_flag_reaches_the_submitter():
    for stage in (True, False):
        submitter = RecordingSubmitter()
        asyncio.run(
            _fan_out_prefetches(
                registry=OnePrefixRegistry(),
                submitter=submitter,
                agent_id=AGENT,
                k=1,
                wait=False,
                stage_l1=stage,
            )
        )
        assert submitter.submissions[0]["stage_l1"] is stage


def test_a_staged_phantom_has_its_own_id_and_says_so():
    engine = RecordingEngine()
    submitter = PhantomPrefetchSubmitter(engine)

    async def both():
        staged = await submitter.submit(
            agent_id=AGENT,
            token_ids=[1, 2, 3],
            prefix_hash=b"\xaa" * 8,
            cache_salt="agent::x",
            stage_l1=True,
        )
        # The same prefix's ordinary warm while the staged one is in flight:
        # not a duplicate.
        warm = await submitter.submit(
            agent_id=AGENT,
            token_ids=[1, 2, 3],
            prefix_hash=b"\xaa" * 8,
            cache_salt="agent::x",
        )
        await asyncio.gather(staged, warm)
        return staged, warm

    staged, warm = asyncio.run(both())
    assert staged is not None and warm is not None
    (staged_id, staged_args), (warm_id, warm_args) = engine.calls
    assert staged_id.endswith("::l1") and staged_id[: -len("::l1")] == warm_id
    assert staged_args["kv_transfer_params"]["stage_l1"] is True
    assert "stage_l1" not in warm_args["kv_transfer_params"]


class BlockingEngine:
    """A phantom stays in flight until aborted, as a staged one does."""

    def __init__(self) -> None:
        self.aborted: list[str] = []
        self._events: dict[str, asyncio.Event] = {}

    async def generate(self, prompt, params, request_id, priority=0):
        event = self._events.setdefault(request_id, asyncio.Event())
        await event.wait()
        if False:
            yield None

    async def abort(self, request_ids):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        self.aborted += ids
        for rid in ids:
            self._events.setdefault(rid, asyncio.Event()).set()


def test_release_job_aborts_only_that_jobs_staged_phantoms():
    async def run():
        engine = BlockingEngine()
        submitter = PhantomPrefetchSubmitter(engine)

        async def submit(job, node, prefix, stage):
            return await submitter.submit(
                agent_id=f"langgraph:{job}:swe_architect:{node}",
                token_ids=[1, 2, 3], prefix_hash=prefix, cache_salt="s",
                identity={"job_id": job}, stage_l1=stage,
            )

        await submit("1", "conduct_research", b"\x01" * 8, True)
        await submit("1", "extract_implementation_plan", b"\x02" * 8, False)
        await submit("10", "conduct_research", b"\x03" * 8, True)

        async def settle():
            for _ in range(10):
                await asyncio.sleep(0)

        await settle()
        released = await submitter.release_job("1")
        await settle()
        assert released == ["prefetch::langgraph:1:swe_architect:conduct_research::0101010101010101::l1"]
        assert engine.aborted == released, "job 10's and the GPU warm stay"
        everything = await submitter.release_job("1", staged_only=False)
        assert everything == ["prefetch::langgraph:1:swe_architect:extract_implementation_plan::0202020202020202"]
        await settle()
        await submitter.release_job("10")
        await settle()
        assert submitter.inflight_size() == 0

    asyncio.run(run())
