# SPDX-License-Identifier: Apache-2.0
"""A question keeps one data-parallel engine; a seed runs only on an idle one."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from vllm.v1.agent_prefetch import dp_pin

SEED = "chatcmpl-prefetch::langgraph:7:g:target::seed-1"


@pytest.fixture(autouse=True)
def _pinning(monkeypatch):
    monkeypatch.setenv("VLLM_AGENT_DP_PIN", "1")
    monkeypatch.setattr(dp_pin, "POLL_S", 0.01)
    for table in (dp_pin._rank_of_job, dp_pin._last_call_at, dp_pin._seed_placed):
        table.clear()
    yield
    for table in (dp_pin._rank_of_job, dp_pin._last_call_at, dp_pin._seed_placed):
        table.clear()


def _client(in_flight):
    engines = [b"e0", b"e1"]
    return SimpleNamespace(engine_core=SimpleNamespace(
        core_engines=engines,
        reqs_in_flight={rid: engines[i] for rid, i in in_flight.items()},
    ))


def _place(client, request_id=SEED, issued_at=None):
    return asyncio.run(dp_pin.place_seed(client, request_id, issued_at))


def test_ids():
    assert dp_pin.job_of("langgraph:5:swe_architect:conduct_research") == "5"
    assert dp_pin.job_of("chatcmpl-prefetch::langgraph:10:x:y::seed-ab") == "10"
    assert dp_pin.job_of("prefetch::langgraph:*:**:node::h") is None
    assert dp_pin.job_of("chatcmpl-abc") is None
    assert dp_pin.seed_agent_id(SEED) == "langgraph:7:g:target"
    assert dp_pin.seed_agent_id("chatcmpl-abc") is None


def test_phantoms_follow_their_job_and_only_when_pinning_is_on(monkeypatch):
    dp_pin.set_job_rank("3", 1)
    assert dp_pin.phantom_rank("3") == 1
    assert dp_pin.phantom_rank("4") is None
    dp_pin.set_job_rank("3", -1)
    assert dp_pin.phantom_rank("3") is None
    dp_pin.set_job_rank("3", 1)
    monkeypatch.setenv("VLLM_AGENT_DP_PIN", "0")
    assert dp_pin.phantom_rank("3") is None


def test_seed_goes_to_the_idle_engine_phantoms_do_not_count():
    client = _client({"chatcmpl-a": 0, "prefetch::langgraph:1:n::h::l1": 1})
    assert _place(client) == (1, None)


def test_both_idle_prefers_its_own_questions_engine():
    dp_pin.set_job_rank("7", 1)
    assert _place(_client({})) == (1, None)
    dp_pin._seed_placed.clear()
    assert _place(_client({}), "chatcmpl-prefetch::langgraph:8:g:n::seed-2") == (0, None)


def test_one_seed_per_engine():
    client = _client({"chatcmpl-a": 0, "chatcmpl-prefetch::x::seed-0": 1})
    dp_pin.set_job_rank("7", 0)

    async def run():
        task = asyncio.create_task(dp_pin.place_seed(client, SEED, None))
        await asyncio.sleep(0.05)
        assert not task.done()  # engine 0 has a call, engine 1 a seed
        del client.engine_core.reqs_in_flight["chatcmpl-prefetch::x::seed-0"]
        return await asyncio.wait_for(task, 1.0)

    assert asyncio.run(run()) == (1, None)
    # placed but not yet in flight: the next seed does not join it
    client.engine_core.reqs_in_flight.clear()
    assert _place(client, "chatcmpl-prefetch::langgraph:9:g:n::seed-3") == (0, None)


def test_waits_while_busy_then_takes_the_engine_that_frees():
    client = _client({"chatcmpl-a": 0, "chatcmpl-b": 1})
    dp_pin.set_job_rank("7", 0)

    async def run():
        task = asyncio.create_task(dp_pin.place_seed(client, SEED, None))
        await asyncio.sleep(0.05)
        assert not task.done()
        del client.engine_core.reqs_in_flight["chatcmpl-b"]
        return await asyncio.wait_for(task, 1.0)

    assert asyncio.run(run()) == (1, None)


def test_dropped_when_its_call_arrives_or_its_question_ends():
    issued = time.time()
    dp_pin.set_job_rank("7", 0)
    dp_pin.note_call("langgraph:7:g:target")
    assert _place(_client({"chatcmpl-a": 0, "chatcmpl-b": 1}), SEED, issued) == (
        None, "its call arrived")
    # a call before the seed was issued does not make it stale
    assert _place(_client({}), SEED, time.time())[0] == 0
    dp_pin._seed_placed.clear()

    client = _client({"chatcmpl-a": 0, "chatcmpl-b": 1})

    async def run():
        task = asyncio.create_task(dp_pin.place_seed(client, SEED, time.time()))
        await asyncio.sleep(0.05)
        dp_pin.set_job_rank("7", -1)
        return await asyncio.wait_for(task, 1.0)

    assert asyncio.run(run()) == (None, "its question ended")


def test_only_seeds_and_only_with_pinning(monkeypatch):
    assert _place(_client({}), "chatcmpl-abc") == (None, None)
    monkeypatch.setenv("VLLM_AGENT_DP_PIN", "0")
    assert _place(_client({})) == (None, None)
    dp_pin.note_call("langgraph:7:g:target")
    assert dp_pin._last_call_at == {}
