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
    for table in (dp_pin._rank_of_job, dp_pin._last_call_at, dp_pin._seed_placed,
                  dp_pin._ended_jobs):
        table.clear()
    yield
    for table in (dp_pin._rank_of_job, dp_pin._last_call_at, dp_pin._seed_placed,
                  dp_pin._ended_jobs):
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
    # one that reaches the server after the release is dropped too
    assert _place(_client({}), SEED, time.time()) == (None, "its question ended")
    # and a job named again takes seeds again
    dp_pin.set_job_rank("7", 1)
    assert _place(_client({}), SEED, time.time()) == (1, None)


def test_only_seeds_and_only_with_pinning(monkeypatch):
    assert _place(_client({}), "chatcmpl-abc") == (None, None)
    monkeypatch.setenv("VLLM_AGENT_DP_PIN", "0")
    assert _place(_client({})) == (None, None)
    dp_pin.note_call("langgraph:7:g:target")
    assert dp_pin._last_call_at == {}


def test_releasing_a_question_aborts_its_running_seeds_only():
    client = _client({"chatcmpl-prefetch::langgraph:7:g:target::seed-1-ab12cd34": 1,
                      "chatcmpl-prefetch::langgraph:8:g:target::seed-2-ef56ab78": 0,
                      "chatcmpl-a-11112222": 0})
    calls = []

    async def abort(ids, internal=False):
        calls.append((list(ids), internal))

    client.abort = abort
    dp_pin.set_job_rank("7", 1)
    aborted = asyncio.run(dp_pin.end_job(client, "7"))
    assert dp_pin.job_ended("7") and not dp_pin.job_ended("8")
    assert aborted == ["chatcmpl-prefetch::langgraph:7:g:target::seed-1-ab12cd34"]
    assert calls == [(aborted, True)]
    assert dp_pin.job_rank("7") is None and "7" in dp_pin._ended_jobs
    # nothing running: no abort call
    calls.clear()
    assert asyncio.run(dp_pin.end_job(client, "9")) == [] and calls == []


def test_a_seed_that_reaches_the_engine_after_the_end_is_aborted(monkeypatch):
    monkeypatch.setattr(dp_pin, "SWEEP_S", (0.01, 0.05))
    client = _client({})
    calls = []

    async def abort(ids, internal=False):
        # as the engine does once it reports them finished
        calls.append(list(ids))
        for rid in ids:
            client.engine_core.reqs_in_flight.pop(rid, None)

    client.abort = abort
    dp_pin.set_job_rank("7", 0)

    async def run():
        assert await dp_pin.end_job(client, "7") == []
        # placed just before the end, it reaches the engine a moment later
        client.engine_core.reqs_in_flight["chatcmpl-prefetch::langgraph:7:g:t::seed-3-aa11bb22"] = b"e0"
        await asyncio.sleep(0.2)

    asyncio.run(run())
    assert calls == [["chatcmpl-prefetch::langgraph:7:g:t::seed-3-aa11bb22"]]


def _unpinned_kv_routing(monkeypatch):
    monkeypatch.setenv("VLLM_AGENT_DP_PIN", "0")
    monkeypatch.setenv("VLLM_DP_KV_ROUTING", "1")


def test_unpinned_kv_routing_places_seeds_too(monkeypatch):
    """Without pinning, KV-aware routing keeps arm C's seed rules: a seed
    waits for an idle engine, a call marks its seeds stale, an ended
    question's seeds are refused."""
    _unpinned_kv_routing(monkeypatch)
    assert _place(_client({"chatcmpl-a": 0})) == (1, None)
    dp_pin.note_call("langgraph:7:g:target")
    assert "langgraph:7:g:target" in dp_pin._last_call_at
    dp_pin.set_job_rank("9", -1)
    assert dp_pin.job_ended("9")
    assert dp_pin.phantom_rank("9") is None
    monkeypatch.setenv("VLLM_DP_KV_ROUTING", "0")
    dp_pin._seed_placed.clear()
    assert _place(_client({})) == (None, None)


def test_unpinned_both_idle_prefers_the_engine_holding_its_prefix(monkeypatch):
    """So the seed's KV is where the router sends its call."""
    from vllm.v1.engine.kv_aware_routing import KVAwareRouter

    _unpinned_kv_routing(monkeypatch)
    router = KVAwareRouter(2, lambda: 110_000)
    prompt = list(range(10_000))
    prev = SimpleNamespace(
        request_id="prev", prompt_token_ids=prompt[:9_000], sampling_params=None
    )
    router.placed(prev, 1)
    router.finished("prev")
    client = _client({})
    client.engine_core.kv_router = router
    assert asyncio.run(dp_pin.place_seed(client, SEED, None, prompt)) == (1, None)
    dp_pin._seed_placed.clear()
    other = list(range(50_000, 51_000))
    assert asyncio.run(dp_pin.place_seed(client, SEED, None, other)) == (0, None)
