# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-aware placement between data-parallel engines (VLLM_DP_KV_ROUTING).

    pytest tests/v1/engine/test_kv_aware_routing.py -v
"""

import asyncio
from collections import Counter
from types import SimpleNamespace

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine import kv_aware_routing as kvr
from vllm.v1.engine.core_client import DPLBAsyncMPClient

KV_TOKENS = 110_000  # per engine, as in the pitlane runs
_COUNTER = 0


def _request(tokens, phantom=False, rank=None, key=None, staged=False):
    global _COUNTER
    _COUNTER += 1
    extra: dict = {}
    if phantom:
        extra["kv_transfer_params"] = {"prefetch_only": True, "stage_l1": staged}
    if key is not None:
        extra["job_id"], extra["agent_id"] = key
    return SimpleNamespace(
        request_id=f"req-{_COUNTER}",
        prompt_token_ids=list(tokens),
        sampling_params=SimpleNamespace(extra_args=extra),
        data_parallel_rank=rank,
        pooling_params=None,
    )


def _prompt(seed: int, n: int) -> list[int]:
    return [(seed * 1_000_003 + i) % 50_000 for i in range(n)]


def _router(engines=2, capacity=KV_TOKENS, **kw):
    return kvr.KVAwareRouter(engines, lambda: capacity, **kw)


def _place(router, request, counts=None, fallback=0):
    counts = counts or [[0, 0, 0.0] for _ in range(router.num_engines)]
    engine, hashes = router.choose(request, counts, 1, fallback)
    router.placed(request, engine, hashes)
    return engine


def test_a_call_follows_its_prefix_on_a_tie():
    """Q4's last call in arm A: its 28k-token prefix on engine 1, one call
    on each engine, and the balancer's tie pick engine 0."""
    router = _router()
    prev = _request(_prompt(4, 28_863))
    router.placed(prev, 1)
    router.finished(prev.request_id)
    other = _request(_prompt(9, 15_242))
    router.placed(other, 1)  # decoding on engine 1
    router.produced(other.request_id)
    counts = [[0, 1, 0.3], [0, 1, 0.2]]  # a call running on each
    nxt = _request(_prompt(4, 28_863) + _prompt(44, 13_561))
    engine, _ = router.choose(nxt, counts, 1, fallback=0)
    assert engine == 1


def test_an_idle_engine_beats_the_prefix_on_a_busier_one():
    """The prefix reloads from CPU memory in a fraction of a second; sharing
    a GPU slows the call's whole decode."""
    router = _router()
    prev = _request(_prompt(1, 20_000))
    router.placed(prev, 1)
    router.finished(prev.request_id)
    counts = [[0, 0, 0.0], [0, 1, 0.4]]
    nxt = _request(_prompt(1, 20_000) + _prompt(2, 3_000))
    assert router.choose(nxt, counts, 1, fallback=0)[0] == 0
    # With a slack of one call, the prefix's engine may be one call busier.
    router = _router(slack=1)
    router.placed(prev, 1)
    router.finished(prev.request_id)
    assert router.choose(nxt, counts, 1, fallback=0)[0] == 1
    counts = [[0, 0, 0.0], [1, 1, 0.4]]
    assert router.choose(nxt, counts, 1, fallback=0)[0] == 0


def test_a_shared_system_prompt_alone_does_not_pull():
    router = _router(min_tokens=4096)
    system = _prompt(7, 1_024)
    router.placed(_request(system + _prompt(10, 20_000)), 1)
    other_question = _request(system + _prompt(11, 20_000))
    counts = [[0, 0, 0.0], [0, 0, 0.0]]
    assert router.choose(other_question, counts, 1, fallback=0)[0] == 0


def test_phantoms_are_not_load_and_a_call_follows_its_phantom():
    router = _router()
    for i in range(5):  # staged phantoms in flight on engine 0
        _place(router, _request(_prompt(100 + i, 9_000), phantom=True), fallback=0)
    assert router.calls == [0, 0]
    warm = _request(_prompt(3, 12_000), phantom=True)
    assert _place(router, warm, fallback=0) == 0
    call = _request(_prompt(3, 12_000) + _prompt(30, 500))
    counts = [[0, 0, 0.0], [0, 0, 0.0]]
    assert router.choose(call, counts, 1, fallback=1)[0] == 0


def test_least_loaded_without_a_prefix_keeps_the_balancers_pick():
    router = _router()
    req = _request(_prompt(5, 8_000))
    counts = [[0, 1, 0.0], [0, 1, 0.0]]
    assert router.choose(req, counts, 1, fallback=1)[0] == 1
    counts = [[0, 2, 0.0], [0, 1, 0.0]]
    assert router.choose(req, counts, 1, fallback=0)[0] == 1


def test_an_engine_forgets_what_its_cache_cannot_hold():
    router = _router(capacity=40_000)
    old = _request(_prompt(1, 30_000))
    router.placed(old, 1)
    router.placed(_request(_prompt(2, 30_000)), 1)
    hashes = kvr.chunk_hashes(old.prompt_token_ids)
    # Its tail goes first, as in vLLM's cache; its head is still held.
    assert 0 < router.matched_tokens(1, hashes) < 30_000
    router.placed(_request(_prompt(3, 30_000)), 1)
    assert router.matched_tokens(1, hashes) == 0
    assert len(router.prefixes[1]) <= 40_000 // kvr.CHUNK_TOKENS


def test_finished_calls_leave_the_load():
    router = _router()
    req = _request(_prompt(6, 5_000))
    engine = _place(router, req)
    assert router.calls[engine] == 1
    router.finished(req.request_id)
    router.finished(req.request_id)
    assert router.calls == [0, 0]


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(kvr.ENABLE_ENV, raising=False)
    assert kvr.KVAwareRouter.from_env(2, lambda: KV_TOKENS) is None
    monkeypatch.setenv(kvr.ENABLE_ENV, "1")
    monkeypatch.setenv(kvr.SLACK_ENV, "2")
    router = kvr.KVAwareRouter.from_env(2, lambda: KV_TOKENS)
    assert router is not None and router.slack == 2


def _dplb_client(kv_router):
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.engine_inflight = Counter()
    client.core_engines = [bytes([i, 0]) for i in range(2)]
    client.lb_engines = [[0, 0, 0.0] for _ in range(2)]
    client.eng_start_index = 0
    client.kv_router = kv_router
    return client


def test_the_load_balancer_keeps_a_question_on_its_engine():
    """Q4's case through the balancer: a call on each engine, the balancer's
    tie pick engine 0, the question's prefix on engine 1."""
    client = _dplb_client(_router())
    zeros = [[0, 0, 0.0], [0, 0, 0.0]]
    client.get_core_engine_for_request(_request(_prompt(9, 1_000)))
    first = _request(_prompt(8, 20_000))
    engine = client.get_core_engine_for_request(first)
    assert engine == client.core_engines[1]
    asyncio.run(
        DPLBAsyncMPClient.process_engine_outputs(
            client, EngineCoreOutputs(finished_requests={first.request_id})
        )
    )
    client.lb_engines = zeros
    assert client.get_core_engine_for_request(_request(_prompt(10, 1_000))) == engine
    nxt = _request(_prompt(8, 20_000) + _prompt(80, 2_000))
    client.lb_engines = [list(c) for c in zeros]
    assert client.get_core_engine_for_request(nxt) == engine
    assert client.kv_router.calls == [1, 2]


def _first_token(client, request):
    asyncio.run(
        DPLBAsyncMPClient.process_engine_outputs(
            client,
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(request_id=request.request_id, new_token_ids=[1])
                ]
            ),
        )
    )


def test_a_pinned_request_is_remembered_where_it_went():
    client = _dplb_client(_router())
    client.get_core_engine_for_request(_request(_prompt(11, 1_000), rank=0))
    pinned = _request(_prompt(12, 10_000), rank=1)
    assert client.get_core_engine_for_request(pinned) == client.core_engines[1]
    _first_token(client, pinned)
    nxt = _request(_prompt(12, 10_000) + _prompt(13, 100))
    assert client.get_core_engine_for_request(nxt) == client.core_engines[1]


def test_without_the_switch_the_balancer_is_unchanged():
    client = _dplb_client(None)
    a = client.get_core_engine_for_request(_request(_prompt(1, 9_000)))
    b = client.get_core_engine_for_request(_request(_prompt(1, 9_000)))
    assert a != b


Q4 = ("4", "langgraph:4:swe_architect:conduct_research")


def test_a_call_releases_its_nodes_phantoms_on_other_engines():
    """A staged phantom ends only when its call reaches its engine; one left
    on the other engine would hold L1 read locks until its question ends."""
    router = _router()
    away = _request(_prompt(20, 9_000), phantom=True, key=Q4, staged=True)
    here = _request(_prompt(20, 9_000), phantom=True, key=Q4)
    other_node = _request(
        _prompt(21, 9_000), phantom=True, key=("4", "langgraph:4:x:y"), staged=True
    )
    router.placed(away, 0)
    router.placed(here, 1)
    router.placed(other_node, 0)
    call = _request(_prompt(20, 9_000) + _prompt(22, 300), key=Q4)
    router.placed(call, 1)
    assert router.take_stranded() == [away.request_id]
    assert router.take_stranded() == []
    assert set(router.phantoms) == {here.request_id, other_node.request_id}


def test_a_staged_phantom_is_not_taken_for_gpu_kv():
    """It reads into LMCache's L1 only; the GPU holding the previous call's
    prefix still gets the call."""
    router = _router()
    prev = _request(_prompt(30, 20_000), key=Q4)
    router.placed(prev, 1)
    router.produced(prev.request_id)
    router.finished(prev.request_id)
    staged = _request(_prompt(30, 20_000) + _prompt(31, 6_000), phantom=True,
                      key=Q4, staged=True)
    router.placed(staged, 0)
    assert router.matched_tokens(0, kvr.chunk_hashes(staged.prompt_token_ids)) == 0
    call = _request(_prompt(30, 20_000) + _prompt(31, 6_200), key=Q4)
    counts = [[0, 0, 0.0], [0, 0, 0.0]]
    assert router.choose(call, counts, 1, fallback=0)[0] == 1


def test_a_tie_avoids_an_engine_in_a_long_prefill():
    """Q4's case the other way round: the prefix's engine is prefilling 24k
    new tokens, the other is decoding."""
    router = _router()
    old = _request(_prompt(40, 28_000))
    router.placed(old, 0)
    router.produced(old.request_id)
    router.finished(old.request_id)
    long_prefill = _request(_prompt(41, 24_000))
    router.placed(long_prefill, 0)
    decoding = _request(_prompt(42, 15_000))
    router.placed(decoding, 1)
    router.produced(decoding.request_id)
    call = _request(_prompt(40, 28_000) + _prompt(43, 14_000))
    counts = [[0, 1, 0.3], [0, 1, 0.2]]
    assert router.choose(call, counts, 1, fallback=0)[0] == 1
    router.produced(long_prefill.request_id)
    assert router.choose(call, counts, 1, fallback=1)[0] == 0


def test_a_tie_avoids_an_engine_without_room():
    router = _router()
    old = _request(_prompt(50, 20_000))
    router.placed(old, 0)
    router.produced(old.request_id)
    router.finished(old.request_id)
    call = _request(_prompt(50, 20_000) + _prompt(51, 12_000))
    full = [[0, 0, 0.95], [0, 0, 0.10]]  # 5.5k tokens free on engine 0
    assert router.choose(call, full, 1, fallback=0)[0] == 1
    roomy = [[0, 0, 0.50], [0, 0, 0.10]]
    assert router.choose(call, roomy, 1, fallback=1)[0] == 0


def test_the_balancer_hands_stranded_phantoms_over_once():
    client = _dplb_client(_router())
    phantom = _request(_prompt(60, 9_000), phantom=True, key=Q4, staged=True, rank=0)
    client.get_core_engine_for_request(phantom)
    client.get_core_engine_for_request(_request(_prompt(61, 1_000), rank=0))
    call = _request(_prompt(60, 9_000) + _prompt(62, 500), key=Q4)
    assert client.get_core_engine_for_request(call) == client.core_engines[1]
    assert client.take_stranded_phantoms() == [phantom.request_id]
    assert client.take_stranded_phantoms() == []
    assert client.kv_router.prefill_ahead(1) > 0
    _first_token(client, call)
    assert client.kv_router.prefill_ahead(1) == 0
