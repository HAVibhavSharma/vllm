# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-aware placement between data-parallel engines (VLLM_DP_KV_ROUTING).

    pytest tests/v1/engine/test_kv_aware_routing.py -v
"""

import asyncio
from collections import Counter
from types import SimpleNamespace

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine import kv_aware_routing as kvr
from vllm.v1.engine.core_client import DPLBAsyncMPClient

KV_TOKENS = 110_000  # per engine, as in the pitlane runs
_COUNTER = 0


def _request(tokens, phantom=False, rank=None):
    global _COUNTER
    _COUNTER += 1
    extra = {"kv_transfer_params": {"prefetch_only": True}} if phantom else {}
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
    router.placed(other, 1)  # running on engine 1
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


def test_a_pinned_request_is_remembered_where_it_went():
    client = _dplb_client(_router())
    client.get_core_engine_for_request(_request(_prompt(11, 1_000), rank=0))
    pinned = _request(_prompt(12, 10_000), rank=1)
    assert client.get_core_engine_for_request(pinned) == client.core_engines[1]
    nxt = _request(_prompt(12, 10_000) + _prompt(13, 100))
    assert client.get_core_engine_for_request(nxt) == client.core_engines[1]


def test_without_the_switch_the_balancer_is_unchanged():
    client = _dplb_client(None)
    a = client.get_core_engine_for_request(_request(_prompt(1, 9_000)))
    b = client.get_core_engine_for_request(_request(_prompt(1, 9_000)))
    assert a != b
