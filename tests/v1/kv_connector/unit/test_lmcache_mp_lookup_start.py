# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LMCacheMPConnector asks LMCache only about what vLLM does not already hold.

The lookup starts at the chunk holding the end of vLLM's own prefix-cache hit
(LMCACHE_MP_LOOKUP_FROM_VLLM_HIT); LMCache's result still counts from token 0.
A stub scheduler adapter stands in for the LMCache server.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    lmcache_mp_connector as mp,
)

BLOCK = 16


class Adapter:
    """Records lookups and lock releases; answers with a fixed result."""

    supports_lookup_start = True

    def __init__(self, result: int | None):
        self.result = result
        self.lookups: list[dict] = []
        self.freed: list[tuple[int, int]] = []

    def maybe_submit_lookup_request(
        self, request_id, token_ids, cache_salt="", start=0
    ):
        self.lookups.append({"start": start, "tokens": len(token_ids)})

    def check_lookup_result(self, request_id):
        return self.result

    def num_blocks_per_chunk(self):
        return 1

    def free_lookup_locks(self, token_ids, start, end, request_id, cache_salt=""):
        self.freed.append((start, end))

    def cleanup_lookup_result(self, request_id):
        pass


class OldAdapter(Adapter):
    """An LMCache adapter from before lookups could start past 0."""

    supports_lookup_start = False

    def maybe_submit_lookup_request(self, request_id, token_ids, cache_salt=""):
        self.lookups.append({"start": None, "tokens": len(token_ids)})


def _connector(adapter: Adapter, enabled: bool = True):
    conn = object.__new__(mp.LMCacheMPConnector)
    conn.scheduler_adapter = adapter
    conn.vllm_block_size = BLOCK
    conn.request_trackers = {}
    conn._full_hit_only = False
    conn._backing_reports = False
    conn._lookup_from_vllm_hit = enabled and adapter.supports_lookup_start
    return conn


def _request(tokens: int):
    return SimpleNamespace(
        request_id="r",
        status=None,
        cache_salt="",
        all_token_ids=list(range(tokens)),
        block_hashes=list(range(tokens // BLOCK)),
        num_prompt_tokens=tokens,
        kv_transfer_params=None,
    )


def _blocks(n: int):
    return SimpleNamespace(get_block_ids=lambda: (list(range(n)),))


def test_lookup_starts_at_vllm_hit_and_loads_only_the_rest():
    # extract (#4) in run 235002: 26,608 of 28,465 tokens already in HBM.
    adapter = Adapter(result=28_464)  # counted from 0
    conn = _connector(adapter)
    req = _request(28_465)
    assert conn.get_num_new_matched_tokens(req, 26_608) == (28_464 - 26_608, True)
    assert adapter.lookups == [{"start": 26_608, "tokens": 28_465}]

    # Retrieve from the lookup's start: nothing before it was locked, so
    # nothing before it is released.
    conn.update_state_after_alloc(req, _blocks(1_780), 28_464 - 26_608)
    assert adapter.freed == []


def test_nothing_past_the_hit_is_a_miss_with_no_locks_to_release():
    adapter = Adapter(result=39_808)  # LMCache adds nothing to the HBM hit
    conn = _connector(adapter)
    req = _request(39_993)
    assert conn.get_num_new_matched_tokens(req, 39_808) == (0, False)
    conn.update_state_after_alloc(req, _blocks(2_500), 0)
    assert adapter.freed == []


def test_an_unaligned_hit_starts_at_its_chunk():
    adapter = Adapter(result=None)
    conn = _connector(adapter)
    adapter_chunk = 4  # blocks per LMCache chunk
    adapter.num_blocks_per_chunk = lambda: adapter_chunk
    conn.get_num_new_matched_tokens(_request(4_000), 5 * BLOCK)
    assert adapter.lookups[0]["start"] == 4 * BLOCK


def test_the_start_is_fixed_by_the_first_call():
    adapter = Adapter(result=None)
    conn = _connector(adapter)
    req = _request(4_000)
    assert conn.get_num_new_matched_tokens(req, 1_600) == (None, True)
    adapter.result = 3_200
    conn.get_num_new_matched_tokens(req, 1_920)  # APC grew while it waited
    assert [lookup["start"] for lookup in adapter.lookups] == [1_600, 1_600]
    # Locks from the lookup's start up to the hit vLLM now holds go back.
    conn.update_state_after_alloc(req, _blocks(250), 3_200 - 1_920)
    assert adapter.freed == [(1_600, 1_920)]


def test_a_hit_that_shrank_below_the_start_loads_nothing():
    adapter = Adapter(result=None)
    conn = _connector(adapter)
    req = _request(4_000)
    conn.get_num_new_matched_tokens(req, 1_600)
    adapter.result = 3_200
    assert conn.get_num_new_matched_tokens(req, 800) == (0, False)
    assert adapter.freed == [(1_600, 3_200)]


def test_turned_off_it_looks_up_from_zero():
    adapter = Adapter(result=28_464)
    conn = _connector(adapter, enabled=False)
    req = _request(28_465)
    conn.get_num_new_matched_tokens(req, 26_608)
    assert adapter.lookups == [{"start": 0, "tokens": 28_465}]
    conn.update_state_after_alloc(req, _blocks(1_780), 28_464 - 26_608)
    assert adapter.freed == [(0, 26_608)]


def test_an_older_lmcache_adapter_keeps_the_old_lookup():
    adapter = OldAdapter(result=28_464)
    conn = _connector(adapter)
    assert conn._lookup_from_vllm_hit is False
    conn.get_num_new_matched_tokens(_request(28_465), 26_608)
    assert adapter.lookups == [{"start": None, "tokens": 28_465}]
