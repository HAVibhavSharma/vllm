# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LMCache lookups that start at vLLM's own prefix-cache hit
(LMCACHE_MP_LOOKUP_FROM_VLLM_HIT).

A request asks LMCache only about the chunks from the one holding the end of
its APC hit, so LMCache does not look up, lock or read the blocks vLLM already
holds. If the hit shrinks while the request waits for GPU blocks, the request
looks up again from the new hit and releases the first lookup's locks only
once the second is back, so it loads what the GPU lost. The numbers are the
request that crashed pitlane run swe_lite10_q1-2_20261120_110959 when this was
done without the second lookup (see test_lmcache_mp_queued_lookup.py).

    pytest tests/v1/kv_connector/unit/test_lmcache_mp_lookup_from_hit.py -v
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    lmcache_mp_connector as mp,
)
from vllm.v1.request import RequestStatus  # noqa: E402

BLOCK = 16
PROMPT = 28_730
HIT = 16_560  # vLLM's APC hit when the lookup is submitted
END = 27_424  # LMCache's hit, counted from token 0
SHRUNK = 5_152  # the APC hit after a neighbour's decode evicted part of it


class Adapter:
    """LMCache's scheduler adapter, one lookup per request at a time."""

    supports_lookup_start = True

    def __init__(self):
        self.lookups: list[dict] = []
        self.results: list[int | None] = []
        self.freed: list[tuple[int, int]] = []
        self._pending = False

    def maybe_submit_lookup_request(
        self, request_id, token_ids, cache_salt="", start=0
    ):
        if self._pending:
            return
        self._pending = True
        self.lookups.append({"start": start, "tokens": len(token_ids)})
        self.results.append(None)

    def check_lookup_result(self, request_id):
        return self.results[-1] if self.results else 0

    def cleanup_lookup_result(self, request_id):
        self._pending = False

    def num_blocks_per_chunk(self):
        return 1

    def free_lookup_locks(self, token_ids, start, end, request_id, cache_salt=""):
        self.freed.append((start, end))

    def answer(self, tokens: int):
        self.results[-1] = tokens


def _connector(adapter: Adapter, on: bool = True):
    conn = object.__new__(mp.LMCacheMPConnector)
    conn.scheduler_adapter = adapter
    conn.vllm_block_size = BLOCK
    conn.request_trackers = {}
    conn._full_hit_only = False
    conn._backing_reports = False
    conn._lookup_from_vllm_hit = on
    return conn


def _request():
    return SimpleNamespace(
        request_id="q2-conduct-research",
        status=RequestStatus.WAITING,
        cache_salt="",
        all_token_ids=list(range(PROMPT)),
        block_hashes=list(range(PROMPT // BLOCK)),
        num_prompt_tokens=PROMPT,
        kv_transfer_params=None,
    )


def _allocate(conn, req, need):
    block_ids = list(range(-(-PROMPT // BLOCK)))
    blocks = SimpleNamespace(get_block_ids=lambda: (block_ids,))
    conn.update_state_after_alloc(req, blocks, need)
    metadata = mp.LMCacheMPConnectorMetadata()
    conn._process_retrieve_requests(metadata)
    return [(m.op.start, m.op.end) for m in metadata.requests if m.direction == "RETRIEVE"]


def _looked_up_then_shrunk(conn, adapter, req, second: int):
    """Lookup at HIT answered with END, offered once, then the hit shrinks and
    the second lookup answers `second`. Returns the first call after it."""
    assert conn.get_num_new_matched_tokens(req, HIT) == (None, True)
    adapter.answer(END)
    assert conn.get_num_new_matched_tokens(req, HIT) == (END - HIT, True)
    assert conn.get_num_new_matched_tokens(req, SHRUNK) == (None, True)
    assert adapter.lookups[-1] == {"start": SHRUNK, "tokens": PROMPT}
    assert adapter.freed == []  # the first lookup keeps its locks meanwhile
    adapter.answer(second)
    return conn.get_num_new_matched_tokens(req, SHRUNK)


def test_the_lookup_starts_at_vllms_hit():
    adapter = Adapter()
    conn = _connector(adapter)
    req = _request()
    assert conn.get_num_new_matched_tokens(req, HIT) == (None, True)
    assert adapter.lookups == [{"start": HIT, "tokens": PROMPT}]
    adapter.answer(END)
    assert conn.get_num_new_matched_tokens(req, HIT) == (END - HIT, True)
    assert _allocate(conn, req, END - HIT) == [(HIT, END)]
    assert adapter.freed == []  # nothing before the lookup's start was locked


def test_a_shrunken_hit_looks_up_again_and_loads_what_the_gpu_lost():
    adapter = Adapter()
    conn = _connector(adapter)
    req = _request()
    assert _looked_up_then_shrunk(conn, adapter, req, END) == (END - SHRUNK, True)
    assert adapter.freed == [(HIT, END)]  # the second lookup holds them now
    for _ in range(3):
        assert conn.get_num_new_matched_tokens(req, SHRUNK) == (END - SHRUNK, True)
    assert len(adapter.lookups) == 2
    assert _allocate(conn, req, END - SHRUNK) == [(SHRUNK, END)]
    assert adapter.freed == [(HIT, END)]


def test_a_partial_gap_loads_up_to_its_first_missing_chunk():
    adapter = Adapter()
    conn = _connector(adapter)
    req = _request()
    part = 9_984
    assert _looked_up_then_shrunk(conn, adapter, req, part) == (part - SHRUNK, True)
    assert _allocate(conn, req, part - SHRUNK) == [(SHRUNK, part)]


def test_a_gap_lmcache_lacks_loads_nothing_and_never_retrieves():
    """The crash: stale hit counts made the scheduler's prefill come with a
    retrieve. Here the counts are the reported ones, so there is none."""
    adapter = Adapter()
    conn = _connector(adapter)
    req = _request()
    assert _looked_up_then_shrunk(conn, adapter, req, SHRUNK) == (0, False)
    assert _allocate(conn, req, 0) == []
    assert adapter.freed == [(HIT, END)]


def test_a_hit_that_shrinks_before_the_first_lookup_is_back():
    adapter = Adapter()
    conn = _connector(adapter)
    req = _request()
    assert conn.get_num_new_matched_tokens(req, HIT) == (None, True)
    assert conn.get_num_new_matched_tokens(req, SHRUNK) == (None, True)
    assert len(adapter.lookups) == 1
    adapter.answer(END)
    assert conn.get_num_new_matched_tokens(req, SHRUNK) == (None, True)
    assert adapter.lookups[-1]["start"] == SHRUNK
    adapter.answer(END)
    assert conn.get_num_new_matched_tokens(req, SHRUNK) == (END - SHRUNK, True)
    assert adapter.freed == [(HIT, END)]


def test_a_grown_hit_frees_the_locks_on_what_vllm_now_holds():
    adapter = Adapter()
    conn = _connector(adapter)
    req = _request()
    assert conn.get_num_new_matched_tokens(req, SHRUNK) == (None, True)
    adapter.answer(END)
    assert conn.get_num_new_matched_tokens(req, HIT) == (END - HIT, True)
    assert len(adapter.lookups) == 1
    assert _allocate(conn, req, END - HIT) == [(HIT, END)]
    assert adapter.freed == [(SHRUNK, HIT)]


def test_a_phantom_that_never_loads_frees_what_its_lookup_holds():
    adapter = Adapter()
    conn = _connector(adapter)
    req = _request()
    conn.get_num_new_matched_tokens(req, HIT)
    adapter.answer(END)
    conn.get_num_new_matched_tokens(req, HIT)
    conn.release_lookup(req)
    assert adapter.freed == [(HIT, END)]


def test_off_the_lookup_starts_at_token_0(monkeypatch):
    monkeypatch.delenv(mp.LOOKUP_FROM_VLLM_HIT_ENV_VAR, raising=False)
    assert mp._lookup_from_vllm_hit_enabled() is False
    monkeypatch.setenv(mp.LOOKUP_FROM_VLLM_HIT_ENV_VAR, "1")
    assert mp._lookup_from_vllm_hit_enabled() is True
    adapter = Adapter()
    conn = _connector(adapter, on=False)
    req = _request()
    conn.get_num_new_matched_tokens(req, HIT)
    assert adapter.lookups == [{"start": 0, "tokens": PROMPT}]
    adapter.answer(END)
    assert conn.get_num_new_matched_tokens(req, SHRUNK) == (END - SHRUNK, True)
    assert len(adapter.lookups) == 1
