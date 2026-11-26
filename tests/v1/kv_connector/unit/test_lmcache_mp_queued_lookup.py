# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A request that waits for GPU room after its LMCache lookup.

pitlane run swe_lite10_q1-2_20261120_110959 (two SWE questions at once) died
on such a request. Q2's second conduct_research call (28,730 tokens) had an
LMCache result up to 27,424 and waited 35 s for GPU blocks while Q1's call
decoded and evicted Q2's cached prefix from 16,560 down to 5,152 tokens. With
lookups starting at vLLM's own hit (the first LMCACHE_MP_LOOKUP_FROM_VLLM_HIT,
c1f0cb269) that shrink took a path that released the lookup's locks on every
scheduling attempt, left stale hit counts behind, and issued a retrieve for a
request the scheduler was prefilling itself; its finished_recving then tripped
the scheduler's assertion.

By default the lookup covers the prompt from token 0, as upstream's does, so a
shrinking hit only means more to load. (LMCACHE_MP_LOOKUP_FROM_VLLM_HIT is back
as an opt-in that looks up again from a shrunken hit; its tests are in
test_lmcache_mp_lookup_from_hit.py.) These tests hold the default down:

- the queued request keeps its locks and loads the difference
  (`test_a_queued_request_whose_hit_shrank_loads_the_difference`);
- the retrieve covers exactly what the scheduler was told to expect
  (`test_the_retrieve_matches_what_the_scheduler_was_told`);
- the scheduler still asserts when finished_recving names a RUNNING request,
  as upstream's does; nothing produces that any more, so it is an expected
  failure that documents the assertion
  (`test_finished_recving_for_a_running_request_does_not_kill_the_engine`).

    pytest tests/v1/kv_connector/unit/test_lmcache_mp_queued_lookup.py -v
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("lmcache")

from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    lmcache_mp_connector as mp,
)
from vllm.v1.request import RequestStatus  # noqa: E402

from .utils import (  # noqa: E402
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

BLOCK = 16

# The crashed request, in tokens.
PROMPT = 28_730
HIT_AT_LOOKUP = 16_560  # vLLM's APC hit when the lookup was submitted
LMCACHE_END = 27_424  # LMCache's result, counted from token 0
HIT_AFTER_WAIT = 5_152  # what was left of the APC hit when blocks came free


class Adapter:
    """Stands in for LMCache's scheduler adapter; records lock releases."""

    def __init__(self, result: int | None):
        self.result = result
        self.lookups: list[dict] = []
        self.freed: list[tuple[int, int]] = []

    def maybe_submit_lookup_request(
        self, request_id, token_ids, cache_salt="", start=0
    ):
        if not self.lookups:  # "maybe": one lookup per request
            self.lookups.append({"start": start, "tokens": len(token_ids)})

    def check_lookup_result(self, request_id):
        return self.result

    def num_blocks_per_chunk(self):
        return 1

    def free_lookup_locks(self, token_ids, start, end, request_id, cache_salt=""):
        self.freed.append((start, end))

    def cleanup_lookup_result(self, request_id):
        pass


def _connector(adapter: Adapter):
    conn = object.__new__(mp.LMCacheMPConnector)
    conn.scheduler_adapter = adapter
    conn.vllm_block_size = BLOCK
    conn.request_trackers = {}
    conn._full_hit_only = False
    conn._backing_reports = False
    return conn


def _request(tokens: int):
    return SimpleNamespace(
        request_id="q2-conduct-research",
        status=RequestStatus.WAITING,
        cache_salt="",
        all_token_ids=list(range(tokens)),
        block_hashes=list(range(tokens // BLOCK)),
        num_prompt_tokens=tokens,
        kv_transfer_params=None,
    )


def _blocks(n: int):
    return SimpleNamespace(get_block_ids=lambda: (list(range(n)),))


def _queue_then_shrink(conn: mp.LMCacheMPConnector, adapter: Adapter, attempts: int):
    """The crashed request's scheduling attempts, up to the one that got blocks.

    Returns what each attempt after the shrink returned.
    """
    req = _request(PROMPT)
    # Attempt 1: submits the lookup, of the whole prompt; LMCache still reading.
    assert conn.get_num_new_matched_tokens(req, HIT_AT_LOOKUP) == (None, True)
    assert adapter.lookups == [{"start": 0, "tokens": PROMPT}]
    # Attempt 2: the lookup is back and the connector offers the load -- but
    # allocate_slots finds no room and the request stays waiting.
    adapter.result = LMCACHE_END
    assert conn.get_num_new_matched_tokens(req, HIT_AT_LOOKUP) == (
        LMCACHE_END - HIT_AT_LOOKUP,
        True,
    )
    # Attempts 3..: a neighbour's decode has evicted the tail of the APC hit.
    returned = [
        conn.get_num_new_matched_tokens(req, HIT_AFTER_WAIT) for _ in range(attempts)
    ]
    return req, returned


def test_a_queued_request_whose_hit_shrank_loads_the_difference():
    adapter = Adapter(result=None)
    conn = _connector(adapter)
    _, returned = _queue_then_shrink(conn, adapter, attempts=5)
    # The lookup locked everything from token 0, so the tokens the GPU lost
    # are still LMCache's to load, and nothing is released while it waits.
    assert returned == [(LMCACHE_END - HIT_AFTER_WAIT, True)] * 5
    assert adapter.freed == []


def test_the_retrieve_matches_what_the_scheduler_was_told():
    adapter = Adapter(result=None)
    conn = _connector(adapter)
    req, returned = _queue_then_shrink(conn, adapter, attempts=1)
    need = LMCACHE_END - HIT_AFTER_WAIT
    assert returned == [(need, True)]
    # Blocks come free: the scheduler allocates the hit plus the load.
    conn.update_state_after_alloc(req, _blocks(-(-PROMPT // BLOCK)), need)
    # Locks on what vLLM itself holds go back; the rest is retrieved.
    assert adapter.freed == [(0, HIT_AFTER_WAIT)]
    metadata = mp.LMCacheMPConnectorMetadata()
    conn._process_retrieve_requests(metadata)
    retrieves = [m for m in metadata.requests if m.direction == "RETRIEVE"]
    assert [(m.op.start, m.op.end) for m in retrieves] == [(HIT_AFTER_WAIT, LMCACHE_END)]


@pytest.mark.xfail(
    strict=True,
    reason="the scheduler (like upstream's) asserts a finished_recving request is "
    "waiting for KV or finished; the connector no longer produces one that is "
    "neither, see the module docstring",
)
def test_finished_recving_for_a_running_request_does_not_kill_the_engine():
    scheduler = create_scheduler(create_vllm_config(kv_load_failure_policy="recompute"))
    request = create_request(num_tokens=100 * scheduler.block_size)
    scheduler.add_request(request=request)

    # The connector offers nothing, so the request is scheduled as a plain
    # prefill and goes RUNNING -- as the crashed request did.
    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (0, False)
    scheduler.connector.take_events.return_value = ()
    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING

    # The stray retrieve's completion arrives with the step's output.
    model_runner_output = create_model_runner_output(
        reqs=[request], finished_recving={request.request_id}
    )
    try:
        scheduler.update_from_output(scheduler_output, model_runner_output)
    except AssertionError as exc:
        pytest.fail(
            "EngineCore would die: finished_recving for a RUNNING request "
            f"tripped an assertion in _update_from_kv_xfer_finished ({exc!r})"
        )
    assert request.status == RequestStatus.RUNNING
    assert request.request_id in scheduler.requests
