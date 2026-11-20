# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A request that waits for GPU room after its LMCache lookup.

Reproduces the engine crash in pitlane run swe_lite10_q1-2_20261120_110959
(two SWE questions at once). Q2's second conduct_research call (28,730 tokens)
was looked up from its 16,560-token APC hit; LMCache found the rest of its
prefix up to 27,424 on disk. It then waited 35 s for GPU blocks while Q1's
call decoded, and Q1's decode evicted Q2's cached prefix down to 5,152 tokens.
What followed, and what each test pins down:

1. Every scheduling attempt after the hit shrank took the "hit shrank below
   the lookup's start" path and released the lookup's locks again: 1,170
   releases in 35 s, the first one evicting the disk-loaded chunks from L1.
   (`test_a_queued_request_releases_its_lookup_locks_once`)
2. That path returns (0, False) but leaves the hit counts an earlier attempt
   recorded. When blocks were finally allocated the request went RUNNING as a
   plain prefill, yet `update_state_after_alloc` read the stale counts, set
   WAITING_FOR_LOAD and the connector issued a retrieve LMCache could no
   longer serve (retrieved_count=0).
   (`test_a_request_scheduled_without_a_load_is_not_retrieved`)
3. That retrieve's completion came back in finished_recving for a RUNNING
   request, and `_update_from_kv_xfer_finished` asserted it was finished: the
   EngineCore died.
   (`test_finished_recving_for_a_running_request_does_not_kill_the_engine`)

Each test states the correct behaviour, so on the code that crashed all three
fail; that failure is the confirmation.

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

    supports_lookup_start = True

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
    conn._lookup_from_vllm_hit = True
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
    # Attempt 1: submits the lookup from the APC hit; LMCache still reading.
    assert conn.get_num_new_matched_tokens(req, HIT_AT_LOOKUP) == (None, True)
    assert adapter.lookups == [{"start": HIT_AT_LOOKUP, "tokens": PROMPT}]
    # Attempt 2: the lookup is back and the hit is intact, so the connector
    # offers the load -- but allocate_slots finds no room and the request
    # stays waiting. (update_state_after_alloc is not called.)
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


def test_a_queued_request_releases_its_lookup_locks_once():
    adapter = Adapter(result=None)
    conn = _connector(adapter)
    _, returned = _queue_then_shrink(conn, adapter, attempts=5)
    assert returned == [(0, False)] * 5
    # The locks on [lookup start, LMCache end) are released once. Every
    # release after the first is of chunks LMCache no longer has (the run
    # logged 795k "finish read on non-existing key" warnings).
    assert adapter.freed == [(HIT_AT_LOOKUP, LMCACHE_END)], (
        f"released {len(adapter.freed)} times: {adapter.freed[:3]}..."
    )


def test_a_request_scheduled_without_a_load_is_not_retrieved():
    adapter = Adapter(result=None)
    conn = _connector(adapter)
    req, returned = _queue_then_shrink(conn, adapter, attempts=1)
    assert returned == [(0, False)]
    # Blocks come free: the scheduler takes (0, False) at its word and
    # allocates the whole prompt as a plain prefill, no external tokens.
    conn.update_state_after_alloc(req, _blocks(-(-PROMPT // BLOCK)), 0)
    tracker = conn.request_trackers[req.request_id]
    state = tracker.state.name  # before the next step's metadata moves it on
    metadata = mp.LMCacheMPConnectorMetadata()
    conn._process_retrieve_requests(metadata)
    retrieves = [m for m in metadata.requests if m.direction == "RETRIEVE"]
    assert retrieves == [], (
        "a retrieve was issued for a request vLLM is prefilling itself: the "
        f"tracker went {state} from stale hit counts (vllm "
        f"{tracker.num_vllm_hit_blocks} blocks, lmcache "
        f"{tracker.num_lmcache_hit_blocks} blocks)"
    )
    assert state == mp.LMCacheMPRequestState.READY.name


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
