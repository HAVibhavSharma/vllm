# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A waiting request that holds KV blocks is scheduled before one that cannot fit.

pitlane run swe_lite10_q1-2_20261120_123326 ran two SWE questions at once and
scheduled nothing for 6 min 43 s. Q2's first call arrived 60 ms before Q1's,
but Q1's LMCache lookup finished first, so Q1 got its blocks (59.7% of the
cache) and loaded asynchronously. Q2, older and so at the head of the priority
queue, then could not fit beside Q1; its failed allocation stopped the waiting
scan every step, Q1 -- loaded, holding its blocks, needing one more block --
was never reached, and neither ever ran.

    pytest tests/v1/kv_connector/unit/test_kv_holder_scheduling.py -v
"""

from unittest.mock import Mock

import pytest

from vllm.v1.request import RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

PROMPT_BLOCKS = 60  # each request alone; two do not fit in NUM_BLOCKS
NUM_BLOCKS = 100


@pytest.mark.parametrize("policy", ["priority", "fcfs"])
def test_a_loaded_request_is_not_stuck_behind_one_that_cannot_fit(policy: str):
    vllm_config = create_vllm_config()
    vllm_config.scheduler_config.policy = policy
    scheduler = create_scheduler(vllm_config, num_blocks=NUM_BLOCKS)
    block = scheduler.block_size
    prompt = PROMPT_BLOCKS * block
    external = prompt - block  # the last block is computed after the load

    older = create_request(num_tokens=prompt)  # Q2: arrives first
    loaded = create_request(num_tokens=prompt)  # Q1: lookup finishes first
    scheduler.add_request(older)
    scheduler.add_request(loaded)

    # LMCache lookup results: None while a lookup is still running.
    matched: dict[str, int | None] = {older.request_id: None, loaded.request_id: external}

    def get_num_new_matched_tokens(request, _num_computed_tokens):
        value = matched[request.request_id]
        if value is None:
            return None, True
        return value, value > 0

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        get_num_new_matched_tokens
    )
    scheduler.connector.take_events.return_value = ()

    # Step 1: the older request's lookup is still running, so it is skipped;
    # the other gets its blocks and starts loading.
    output = scheduler.schedule()
    assert loaded.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert older.status == RequestStatus.WAITING

    # Its load finishes, and the older request's lookup comes back too.
    scheduler.update_from_output(
        output,
        create_model_runner_output(reqs=[], finished_recving={loaded.request_id}),
    )
    matched[older.request_id] = external

    # Step 2: the older request cannot fit beside the loaded one. The loaded
    # one must still run -- it needs one block, and it is what will free room.
    output = scheduler.schedule()
    assert loaded.request_id in output.num_scheduled_tokens, (
        "the loaded request was not scheduled: it sits behind a request whose "
        f"allocation fails ({loaded.status.name}, older {older.status.name})"
    )
    assert loaded.status == RequestStatus.RUNNING
    assert older.request_id not in output.num_scheduled_tokens
    # Nothing was lost from the waiting queues.
    waiting = list(scheduler.waiting) + list(scheduler.skipped_waiting)
    assert older in waiting and loaded not in waiting
