# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The phantom queue and basic eviction (plan/new-eviction/13-phantom-queue.md).

A phantom is a `prefetch_only` request. With VLLM_PHANTOM_QUEUE=1 it waits
in its own queue and is admitted one at a time, only into a step where no
real request is waiting, the phantom slot is free and memory stays spare.
Basic eviction then protects what it warmed until there is contention.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.agent_prefetch.submitter import _PHANTOM_PRIORITY, _phantom_priority
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

BLOCK_SIZE = 16


# What langgraph's `derive_agent_id` names a node's calls: the real request
# and the phantom warming it carry the same one, beside the job id.
NODE = "langgraph:1:swe_architect:conduct_research"
OTHER_NODE = "langgraph:1:swe_architect:extract_implementation_plan"


def _make(
    index: int,
    req_id: str,
    num_tokens: int,
    *,
    phantom: bool = False,
    agent: str | None = None,
    job: str = "1",
    staged: bool = False,
) -> Request:
    """Request `index` of a batch, so its prompt is `[index] * num_tokens`.

    Two requests made with the same index share a prompt prefix; different
    indices never do. With `agent`, it carries `(job, agent)` in its extra
    args, the key a phantom is matched to its request by. A `staged` phantom
    asks for its L2 read into L1 only (`stage_l1`).
    """
    request = create_requests(
        num_requests=index + 1,
        num_tokens=num_tokens,
        max_tokens=1 if phantom or staged else 16,
        block_size=BLOCK_SIZE,
        req_ids=[f"_{j}" for j in range(index)] + [req_id],
    )[index]
    if phantom or staged:
        request.kv_transfer_params = {"prefetch_only": True, "cache_salt": "agent::x"}
    if staged:
        request.kv_transfer_params["stage_l1"] = True
    if agent is not None:
        request.sampling_params.extra_args = {"job_id": job, "agent_id": agent}
    return request


def _scheduler(monkeypatch, **kwargs):
    monkeypatch.setenv("VLLM_PHANTOM_QUEUE", "1")
    # A partial LMCache hit, loaded synchronously: an admitted phantom then
    # shows up in the step's scheduled tokens.
    return create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=BLOCK_SIZE * 2, is_async=False),
        block_size=BLOCK_SIZE,
        **kwargs,
    )


def _finished(ecos, req_id: str) -> list:
    return [
        out
        for engine_outputs in ecos.values()
        for out in engine_outputs.outputs
        if out.request_id == req_id
    ]


def test_with_the_queue_off_a_phantom_shares_the_waiting_queue():
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=BLOCK_SIZE * 2, is_async=False),
        block_size=BLOCK_SIZE,
    )
    phantom = _make(1, "phantom", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(phantom)
    assert not scheduler.phantom_waiting
    assert phantom in scheduler.waiting


def test_a_phantom_waits_while_a_real_request_is_waiting(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    phantom = _make(1, "phantom", BLOCK_SIZE * 4, phantom=True)
    real = _make(2, "real", BLOCK_SIZE * 4)
    scheduler.add_request(phantom)
    scheduler.add_request(real)
    assert scheduler.phantom_waiting == [phantom]
    assert phantom not in scheduler.waiting

    first = scheduler.schedule()
    assert "real" in first.num_scheduled_tokens
    assert "phantom" not in first.num_scheduled_tokens
    assert scheduler.phantom_waiting == [phantom]
    assert scheduler._phantom_stats["held_real_waiting"] == 1

    # The real request is running now, so nothing real is waiting.
    second = scheduler.schedule()
    assert "phantom" in second.num_scheduled_tokens
    assert not scheduler.phantom_waiting
    assert scheduler._phantom_stats["admitted"] == 1


def test_one_phantom_at_a_time_oldest_first(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    older = _make(1, "older", BLOCK_SIZE * 4, phantom=True)
    newer = _make(2, "newer", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(older)
    scheduler.add_request(newer)

    first = scheduler.schedule()
    assert "older" in first.num_scheduled_tokens
    assert "newer" not in first.num_scheduled_tokens
    assert scheduler.phantom_waiting == [newer]

    # `older` still holds its blocks, so the slot is taken.
    second = scheduler.schedule()
    assert "newer" not in second.num_scheduled_tokens
    assert scheduler.phantom_waiting == [newer]
    assert scheduler._phantom_stats["held_slot"] >= 1


def test_an_unadmitted_phantom_goes_to_the_back_of_the_line(monkeypatch):
    # A lookup that has not resolved: the admitted phantom gets no blocks this
    # step and comes back behind the one that arrived after it.
    scheduler = _scheduler(monkeypatch)
    scheduler.connector.get_num_new_matched_tokens = lambda *a, **k: (None, False)
    older = _make(1, "older", BLOCK_SIZE * 4, phantom=True)
    newer = _make(2, "newer", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(older)
    scheduler.add_request(newer)

    output = scheduler.schedule()
    assert "older" not in output.num_scheduled_tokens
    assert scheduler.phantom_waiting == [newer, older]


def test_a_phantom_never_takes_a_real_requests_slot(monkeypatch):
    scheduler = _scheduler(monkeypatch, max_num_seqs=1)
    real = _make(1, "real", BLOCK_SIZE * 4)
    phantom = _make(2, "phantom", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(real)
    scheduler.schedule()
    scheduler.add_request(phantom)

    output = scheduler.schedule()
    assert "phantom" not in output.num_scheduled_tokens
    assert scheduler._phantom_stats["held_slot"] == 1


def test_a_phantom_leaves_memory_for_live_requests(monkeypatch):
    # 23 usable blocks. The real request takes about 5, leaving about 18;
    # the reserve for it is 4.
    scheduler = _scheduler(monkeypatch, num_blocks=24)
    real = _make(1, "real", BLOCK_SIZE * 4)
    scheduler.add_request(real)
    scheduler.schedule()

    big = _make(2, "big", BLOCK_SIZE * 16, phantom=True)
    scheduler.add_request(big)
    held = scheduler.schedule()
    assert "big" not in held.num_scheduled_tokens
    assert scheduler._phantom_stats["held_memory"] == 1

    # `big` is older but still does not fit, so it must not hold up `small`.
    small = _make(3, "small", BLOCK_SIZE * 6, phantom=True)
    scheduler.add_request(small)
    admitted = scheduler.schedule()
    assert "small" in admitted.num_scheduled_tokens
    assert scheduler.phantom_waiting == [big]


def test_an_expired_phantom_is_finished_unrun(monkeypatch):
    monkeypatch.setenv("VLLM_PHANTOM_QUEUE_TIMEOUT_S", "-1")
    scheduler = _scheduler(monkeypatch)
    phantom = _make(1, "phantom", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(phantom)

    output = scheduler.schedule()
    assert "phantom" not in output.num_scheduled_tokens
    ecos = scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)

    assert scheduler._phantom_stats["expired"] == 1
    assert phantom.status == RequestStatus.FINISHED_STOPPED
    finished = _finished(ecos, "phantom")
    assert len(finished) == 1
    assert finished[0].finish_reason == FinishReason.STOP


def _ran(req_id: str) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_a_real_request_overtakes_its_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    real = _make(0, "real", BLOCK_SIZE * 8, agent=NODE)
    scheduler.add_request(phantom)
    scheduler.add_request(real)
    assert not scheduler.phantom_waiting
    assert scheduler._phantom_stats["dropped_on_arrival"] == 1

    output = scheduler.schedule()
    assert "real" in output.num_scheduled_tokens
    # `real` was scheduled, so the step's output has to carry it.
    ecos = scheduler.update_from_output(output, _ran("real"))
    assert phantom.status == RequestStatus.FINISHED_STOPPED
    assert len(_finished(ecos, "phantom")) == 1


def test_another_nodes_request_does_not_drop_a_phantom(monkeypatch):
    # The same prompt, but another node's call: matched by key, not prompt.
    scheduler = _scheduler(monkeypatch)
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    scheduler.add_request(_make(0, "real", BLOCK_SIZE * 8, agent=OTHER_NODE))
    assert scheduler.phantom_waiting == [phantom]


def test_another_questions_request_does_not_drop_a_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    scheduler.add_request(_make(0, "real", BLOCK_SIZE * 8, agent=NODE, job="2"))
    assert scheduler.phantom_waiting == [phantom]


def test_its_request_with_another_prompt_keeps_a_queued_phantom(monkeypatch):
    # The key matches but the phantom's prefix is not the request's: kept,
    # and counted (the predictor built another prefix than the call has).
    scheduler = _scheduler(monkeypatch)
    phantom = _make(1, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    scheduler.add_request(_make(2, "real", BLOCK_SIZE * 8, agent=NODE))
    assert scheduler.phantom_waiting == [phantom]
    assert scheduler._phantom_stats["dropped_on_arrival"] == 0
    assert scheduler._phantom_stats["key_prompt_mismatch"] == 1


def test_a_phantom_after_its_request_with_another_prompt_is_queued(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_make(2, "real", BLOCK_SIZE * 8, agent=NODE))
    phantom = _make(1, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    assert scheduler.phantom_waiting == [phantom]
    assert scheduler._phantom_stats["dropped_after_request"] == 0
    assert scheduler._phantom_stats["key_prompt_mismatch"] == 1


def test_a_phantom_shorter_than_a_block_is_kept(monkeypatch):
    # No full block of prompt, so nothing to confirm the key's match by.
    scheduler = _scheduler(monkeypatch)
    phantom = _make(0, "phantom", BLOCK_SIZE - 1, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    scheduler.add_request(_make(0, "real", BLOCK_SIZE * 8, agent=NODE))
    assert scheduler.phantom_waiting == [phantom]
    assert scheduler._phantom_stats["dropped_on_arrival"] == 0
    assert scheduler._phantom_stats["key_prompt_mismatch"] == 1


def test_without_a_key_nothing_is_dropped(monkeypatch):
    # No ids, or a seed's pattern id: no call to match, whatever the prompt.
    # Queued as usual, and logged.
    scheduler = _scheduler(monkeypatch)
    queued = _make(0, "queued", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(queued)
    scheduler.add_request(_make(0, "real", BLOCK_SIZE * 8))
    seed = _make(0, "seed", BLOCK_SIZE * 4, phantom=True,
                 agent="langgraph:*:**:conduct_research")
    scheduler.add_request(seed)
    assert scheduler.phantom_waiting == [queued, seed]
    assert scheduler._phantom_stats["dropped_on_arrival"] == 0
    assert scheduler._phantom_stats["dropped_after_request"] == 0
    assert scheduler._phantom_stats["no_key"] == 2, "both logged and counted"


def test_a_phantom_after_its_waiting_request_is_never_queued(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    real = _make(0, "real", BLOCK_SIZE * 8, agent=NODE)
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(real)
    scheduler.add_request(phantom)  # came after its request
    assert not scheduler.phantom_waiting
    assert scheduler._phantom_stats["dropped_after_request"] == 1
    assert scheduler._phantom_stats["dropped_on_arrival"] == 0

    output = scheduler.schedule()
    assert "phantom" not in output.num_scheduled_tokens
    ecos = scheduler.update_from_output(output, _ran("real"))
    assert phantom.status == RequestStatus.FINISHED_STOPPED
    assert len(_finished(ecos, "phantom")) == 1


def test_a_phantom_after_its_running_request_is_never_queued(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    real = _make(0, "real", BLOCK_SIZE * 8, agent=NODE)
    scheduler.add_request(real)
    scheduler.update_from_output(scheduler.schedule(), _ran("real"))
    assert real.status == RequestStatus.RUNNING

    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    assert not scheduler.phantom_waiting
    assert scheduler._phantom_stats["dropped_after_request"] == 1


def test_a_phantom_after_its_request_finished_is_queued(monkeypatch):
    # It may be warming the node's next call: kept.
    scheduler = _scheduler(monkeypatch)
    real = _make(0, "real", BLOCK_SIZE * 8, agent=NODE)
    scheduler.add_request(real)
    scheduler.finish_requests("real", RequestStatus.FINISHED_ABORTED)
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    assert scheduler.phantom_waiting == [phantom]
    assert scheduler._phantom_stats["dropped_after_request"] == 0


def test_a_phantom_after_another_nodes_request_is_queued(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_make(0, "real", BLOCK_SIZE * 8, agent=OTHER_NODE))
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    assert scheduler.phantom_waiting == [phantom]


def test_a_phantom_that_started_before_its_request_runs_on(monkeypatch):
    # Admitted and loading when its request arrives: it runs as before.
    scheduler = _scheduler(monkeypatch)
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    output = scheduler.schedule()
    assert "phantom" in output.num_scheduled_tokens

    scheduler.add_request(_make(0, "real", BLOCK_SIZE * 8, agent=NODE))
    assert not phantom.is_finished()
    assert scheduler._phantom_stats["dropped_after_request"] == 0
    assert scheduler._phantom_stats["dropped_on_arrival"] == 0
    assert phantom not in scheduler._phantoms_to_drop


def _copying_scheduler(monkeypatch, **env):
    """LMCache loads asynchronously: an admitted phantom sits in
    WAITING_FOR_REMOTE_KVS, copying its prefix into the GPU, until the worker
    reports it received. Prefix caching on, so its blocks can be hit after."""
    monkeypatch.setenv("VLLM_PHANTOM_QUEUE", "1")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=BLOCK_SIZE * 2, is_async=True),
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
    )


def _received(*req_ids: str) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=KVConnectorOutput(finished_recving=set(req_ids)),
    )


def _phantom_copying(scheduler) -> Request:
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True, agent=NODE)
    scheduler.add_request(phantom)
    output = scheduler.schedule()
    assert phantom.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)
    return phantom


def test_a_request_waits_for_its_phantoms_copy_and_hits_its_blocks(monkeypatch):
    scheduler = _copying_scheduler(monkeypatch)
    phantom = _phantom_copying(scheduler)

    real = _make(0, "real", BLOCK_SIZE * 8, agent=NODE)
    scheduler.add_request(real)
    output = scheduler.schedule()
    assert real.status == RequestStatus.WAITING, "held back while the phantom copies"
    assert real.prefill_stats is not None and real.prefill_stats.num_prompt_tokens == 0
    assert scheduler._phantom_stats["waited_for_phantom_copy"] == 1

    scheduler.update_from_output(output, _received("phantom"))  # copy done
    assert phantom.status == RequestStatus.FINISHED_STOPPED

    scheduler.schedule()
    assert real.prefill_stats is not None
    assert real.prefill_stats.num_local_cached_tokens == BLOCK_SIZE * 2, (
        "the phantom's blocks, not a copy of its own"
    )
    assert scheduler._phantom_stats["waited_for_phantom_copy"] == 1


def test_another_nodes_request_does_not_wait_for_a_phantoms_copy(monkeypatch):
    scheduler = _copying_scheduler(monkeypatch)
    _phantom_copying(scheduler)
    real = _make(0, "real", BLOCK_SIZE * 8, agent=OTHER_NODE)
    scheduler.add_request(real)
    scheduler.schedule()
    assert real.status == RequestStatus.WAITING_FOR_REMOTE_KVS, "loads its own"
    assert scheduler._phantom_stats["waited_for_phantom_copy"] == 0


def test_a_request_stops_waiting_for_a_copy_that_never_ends(monkeypatch):
    scheduler = _copying_scheduler(monkeypatch, VLLM_PHANTOM_COPY_WAIT_S="-1")
    _phantom_copying(scheduler)
    real = _make(0, "real", BLOCK_SIZE * 8, agent=NODE)
    scheduler.add_request(real)
    scheduler.update_from_output(scheduler.schedule(), EMPTY_MODEL_RUNNER_OUTPUT)
    assert real.status == RequestStatus.WAITING  # waited one step
    scheduler.schedule()
    assert real.status == RequestStatus.WAITING_FOR_REMOTE_KVS, "loads its own"
    assert scheduler._phantom_stats["phantom_copy_wait_timeout"] == 1


def test_the_counters_reach_the_stats(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_make(1, "phantom", BLOCK_SIZE * 4, phantom=True))
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.phantom_queue["queued"] == 1
    assert stats.phantom_queue["real_blocked_with_phantom"] == 0


def test_a_loaded_phantom_reports_what_it_loaded():
    """A phantom finalized after its LMCache load never has a scheduled step,
    which is the only other place prefill stats are handed on; without them
    its stats row read 0 cached and 0 loaded."""
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=BLOCK_SIZE * 2, is_async=True),
        block_size=BLOCK_SIZE,
    )
    phantom = _make(1, "phantom", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(phantom)

    output = scheduler.schedule()
    assert phantom.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)

    output = scheduler.schedule()
    loaded = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=KVConnectorOutput(finished_recving={"phantom"}),
    )
    ecos = scheduler.update_from_output(output, loaded)

    finished = _finished(ecos, "phantom")
    assert len(finished) == 1
    stats = finished[0].prefill_stats
    assert stats is not None
    assert stats.num_external_cached_tokens == BLOCK_SIZE * 2
    assert stats.num_local_cached_tokens == 0


# -- basic eviction: the block pool --------------------------------------


def _pool(num_blocks: int) -> tuple[BlockPool, list]:
    """A pool whose usable blocks are all cached and free, oldest first."""
    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=True,
                     hash_block_size=BLOCK_SIZE)
    pool.warm_protect_s = 10.0
    blocks = pool.get_new_blocks(num_blocks - 1)
    for i, block in enumerate(blocks):
        block.block_hash = make_block_hash_with_group_id(BlockHash(bytes([i]) * 32), 0)
    pool.free_blocks(blocks)
    return pool, blocks


def test_warmed_blocks_are_evicted_after_every_ordinary_block():
    pool, blocks = _pool(8)
    # The two oldest: plain LRU would evict exactly these first.
    warmed = {blocks[0].block_id, blocks[1].block_id}
    assert pool.protect_warm(warmed) == 2

    taken = pool.get_new_blocks(5)
    assert warmed.isdisjoint(b.block_id for b in taken)

    # Contention: only the warmed blocks are left, and a real need takes them.
    last = pool.get_new_blocks(2)
    assert {b.block_id for b in last} == warmed
    assert pool.warm_stats["warm_taken_under_contention"] == 2


def test_a_hit_ends_a_warmed_blocks_protection():
    pool, blocks = _pool(4)
    pool.protect_warm([blocks[0].block_id])
    pool.touch([blocks[0]])
    assert pool.warm_stats["warm_used"] == 1
    assert not pool._warm_until


def test_an_unused_warmed_block_becomes_ordinary_when_its_time_is_up():
    pool, blocks = _pool(4)
    pool.protect_warm([blocks[0].block_id])
    pool._warm_until[blocks[0].block_id] = 0.0  # long past
    taken = pool.get_new_blocks(1)
    assert taken[0].block_id == blocks[0].block_id
    assert pool.warm_stats["warm_expired"] == 1


def test_only_free_cached_blocks_are_protected():
    pool, blocks = _pool(4)
    in_use = pool.get_new_blocks(1)[0]
    assert pool.protect_warm([in_use.block_id]) == 0


# -- configuration ---------------------------------------------------------


def test_basic_mode_turns_the_splice_off(monkeypatch):
    monkeypatch.setenv("VLLM_NODE_EVICTION_BASIC", "1")
    monkeypatch.setenv("VLLM_NODE_EVICTION_WARM_PROTECT_S", "7")
    cfg = NodeEvictionConfig.from_env()
    assert cfg.basic and cfg.enabled and not cfg.observe_only
    assert cfg.splice_max_blocks == 0
    assert cfg.warm_protect_s == 7.0


def test_basic_mode_and_observe_mode_are_exclusive(monkeypatch):
    monkeypatch.setenv("VLLM_NODE_EVICTION_BASIC", "1")
    monkeypatch.setenv("VLLM_NODE_EVICTION_OBSERVE", "1")
    with pytest.raises(ValueError, match="exclusive"):
        NodeEvictionConfig.from_env()


def test_phantoms_go_last_under_the_priority_policy():
    scheduler_config = SimpleNamespace(policy="priority")
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(scheduler_config=scheduler_config)
    )
    assert _phantom_priority(client) == _PHANTOM_PRIORITY
    scheduler_config.policy = "fcfs"
    assert _phantom_priority(client) == 0


# -- Staged phantoms (`stage_l1`) --------------------------------------------
#
# The lookahead's warm of the node after a tool loop, sent at the loop's first
# turn: only its LMCache lookup runs (the L2 read into L1), it takes no GPU
# blocks, and it ends when that node's GPU warm or call arrives.


def test_a_staged_phantom_only_looks_up_and_takes_no_blocks(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    pool = scheduler.kv_cache_manager.block_pool
    free = pool.get_num_free_blocks()
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True, agent=OTHER_NODE)
    scheduler.add_request(staged)
    assert scheduler.phantom_staged == [staged]
    assert not scheduler.phantom_waiting

    output = scheduler.schedule()
    assert "staged" not in output.num_scheduled_tokens
    assert scheduler.phantom_staged == [staged], "it waits, holding L1"
    assert pool.get_num_free_blocks() == free
    assert scheduler._phantom_stats["staged"] == 1
    assert scheduler._phantom_stats["stage_loaded"] == 1
    assert scheduler._phantom_stats["admitted"] == 0


def test_a_staged_phantom_waits_for_its_lookup(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    calls: list[int] = []

    def pending(request, num_computed_tokens):
        calls.append(num_computed_tokens)
        return None, True  # LMCache is still reading it into L1

    scheduler.connector.get_num_new_matched_tokens = pending
    scheduler.add_request(_make(1, "staged", BLOCK_SIZE * 4, staged=True))
    scheduler.schedule()
    scheduler.schedule()
    assert calls == [0, 0], "polled each step, looked up from token 0"
    assert scheduler._phantom_stats["stage_loaded"] == 0


def test_its_nodes_gpu_warm_releases_a_staged_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    released: list[str] = []
    scheduler.connector.release_lookup = lambda r: released.append(r.request_id)
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True, agent=OTHER_NODE)
    scheduler.add_request(staged)
    scheduler.schedule()

    # The loop ended: the same node's ordinary warm, the GPU copy from L1.
    warm = _make(1, "warm", BLOCK_SIZE * 4, phantom=True, agent=OTHER_NODE)
    scheduler.add_request(warm)
    assert not scheduler.phantom_staged
    assert scheduler.phantom_waiting == [warm]
    assert scheduler._phantom_stats["stage_released"] == 1

    output = scheduler.schedule()
    ecos = scheduler.update_from_output(output, _ran("warm"))
    assert staged.status == RequestStatus.FINISHED_STOPPED
    assert len(_finished(ecos, "staged")) == 1
    assert released == ["staged"], "its L1 locks are freed"


def test_another_nodes_warm_leaves_a_staged_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True, agent=OTHER_NODE)
    scheduler.add_request(staged)
    scheduler.add_request(_make(2, "warm", BLOCK_SIZE * 4, phantom=True, agent=NODE))
    assert scheduler.phantom_staged == [staged]
    assert scheduler._phantom_stats["stage_released"] == 0


def test_its_call_drops_a_staged_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True, agent=OTHER_NODE)
    scheduler.add_request(staged)
    scheduler.add_request(_make(1, "real", BLOCK_SIZE * 8, agent=OTHER_NODE))
    assert not scheduler.phantom_staged
    assert scheduler._phantom_stats["dropped_on_arrival"] == 1


def test_its_call_with_another_prompt_keeps_a_staged_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True, agent=OTHER_NODE)
    scheduler.add_request(staged)
    scheduler.add_request(_make(2, "real", BLOCK_SIZE * 8, agent=OTHER_NODE))
    assert scheduler.phantom_staged == [staged]
    assert scheduler._phantom_stats["key_prompt_mismatch"] == 1


def test_a_staged_phantom_after_its_request_is_never_staged(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_make(1, "real", BLOCK_SIZE * 8, agent=OTHER_NODE))
    scheduler.add_request(
        _make(1, "staged", BLOCK_SIZE * 4, staged=True, agent=OTHER_NODE)
    )
    assert not scheduler.phantom_staged
    assert scheduler._phantom_stats["dropped_after_request"] == 1


def test_a_staged_phantom_expires(monkeypatch):
    monkeypatch.setenv("VLLM_PHANTOM_STAGE_TIMEOUT_S", "-1")
    scheduler = _scheduler(monkeypatch)
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True)
    scheduler.add_request(staged)
    output = scheduler.schedule()
    ecos = scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert scheduler._phantom_stats["stage_expired"] == 1
    assert staged.status == RequestStatus.FINISHED_STOPPED
    assert len(_finished(ecos, "staged")) == 1


def test_an_aborted_staged_phantom_frees_its_lookup(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    released: list[str] = []
    scheduler.connector.release_lookup = lambda r: released.append(r.request_id)
    scheduler.add_request(_make(1, "staged", BLOCK_SIZE * 4, staged=True))
    scheduler.schedule()
    scheduler.finish_requests("staged", RequestStatus.FINISHED_ABORTED)
    assert not scheduler.phantom_staged
    assert released == ["staged"]


def test_without_the_queue_a_staged_phantom_is_an_ordinary_one():
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(matched_tokens=BLOCK_SIZE * 2, is_async=False),
        block_size=BLOCK_SIZE,
    )
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True)
    scheduler.add_request(staged)
    assert not scheduler.phantom_staged
    assert staged in scheduler.waiting


def test_without_a_connector_a_staged_phantom_is_queued_as_usual(monkeypatch):
    monkeypatch.setenv("VLLM_PHANTOM_QUEUE", "1")
    scheduler = create_scheduler(block_size=BLOCK_SIZE)
    staged = _make(1, "staged", BLOCK_SIZE * 4, staged=True)
    scheduler.add_request(staged)
    assert not scheduler.phantom_staged
    assert scheduler.phantom_waiting == [staged]


def test_the_staged_counters_reach_the_stats(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_make(1, "staged", BLOCK_SIZE * 4, staged=True))
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.phantom_queue["staged_queued"] == 1
    assert stats.phantom_queue["staged"] == 1
