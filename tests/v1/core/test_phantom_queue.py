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
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

BLOCK_SIZE = 16


def _make(
    index: int, req_id: str, num_tokens: int, *, phantom: bool = False
) -> Request:
    """Request `index` of a batch, so its prompt is `[index] * num_tokens`.

    Two requests made with the same index share a prompt prefix; different
    indices never do.
    """
    request = create_requests(
        num_requests=index + 1,
        num_tokens=num_tokens,
        max_tokens=1 if phantom else 16,
        block_size=BLOCK_SIZE,
        req_ids=[f"_{j}" for j in range(index)] + [req_id],
    )[index]
    if phantom:
        request.kv_transfer_params = {"prefetch_only": True, "cache_salt": "agent::x"}
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


def test_one_phantom_at_a_time_newest_first(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    older = _make(1, "older", BLOCK_SIZE * 4, phantom=True)
    newer = _make(2, "newer", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(older)
    scheduler.add_request(newer)

    first = scheduler.schedule()
    assert "newer" in first.num_scheduled_tokens
    assert "older" not in first.num_scheduled_tokens
    assert scheduler.phantom_waiting == [older]

    # `newer` still holds its blocks, so the slot is taken.
    second = scheduler.schedule()
    assert "older" not in second.num_scheduled_tokens
    assert scheduler.phantom_waiting == [older]
    assert scheduler._phantom_stats["held_slot"] >= 1


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


def test_a_real_request_overtakes_its_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    # Same index: the phantom's prompt is the first half of the real one's.
    phantom = _make(0, "phantom", BLOCK_SIZE * 4, phantom=True)
    real = _make(0, "real", BLOCK_SIZE * 8)
    scheduler.add_request(phantom)
    scheduler.add_request(real)
    assert not scheduler.phantom_waiting
    assert scheduler._phantom_stats["dropped_on_arrival"] == 1

    output = scheduler.schedule()
    assert "real" in output.num_scheduled_tokens
    ecos = scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert phantom.status == RequestStatus.FINISHED_STOPPED
    assert len(_finished(ecos, "phantom")) == 1


def test_a_different_prompt_does_not_drop_a_phantom(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    phantom = _make(1, "phantom", BLOCK_SIZE * 4, phantom=True)
    scheduler.add_request(phantom)
    scheduler.add_request(_make(2, "real", BLOCK_SIZE * 8))
    assert scheduler.phantom_waiting == [phantom]


def test_the_counters_reach_the_stats(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler.add_request(_make(1, "phantom", BLOCK_SIZE * 4, phantom=True))
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.phantom_queue["queued"] == 1
    assert stats.phantom_queue["real_blocked_with_phantom"] == 0


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
