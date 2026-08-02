# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one primitive the policy had to add, and the counter that gates it.

`appendleft` is purely additive — no existing method changes behaviour, and
with the policy off it is never called. That property is why the re-splice
was chosen over a banded free queue (01 §6.1, §6.5), so it is worth a test
that says so.
"""

import pytest

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


def make_queue(n: int, track_fresh: bool = True) -> tuple[
    FreeKVCacheBlockQueue, list[KVCacheBlock]
]:
    blocks = [KVCacheBlock(i) for i in range(n)]
    return FreeKVCacheBlockQueue(blocks, track_fresh=track_fresh), blocks


def ids(queue: FreeKVCacheBlockQueue) -> list[int]:
    return [b.block_id for b in queue.get_all_free_blocks()]


def test_appendleft_puts_block_at_head():
    queue, blocks = make_queue(4)
    block = queue.popleft()
    assert block.block_id == 0
    queue.append(block)
    assert ids(queue) == [1, 2, 3, 0]

    queue.remove(block)
    queue.appendleft(block)
    assert ids(queue) == [0, 1, 2, 3]
    assert queue.num_free_blocks == 4


def test_appendleft_n_preserves_order():
    """The sign-error trap: repeated `appendleft` reverses, `appendleft_n`
    does not. Callers pass candidates ascending so the worst-scoring block
    ends up at the head, i.e. first out of `popleft_n`."""
    queue, blocks = make_queue(6)
    moved = [blocks[4], blocks[5]]
    for block in moved:
        queue.remove(block)
    queue.appendleft_n(moved)

    assert ids(queue) == [4, 5, 0, 1, 2, 3]
    # blocks[4] was passed first, so it is what popleft returns first.
    assert queue.popleft().block_id == 4
    assert queue.num_free_blocks == 5


def test_appendleft_n_empty_is_a_noop():
    queue, _ = make_queue(3)
    before = ids(queue)
    queue.appendleft_n([])
    assert ids(queue) == before
    assert queue.num_free_blocks == 3


def test_appendleft_into_empty_queue():
    queue, blocks = make_queue(1)
    block = queue.popleft()
    assert queue.num_free_blocks == 0
    queue.appendleft(block)
    assert ids(queue) == [0]
    assert queue.popleft().block_id == 0


def test_round_trip_leaves_queue_consistent():
    queue, blocks = make_queue(8)
    for block in (blocks[7], blocks[3], blocks[0]):
        queue.remove(block)
    queue.appendleft_n([blocks[7], blocks[3], blocks[0]])
    assert ids(queue) == [7, 3, 0, 1, 2, 4, 5, 6]
    assert queue.num_free_blocks == 8
    # Walking the list backwards must produce the same order.
    backward = []
    cursor = queue.fake_free_list_tail.prev_free_block
    while cursor is not queue.fake_free_list_head:
        backward.append(cursor.block_id)
        cursor = cursor.prev_free_block
    assert backward[::-1] == ids(queue)


# -- num_free_fresh ------------------------------------------------------
#
# Rule 1 (01 §6): a fresh block is exactly `block_hash is None`, and it must
# be consumed before any cached block is destroyed. The previous attempt
# estimated this quantity from aggregate counters and admitted in-code that
# getting it wrong collapsed the cache to a single prompt.


def cache(block: KVCacheBlock, value: bytes = b"h") -> None:
    block.block_hash = value


def test_fresh_count_starts_at_all_blocks():
    queue, _ = make_queue(5)
    assert queue.num_free_fresh == 5


def test_fresh_count_tracks_pop_and_append():
    queue, blocks = make_queue(4)
    popped = queue.popleft()
    assert queue.num_free_fresh == 3

    cache(popped)
    queue.append(popped)
    # A cached block re-entering the queue is not fresh.
    assert queue.num_free_fresh == 3
    assert queue.num_free_blocks == 4


def test_fresh_count_tracks_bulk_ops():
    queue, blocks = make_queue(6)
    popped = queue.popleft_n(3)
    assert queue.num_free_fresh == 3
    for block in popped:
        cache(block)
    queue.append_n(popped)
    assert queue.num_free_fresh == 3
    assert queue.num_free_blocks == 6


def test_fresh_count_tracks_remove_and_appendleft():
    queue, blocks = make_queue(4)
    cache(blocks[2])
    # Caching happens while the block is allocated, so the counter is only
    # corrected when the block moves; emulate a fresh rebuild here.
    queue2, blocks2 = make_queue(4)
    cache(blocks2[2])
    queue2.remove(blocks2[2])
    # blocks2[2] was cached, so removing it does not change the fresh count.
    assert queue2.num_free_fresh == 4
    queue2.appendleft(blocks2[2])
    assert queue2.num_free_fresh == 4

    queue2.remove(blocks2[1])
    assert queue2.num_free_fresh == 3


def test_hash_reset_on_a_queued_block_is_observed():
    """`evict_blocks` can strip the hash from a block sitting in the queue,
    turning a cached block into a fresh one with no queue operation."""
    queue, blocks = make_queue(3)
    block = queue.popleft()
    cache(block)
    queue.append(block)
    assert queue.num_free_fresh == 2

    block.reset_hash()
    queue.on_queued_block_hash_reset(block)
    assert queue.num_free_fresh == 3


def test_hash_reset_on_an_unqueued_block_is_ignored():
    queue, blocks = make_queue(3)
    block = queue.popleft()
    cache(block)
    before = queue.num_free_fresh
    block.reset_hash()
    queue.on_queued_block_hash_reset(block)
    assert queue.num_free_fresh == before


def test_tracking_off_costs_nothing_and_stays_zero():
    """Flag-off must be byte-identical to upstream behaviour."""
    queue, blocks = make_queue(4, track_fresh=False)
    assert queue.num_free_fresh == 0
    queue.popleft_n(2)
    queue.append_n(blocks[:2])
    assert queue.num_free_fresh == 0
    assert queue.num_free_blocks == 4


def test_is_queued():
    queue, blocks = make_queue(3)
    assert queue.is_queued(blocks[0])
    popped = queue.popleft()
    assert not queue.is_queued(popped)


@pytest.mark.parametrize("n", [1, 2, 16])
def test_fresh_count_matches_a_full_scan(n):
    """Property: the O(1) counter must equal what an O(F) scan would say."""
    queue, blocks = make_queue(n)
    popped = queue.popleft_n(min(n, 2))
    for i, block in enumerate(popped):
        if i % 2 == 0:
            cache(block)
    queue.append_n(popped)
    scanned = sum(
        1 for b in queue.get_all_free_blocks() if b.block_hash is None
    )
    assert queue.num_free_fresh == scanned
