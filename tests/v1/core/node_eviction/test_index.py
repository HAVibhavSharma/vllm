# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build step 1: the block ownership index, against no pool at all."""

from vllm.v1.core.node_eviction.index import BlockOwnershipIndex
from vllm.v1.core.node_eviction.types import NodeKey

RESEARCH = NodeKey("run-42", "research", "tavily:summary")
SUPERVISOR = NodeKey("run-42", "supervisor", "plan")
OTHER_JOB = NodeKey("run-43", "research", "tavily:summary")


def test_add_and_read_back_positions():
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(10, 0), (11, 1), (12, 2)], now=1.0)

    entry = index.get_entry(RESEARCH)
    assert entry is not None
    assert entry.num_blocks == 3
    assert entry.run_len == 3
    assert index.position_of(RESEARCH, 11) == 1
    assert index.num_blocks == 3


def test_positions_tolerate_gaps():
    """`remove_skipped_blocks` replaces mid-list blocks with the null block
    for windowed and mamba models, so a run can have a hole (01 §7)."""
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(10, 0), (12, 2), (13, 3)], now=1.0)
    entry = index.get_entry(RESEARCH)
    assert entry.positions == {10: 0, 12: 2, 13: 3}
    # run_len reflects the furthest position, not the block count.
    assert entry.run_len == 4
    assert entry.num_blocks == 3


def test_multi_owner_shared_preamble():
    """A shared system preamble belongs to every node whose prefix starts
    with it. Single-owner tagging is what produced the previous attempt's
    RETAG thrash (00 Part 2, weakness 4)."""
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0), (2, 1)], now=1.0)
    index.add_blocks(SUPERVISOR, [(1, 0), (3, 1)], now=1.0)

    assert index.owners_of(1) == {RESEARCH, SUPERVISOR}
    assert index.owners_of(2) == {RESEARCH}
    assert index.num_blocks == 3


def test_remove_block_drops_it_from_every_owner():
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0), (2, 1)], now=1.0)
    index.add_blocks(SUPERVISOR, [(1, 0)], now=1.0)

    index.remove_block(1)
    assert index.owners_of(1) == set()
    assert index.get_entry(RESEARCH).positions == {2: 1}
    # SUPERVISOR owned only block 1, so its entry carries no information now.
    assert index.get_entry(SUPERVISOR) is None


def test_two_jobs_running_the_same_node_stay_separate():
    """Collapsing `(job, node)` to `node` would average unrelated forecasts
    from concurrent jobs (01 §4)."""
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0)], now=1.0)
    index.add_blocks(OTHER_JOB, [(2, 0)], now=1.0)
    assert index.num_keys == 2
    assert index.owners_of(1) == {RESEARCH}
    assert index.owners_of(2) == {OTHER_JOB}


def test_speculative_stamp_and_confirm():
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0)], now=1.0, speculative=True)
    assert index.get_entry(RESEARCH).speculative
    assert index.num_speculative_keys == 1

    assert index.confirm(RESEARCH, now=2.0) is True
    assert not index.get_entry(RESEARCH).speculative
    # Idempotent: a second confirm changes nothing.
    assert index.confirm(RESEARCH, now=3.0) is False


def test_confirm_block_confirms_every_owner():
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0)], now=1.0, speculative=True)
    index.add_blocks(SUPERVISOR, [(1, 0)], now=1.0, speculative=True)
    assert index.confirm_block(1, now=2.0) == 2
    assert index.num_speculative_keys == 0


def test_a_real_write_clears_the_speculative_stamp():
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0)], now=1.0, speculative=True)
    index.add_blocks(RESEARCH, [(2, 1)], now=2.0, speculative=False)
    assert not index.get_entry(RESEARCH).speculative


def test_gc_drops_silent_keys():
    """Age is a term in the score so a silent job decays out, but the index
    still needs a hard drop or it grows for the server's lifetime."""
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0)], now=0.0)
    index.add_blocks(SUPERVISOR, [(2, 0)], now=100.0)

    dropped = index.gc(now=150.0, hard_drop_age=60.0)
    assert dropped == 1
    assert index.get_entry(RESEARCH) is None
    assert index.get_entry(SUPERVISOR) is not None
    assert index.owners_of(1) == set()


def test_speculative_keys_age_out_on_the_same_clock_as_everything_else():
    """Speculative entries used to drop on a second clock — `TTL x multiple`
    past creation. That TTL was `time_to_next_call_ms`, the same clock the
    floor decay ran on, and it went with the decay: keeping it would be the
    same expiry under another name, deleting an entry the scorer is still
    holding at the floor.
    """
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0)], now=0.0, speculative=True)

    # Far past any old TTL multiple, and still held.
    assert index.gc(now=10_000.0, hard_drop_age=100_000.0) == 0
    assert index.get_entry(RESEARCH) is not None

    assert index.gc(now=100_001.0, hard_drop_age=100_000.0) == 1
    assert index.get_entry(RESEARCH) is None


def test_clear_drops_everything():
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0), (2, 1)], now=1.0)
    index.clear()
    assert index.num_keys == 0
    assert index.num_blocks == 0
    assert index.owners_of(1) == set()


def test_drop_key_leaves_co_owned_blocks_alone():
    index = BlockOwnershipIndex()
    index.add_blocks(RESEARCH, [(1, 0)], now=1.0)
    index.add_blocks(SUPERVISOR, [(1, 0)], now=1.0)
    index.drop_key(RESEARCH)
    assert index.owners_of(1) == {SUPERVISOR}
    assert index.num_blocks == 1
