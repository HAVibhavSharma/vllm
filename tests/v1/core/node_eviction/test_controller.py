# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build step 3: the bounded re-splice, against a fake pool.

No GPU, no model, no Redis — the snapshot comes from a fixture, which is the
whole reason the snapshot interface exists.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.controller import (
    NodeEvictionController,
    node_key_for_request,
)
from vllm.v1.core.node_eviction.snapshot import StaticSnapshotSource
from vllm.v1.core.node_eviction.types import ImportanceRow, NodeKey

RESEARCH = NodeKey("run-42", "research", "tavily:summary")
SUPERVISOR = NodeKey("run-42", "supervisor", "plan")


class FakePool:
    """Just enough of BlockPool for the splice: a block table and the real
    free queue, so the linked-list bookkeeping is genuinely exercised.

    `cached=True` is the default because it is the only state in which the
    splice runs at all: Rule 1 skips the tick entirely while any unused block
    is still queued, so a half-empty pool would make every splice assertion
    below vacuously pass.
    """

    def __init__(self, num_blocks: int = 16, cached: bool = True):
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.num_gpu_blocks = num_blocks
        if cached:
            for block in self.blocks:
                block.block_hash = f"h{block.block_id}".encode()
        self.free_block_queue = FreeKVCacheBlockQueue(
            self.blocks, track_fresh=True
        )

    @property
    def num_free_fresh(self) -> int:
        return self.free_block_queue.num_free_fresh

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def queue_ids(self) -> list[int]:
        return [b.block_id for b in self.free_block_queue.get_all_free_blocks()]


def make_request(
    job_id="run-42",
    node="research",
    call_type="tavily:summary",
    prefetch_only=False,
    request_id="req-0",
):
    extra_args = {}
    if job_id is not None:
        extra_args["job_id"] = job_id
    if node is not None:
        extra_args["langgraph_node"] = node
    if call_type is not None:
        extra_args["call_type"] = call_type
    return SimpleNamespace(
        request_id=request_id,
        num_preemptions=0,
        sampling_params=SimpleNamespace(extra_args=extra_args),
        kv_transfer_params={"prefetch_only": True} if prefetch_only else None,
    )


def make_controller(pool, rows=None, **overrides):
    config = NodeEvictionConfig(enabled=True, tick_period_ms=0.0, **overrides)
    source = StaticSnapshotSource(rows or {})
    return NodeEvictionController(
        block_pool=pool, config=config, snapshot_source=source
    )


def fresh_rows(per_key: dict[NodeKey, dict]) -> dict[NodeKey, ImportanceRow]:
    """Rows with an update_ts far in the future so the staleness gate is not
    what the test is measuring.

    Takes the mapping positionally rather than as `**kwargs`: the keys are
    `NodeKey` tuples, and `**` unpacking requires string keys.
    """
    import time

    now_ms = time.time() * 1000.0
    return {
        key: ImportanceRow(update_ts_ms=now_ms, **kwargs)
        for key, kwargs in per_key.items()
    }


# -- identity ------------------------------------------------------------


def test_identity_is_read_from_extra_args():
    """All three ride in sampling_params.extra_args and already cross into
    engine core, so there is no protocol change (00 Part 4)."""
    assert node_key_for_request(make_request()) == RESEARCH


def test_missing_identity_yields_no_key():
    assert node_key_for_request(make_request(job_id=None)) is None
    assert node_key_for_request(make_request(node=None)) is None
    assert node_key_for_request(SimpleNamespace(sampling_params=None)) is None


def test_missing_call_type_falls_back_to_the_empty_label():
    key = node_key_for_request(make_request(call_type=None))
    assert key == NodeKey("run-42", "research", "")


def test_call_type_can_be_switched_off():
    """03 §1 requirement 2: if the flattening is not deterministic the engine
    keys at (job_id, node) instead. It is a one-line change on our side."""
    key = node_key_for_request(make_request(), use_call_type=False)
    assert key == NodeKey("run-42", "research", "")


# -- indexing ------------------------------------------------------------


def test_cached_blocks_are_indexed_with_prefix_positions():
    pool = FakePool()
    controller = make_controller(pool)
    controller.on_blocks_cached(
        make_request(), [pool.blocks[4], pool.blocks[5]], start_position=2
    )
    entry = controller.index.get_entry(RESEARCH)
    assert entry.positions == {4: 2, 5: 3}


def test_requests_without_identity_are_not_indexed():
    pool = FakePool()
    controller = make_controller(pool)
    controller.on_blocks_cached(
        make_request(job_id=None), [pool.blocks[1]], start_position=0
    )
    assert controller.index.num_keys == 0


# -- the splice ----------------------------------------------------------


def index_prefix(controller, pool, key, block_ids, start=0, speculative=False):
    request = make_request(
        job_id=key.job_id,
        node=key.node,
        call_type=key.call_type,
        prefetch_only=speculative,
    )
    controller.on_blocks_cached(
        request, [pool.blocks[b] for b in block_ids], start_position=start
    )


def test_low_scoring_blocks_move_to_the_head():
    """The walkthrough scenario (06 §3.3): `supervisor` is not needed for
    another minute, `research` is back in 8 seconds. LRU evicts `research`
    because it was freed first; the policy must evict `supervisor`."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.92, time_to_next_call_ms=8_000.0),
            SUPERVISOR: dict(prob=0.85, time_to_next_call_ms=60_000.0),
        }
    )
    controller = make_controller(pool, rows)
    # research freed first, so LRU has it nearest the head.
    index_prefix(controller, pool, RESEARCH, [1, 2, 3])
    index_prefix(controller, pool, SUPERVISOR, [4, 5, 6])

    # Precondition, stated as the ordering rather than as an absolute list:
    # `index_prefix` only records ownership, it does not touch the queue, so
    # the queue is still in construction order and block 0 — owned by nobody
    # — is at the head. What matters is that research sits *ahead of*
    # supervisor, which is the LRU-anti-correlated setup: age says evict
    # research, the forecast says evict supervisor.
    queue = pool.queue_ids()
    assert queue.index(1) < queue.index(4)

    controller.maybe_tick()

    head = pool.queue_ids()[:3]
    assert set(head) == {4, 5, 6}, "supervisor's blocks must be evicted first"


def test_worst_block_ends_up_first_out():
    """Ascending order into `appendleft_n`, so `popleft` returns the
    worst-scoring block. Getting this backwards evicts the *best* blocks
    first, and the only symptom is an inverted hit rate."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0),
            SUPERVISOR: dict(prob=0.1, time_to_next_call_ms=600_000.0),
        }
    )
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1])
    index_prefix(controller, pool, SUPERVISOR, [2])

    controller.maybe_tick()
    assert pool.free_block_queue.popleft().block_id == 2


def test_tail_of_prefix_is_evicted_before_its_head():
    """Rule 3: a prefix is only worth anything as a contiguous run from
    position 0, so freeing from the middle reclaims one block and destroys
    the whole match."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.5, time_to_next_call_ms=1_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3], start=0)

    controller.maybe_tick()
    # Same score across the key, so position breaks the tie: highest
    # position nearest the head.
    assert pool.queue_ids()[:3] == [3, 2, 1]


def test_unscored_blocks_keep_their_lru_position():
    """Rule 2: the splice only *moves* the worst K, so an unscored block is
    simply never selected. No neutral-score arithmetic is needed."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({SUPERVISOR: dict(prob=0.1, time_to_next_call_ms=600_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1, 2])  # no forecast row
    index_prefix(controller, pool, SUPERVISOR, [5])

    before = pool.queue_ids()
    controller.maybe_tick()
    after = pool.queue_ids()

    assert after[0] == 5
    # research keeps its relative order among everything else.
    assert [b for b in after if b in (1, 2)] == [b for b in before if b in (1, 2)]


def test_running_blocks_are_never_selected():
    """Invariant 8: ref_cnt > 0 keeps a running node's blocks out of the
    free queue entirely."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.01, time_to_next_call_ms=600_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1, 2])

    # Simulate allocation: pull block 1 out of the queue and reference it.
    pool.free_block_queue.remove(pool.blocks[1])
    pool.blocks[1].ref_cnt = 1

    controller.maybe_tick()
    assert pool.queue_ids()[0] == 2
    assert 1 not in pool.queue_ids()


def test_splice_is_bounded_by_k():
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.1, time_to_next_call_ms=600_000.0)})
    controller = make_controller(pool, rows, splice_max_blocks=2)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3, 4, 5])

    controller.maybe_tick()
    assert controller.observer.counters.blocks_spliced_total == 2


def test_splice_skipped_while_fresh_blocks_are_plentiful():
    """Rule 1: `appendleft` inserts at the absolute head, so splicing a
    cached block ahead of a fresh one throws away a prefix while a free
    unused block was available."""
    pool = FakePool(num_blocks=16, cached=False)
    rows = fresh_rows({RESEARCH: dict(prob=0.1, time_to_next_call_ms=600_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [10, 11])
    # Nothing in this pool carries a hash, so every queued block is fresh.
    assert pool.num_free_fresh > 0

    before = pool.queue_ids()
    controller.maybe_tick()
    assert pool.queue_ids() == before
    assert controller.observer.counters.ticks_skipped_fresh == 1


def test_multi_owner_takes_the_max():
    """A shared preamble is worth what its most valuable owner makes it
    worth; summing would make common prefixes permanently unevictable."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.99, time_to_next_call_ms=100.0),
            SUPERVISOR: dict(prob=0.01, time_to_next_call_ms=600_000.0),
        }
    )
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1])
    index_prefix(controller, pool, SUPERVISOR, [1, 2])

    controller.maybe_tick()
    # Block 1 is co-owned by the valuable key, so block 2 goes first.
    assert pool.queue_ids()[0] == 2


def test_score_threshold_leaves_valuable_blocks_alone():
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.99, time_to_next_call_ms=0.0)})
    controller = make_controller(pool, rows, score_threshold=1e-9)
    index_prefix(controller, pool, RESEARCH, [1, 2])

    before = pool.queue_ids()
    controller.maybe_tick()
    assert pool.queue_ids() == before


# -- the speculative floor ----------------------------------------------


def test_the_floor_is_off_by_default_and_a_prefetch_is_scored_honestly():
    """The shipped default. A prefetched key is ranked on its own row, so a
    low-probability prediction sorts below a real, imminent one instead of
    above every key in the table.

    This is the trade the floor was making: protecting the prediction meant
    the splice chose *real* blocks instead, and via the `max` over owners one
    phantom sharing a system preamble lifted that preamble out of the
    candidate set for every node sharing it."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.01, time_to_next_call_ms=30_000.0),
            SUPERVISOR: dict(prob=0.9, time_to_next_call_ms=1_000.0),
        }
    )
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1], speculative=True)
    index_prefix(controller, pool, SUPERVISOR, [2])

    controller.maybe_tick()

    assert controller.get_value(RESEARCH) < controller.get_value(SUPERVISOR)
    assert pool.queue_ids()[0] == 1, "the weaker forecast goes first"
    # Provenance survives the floor's removal: the waste ratio is how you
    # tell whether the prefetch half is worth running at all.
    assert controller.observer.counters.speculative_created == 1
    assert controller.index.get_entry(RESEARCH).speculative


def test_a_fresh_prefetch_is_protected_from_its_own_policy():
    """The regression this policy is most capable of producing: the splice
    removes LRU's accidental tail protection, so without the floor a prefix
    the policy just paid ~11.4s to fetch can be evicted immediately
    (02 §5).

    Opt-in now — `speculative_floor_high` defaults to 0. Kept so the
    protected behaviour stays A/B-able against the default."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.01, time_to_next_call_ms=30_000.0),
            SUPERVISOR: dict(prob=0.9, time_to_next_call_ms=1_000.0),
        }
    )
    controller = make_controller(pool, rows, speculative_floor_high=1e9)
    # research arrives as a phantom prefetch: low base score, but freshly
    # predicted.
    index_prefix(controller, pool, RESEARCH, [1], speculative=True)
    index_prefix(controller, pool, SUPERVISOR, [2])

    controller.maybe_tick()
    assert pool.queue_ids()[0] == 2, "the prefetched block must not go first"
    assert controller.observer.counters.speculative_created == 1


def test_an_old_prefetch_is_still_protected():
    """The floor does not expire. It used to decay to
    `speculative_floor_low = -1.0` over `time_to_next_call_ms`, which put an
    unused prefetch *below* every real score — so the block the policy paid a
    prefill for became the first one evicted, at roughly the moment its
    predicted call was due.

    `time_to_next_call_ms` is deliberately 60s here, the arm that used to run
    the countdown to first-out.

    Opt-in now, like `test_a_fresh_prefetch_is_protected_from_its_own_policy`.
    """
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.01, time_to_next_call_ms=60_000.0),
            SUPERVISOR: dict(prob=0.9, time_to_next_call_ms=1_000.0),
        }
    )
    controller = make_controller(pool, rows, speculative_floor_high=1e9)
    index_prefix(controller, pool, RESEARCH, [1], speculative=True)
    index_prefix(controller, pool, SUPERVISOR, [2])

    # Backdate the entry well past every TTL that used to exist: the old
    # default (30s), the row's own 60s, and the 4x hard-drop multiple.
    entry = controller.index.get_entry(RESEARCH)
    entry.created_at -= 600.0
    entry.last_seen -= 600.0

    controller.maybe_tick()

    assert pool.queue_ids()[0] == 2, "the aged prefetch must still not go first"
    assert controller.get_value(RESEARCH) > controller.get_value(SUPERVISOR)


def test_confirm_on_touch_clears_the_stamp():
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1], speculative=True)
    assert controller.index.get_entry(RESEARCH).speculative

    controller.on_prefix_hit(make_request(), [[pool.blocks[1]]])
    assert not controller.index.get_entry(RESEARCH).speculative
    assert controller.observer.counters.speculative_confirmed == 1


def test_a_phantom_cannot_confirm_another_phantoms_guess():
    """Ungated, one prediction confirms another — the forecast marking its
    own homework (02 §5 part 2)."""
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1], speculative=True)

    controller.on_prefix_hit(make_request(prefetch_only=True), [[pool.blocks[1]]])
    assert controller.index.get_entry(RESEARCH).speculative


def test_fresh_blocks_behind_cached_ones_do_not_block_the_splice():
    """Rule 1 is about blocks `appendleft` can jump, i.e. the head. A block
    freed with no hash is appended to the *tail* and still counts as fresh, so
    the whole-queue count has a permanent floor in a steady workload — a real
    run never saw it below 21, the tick skipped every time, and every score in
    the decision log was null."""
    pool = FakePool(num_blocks=16)
    # Head is cached, tail is fresh: the next allocation destroys a prefix no
    # matter what, so the splice is exactly what should decide which one.
    for block in pool.blocks[-4:]:
        block.block_hash = None
    assert pool.num_free_fresh == 4, "the whole-queue count is above threshold"

    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.92, time_to_next_call_ms=8_000.0),
            SUPERVISOR: dict(prob=0.85, time_to_next_call_ms=60_000.0),
        }
    )
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1])
    index_prefix(controller, pool, SUPERVISOR, [2])

    controller.maybe_tick()

    assert controller.observer.counters.ticks_skipped_fresh == 0
    assert pool.queue_ids()[0] == 2, "supervisor is worth less and must go first"


def test_a_fresh_head_still_stops_the_splice():
    """The other half of the rule: while the very next block to be popped is
    unused, splicing a cached block in front of it would throw away a prefix
    with a free block sitting right there."""
    pool = FakePool(num_blocks=16)
    pool.blocks[0].block_hash = None

    rows = fresh_rows({SUPERVISOR: dict(prob=0.1, time_to_next_call_ms=60_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, SUPERVISOR, [2])

    controller.maybe_tick()

    assert controller.observer.counters.ticks_skipped_fresh == 1
    assert controller.observer.counters.splices_total == 0


def test_waste_is_counted_even_before_a_tick_has_scored_the_key():
    """Provenance comes from the index, which knows at insert time. Reading
    it off the score breakdown instead loses every prefetch evicted inside
    one tick period (250ms) — the fastest and most wasteful case there is."""
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3], speculative=True)
    assert controller.get_value(RESEARCH) is None, "no tick has scored it"

    controller.on_block_evicted(1)

    counters = controller.observer.counters
    assert counters.speculative_evicted_before_confirm == 1
    assert counters.speculative_blocks_created == 3
    assert counters.speculative_waste == 1 / 3


def test_a_block_a_confirmed_owner_also_holds_is_not_prefetch_waste():
    """Multi-owner `max` already protects a co-owned block (01 §4), and a
    confirmed owner means it was genuinely used. Blaming the prefetch for it
    would inflate the only number that says whether the forecast is worth
    anything."""
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1], speculative=True)
    index_prefix(controller, pool, SUPERVISOR, [1])

    controller.on_block_evicted(1)

    assert controller.observer.counters.speculative_evicted_before_confirm == 0


# -- lifecycle -----------------------------------------------------------


def test_eviction_drops_the_block_and_records_a_decision():
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1, 2])

    controller.on_block_evicted(1)
    assert controller.index.owners_of(1) == set()
    assert controller.observer.counters.evictions_total == 1


def test_reset_prefix_cache_clears_the_index():
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1, 2])
    controller.on_reset_prefix_cache()
    assert controller.index.num_keys == 0
    assert controller.get_value(RESEARCH) is None


def test_per_request_consult_is_a_lookup():
    """02 §2: the tick computes, the request reads. A miss returns None,
    which callers read as 'use the default'."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.5, time_to_next_call_ms=1_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, list(range(1, 16)))

    assert controller.get_value(RESEARCH) is None  # no tick yet
    controller.maybe_tick()
    assert controller.get_value(RESEARCH) > 0.0
    assert controller.get_value(SUPERVISOR) is None
    assert controller.get_value(None) is None


def test_hybrid_models_disable_the_policy():
    """One free queue is shared by every group, and a hit at prefix position
    p needs a hit in *every* group — so evicting one group's block at p
    destroys the hit for all of them while reclaiming a single block."""
    pool = FakePool(num_blocks=16)
    config = NodeEvictionConfig(enabled=True, tick_period_ms=0.0)
    controller = NodeEvictionController(
        block_pool=pool, config=config, num_kv_cache_groups=2
    )
    assert not controller.enabled

    controller.on_blocks_cached(make_request(), [pool.blocks[1]], 0)
    before = pool.queue_ids()
    controller.maybe_tick()
    assert controller.index.num_keys == 0
    assert pool.queue_ids() == before


def test_tick_skips_when_nothing_changed():
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1])
    controller.maybe_tick()
    controller.maybe_tick()
    assert controller.observer.counters.ticks_skipped_unchanged >= 1


def test_config_rejects_a_floor_below_the_score_range():
    """A floor between 0 and delta_cold protects some keys and not others —
    a silent partial policy. 0 is exempt because it means "no floor"."""
    with pytest.raises(ValueError):
        NodeEvictionConfig(
            enabled=True, speculative_floor_high=1.0
        ).validate()
    with pytest.raises(ValueError):
        NodeEvictionConfig(
            enabled=True, speculative_floor_high=-1.0
        ).validate()


def test_config_accepts_the_floor_switched_off():
    NodeEvictionConfig(enabled=True, speculative_floor_high=0.0).validate()
    assert NodeEvictionConfig().speculative_floor_high == 0.0


def test_config_rejects_a_cold_miss_cheaper_than_l1():
    with pytest.raises(ValueError):
        NodeEvictionConfig(delta_l1_ms=500.0, delta_cold_ms=100.0).validate()


# -- HBM accounting line -------------------------------------------------


def test_hbm_summary_reports_the_pool_split():
    """total = used + free, and `queue` is the eviction candidate count.

    Asserted as an identity rather than against literals so the test still
    means something if the fake pool's size changes.
    """
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    pool.blocks[3].ref_cnt = 1
    pool.free_block_queue.remove(pool.blocks[3])

    line = controller.hbm_summary()
    fields = dict(
        part.split("=", 1) for part in line.split() if "=" in part
    )
    assert fields["variant"] == "node_eviction"
    total = int(fields["total"])
    used = int(fields["used"])
    free = int(fields["free"])
    assert total == 16
    assert used + free == total
    assert used == 1
    assert int(fields["queue"]) == free


def test_hbm_summary_names_the_node_that_lost_the_block():
    """The point of the line: not "12k blocks went" but whose they were."""
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3])
    index_prefix(controller, pool, SUPERVISOR, [4])

    for block_id in (1, 2, 3):
        controller.on_block_evicted(block_id)
    controller.on_block_evicted(4)

    line = controller.hbm_summary()
    assert "top_evicted=run-42:research=3,run-42:supervisor=1" in line


def test_hbm_summary_attributes_untracked_blocks_separately():
    """A block nobody claimed must not be silently dropped from the tally,
    or `evicted=` and `top_evicted=` stop adding up."""
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    controller.on_block_evicted(7)
    assert "top_evicted=<untracked>=1" in controller.hbm_summary()


def test_hbm_line_is_rate_limited_and_change_gated():
    """An idle server must log nothing: an identical line every period
    trains everyone to filter the line out."""
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool, hbm_summary_period_ms=10_000.0)

    assert controller._maybe_log_hbm_summary(1000.0) is not None
    # Same window.
    assert controller._maybe_log_hbm_summary(1001.0) is None
    # New window, but nothing moved.
    assert controller._maybe_log_hbm_summary(1_000_000.0) is None

    index_prefix(controller, pool, RESEARCH, [1])
    controller.on_block_evicted(1)
    assert controller._maybe_log_hbm_summary(2_000_000.0) is not None


def test_hbm_attribution_is_per_window():
    """Counts reset when the line is emitted, which is also what stops a
    finished job's id from being pinned for the life of the server."""
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool, hbm_summary_period_ms=10_000.0)
    index_prefix(controller, pool, RESEARCH, [1])
    controller.on_block_evicted(1)

    first = controller._maybe_log_hbm_summary(1_000_000.0)
    assert "run-42:research=1" in first

    index_prefix(controller, pool, SUPERVISOR, [4])
    controller.on_block_evicted(4)
    second = controller._maybe_log_hbm_summary(2_000_000.0)
    assert "run-42:supervisor=1" in second
    assert "run-42:research" not in second


def test_hbm_line_can_be_switched_off():
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool, hbm_summary_period_ms=0.0)
    controller.on_block_evicted(1)
    assert controller._maybe_log_hbm_summary(1_000_000.0) is None


def test_stats_carry_the_pool_split():
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool)
    stats = controller.stats()
    assert stats["hbm_total_blocks"] == 16
    assert (
        stats["hbm_used_blocks"] + stats["hbm_free_blocks"]
        == stats["hbm_total_blocks"]
    )
    assert stats["free_queue_len"] == stats["hbm_free_blocks"]


# -- hit rate and cache movement -----------------------------------------


def cache_block(controller, pool, block_id, tag, node="research"):
    block = pool.blocks[block_id]
    if block.block_hash is not None:
        block.reset_hash()
    block.block_hash = tag
    controller.on_blocks_cached(
        make_request(node=node), [block], start_position=0
    )


def test_hit_rate_is_tokens_not_requests():
    """A 10-token request that fully hit and a 10k-token one that missed are
    not one hit and one miss — the second cost a thousand times more."""
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=10, num_hits=10)
    controller.on_cache_query(num_tokens=10_000, num_hits=0)
    assert controller.movement.hit_rate == 10 / 10_010


def test_windowed_hit_rate_resets_but_cumulative_does_not():
    """A cumulative rate over a long run is dominated by whatever the
    workload did first, so the line carries both."""
    controller = make_controller(FakePool(), hbm_summary_period_ms=10_000.0)
    controller.on_cache_query(num_tokens=100, num_hits=0)
    controller._maybe_log_hbm_summary(1_000_000.0)

    controller.on_cache_query(num_tokens=100, num_hits=100)
    line = controller._maybe_log_hbm_summary(2_000_000.0)
    fields = dict(p.split("=", 1) for p in line.split() if "=" in p)
    assert float(fields["hit_rate_win"]) == 1.0
    assert float(fields["hit_rate"]) == 0.5


def test_a_phantom_lookup_is_not_counted_as_demand():
    """A phantom is work the policy originated, not work the cache served.

    Counting it does not merely add noise, it adds noise with no consistent
    sign: the miss below would drag the rate to 0.5 while a phantom for an
    already-resident prefix would push it to 1.0. The baseline this line is
    diffed against originates nothing, so either direction is a false
    difference attributed to the policy.
    """
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=100, num_hits=100)
    controller.on_cache_query(
        num_tokens=100, num_hits=0, request=make_request(prefetch_only=True)
    )

    m = controller.movement
    assert m.hit_rate == 1.0
    assert m.window_hit_rate == 1.0
    assert m.query_tokens == 100


def test_phantom_traffic_is_reported_rather_than_discarded():
    """Dropping it would hide the prefill origination bought. With
    `--max-num-seqs 1` a phantom that misses is a full prefill serialized
    ahead of real traffic (12 §5.6), so it has to stay visible — just not
    inside the number it was meant to improve."""
    controller = make_controller(FakePool())
    phantom = make_request(prefetch_only=True)
    controller.on_cache_query(num_tokens=100, num_hits=25, request=phantom)

    m = controller.movement
    assert m.phantom_query_tokens == 100
    assert m.phantom_hit_rate == 0.25
    # No demand was ever observed, so the real rate has nothing to report.
    assert m.query_tokens == 0
    assert m.hit_rate == 0.0


def test_a_request_with_no_transfer_params_is_demand():
    """The overwhelmingly common path: `request` is passed on every query, so
    the phantom test must not misfire on an ordinary request."""
    controller = make_controller(FakePool())
    controller.on_cache_query(
        num_tokens=100, num_hits=50, request=make_request()
    )
    assert controller.movement.query_tokens == 100
    assert controller.movement.phantom_query_tokens == 0


def test_reset_prefix_cache_clears_the_hit_rate():
    """The tokens described a cache that no longer exists.

    `reset_prefix_cache` is only ever invoked explicitly, which in practice
    means "start a clean measurement". Upstream agrees: `CachingMetrics`
    resets its own aggregation on the same signal, so keeping these would
    also put the two numbers permanently out of step after the first reset.
    """
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=1_000, num_hits=1_000)
    assert controller.movement.hit_rate == 1.0

    controller.on_reset_prefix_cache()
    assert controller.movement.query_tokens == 0
    assert controller.movement.hit_rate == 0.0

    controller.on_cache_query(num_tokens=100, num_hits=0)
    assert controller.movement.hit_rate == 0.0, "the old tokens must not carry"


def test_the_summary_line_prints_when_only_phantoms_ran():
    """Phantom queries no longer move `query_tokens`, so the change gate has
    to watch them too — otherwise origination goes silent exactly in the
    window where it is doing all the work."""
    controller = make_controller(FakePool(), hbm_summary_period_ms=10_000.0)
    controller.on_cache_query(num_tokens=100, num_hits=100)
    assert controller._maybe_log_hbm_summary(1_000_000.0) is not None

    controller.on_cache_query(
        num_tokens=500, num_hits=0, request=make_request(prefetch_only=True)
    )
    line = controller._maybe_log_hbm_summary(2_000_000.0)
    assert line is not None
    fields = dict(p.split("=", 1) for p in line.split() if "=" in p)
    assert fields["phantom_query_tokens"] == "500"
    assert float(fields["phantom_hit_rate"]) == 0.0
    # The real rate is untouched by the phantom.
    assert float(fields["hit_rate"]) == 1.0


def test_cold_tokens_excludes_what_lmcache_served():
    """The field exists precisely to stop `query - hit` being read as prefill.

    HBM held 200 of 1,000 and the connector supplied another 700, so only
    100 tokens ever reached the model — against a miss count of 800.
    """
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=1_000, num_hits=200)
    controller.on_external_cache_query(
        num_tokens=1_000, num_local_hits=200, num_external_hits=700
    )

    m = controller.movement
    assert m.query_tokens - m.hit_tokens == 800
    assert m.cold_tokens == 100


def test_cold_tokens_equals_the_miss_count_with_no_connector():
    """No external tier means the two numbers must agree — otherwise the
    field would quietly under-report prefill on a server without LMCache."""
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=500, num_hits=120)
    controller.on_external_cache_query(
        num_tokens=500, num_local_hits=120, num_external_hits=0
    )

    m = controller.movement
    assert m.cold_tokens == m.query_tokens - m.hit_tokens == 380


def test_cold_tokens_never_goes_negative():
    """A connector reporting a span that overlaps the local hit would drive
    the subtraction below zero; the counter is a volume, so it clamps."""
    controller = make_controller(FakePool())
    controller.on_external_cache_query(
        num_tokens=100, num_local_hits=80, num_external_hits=90
    )
    assert controller.movement.cold_tokens == 0


def test_a_phantom_contributes_no_cold_tokens():
    """Same rule as every other headline field: a phantom's prefill is work
    the policy originated, and mixing it in would break the baseline diff."""
    controller = make_controller(FakePool())
    controller.on_external_cache_query(
        num_tokens=1_000,
        num_local_hits=0,
        num_external_hits=0,
        request=make_request(prefetch_only=True),
    )
    assert controller.movement.cold_tokens == 0


def test_cold_tokens_is_on_the_summary_line():
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=1_000, num_hits=200)
    controller.on_external_cache_query(
        num_tokens=1_000, num_local_hits=200, num_external_hits=700
    )
    fields = dict(
        p.split("=", 1) for p in controller.hbm_summary().split() if "=" in p
    )
    assert fields["cold_tokens"] == "100"


def test_a_measurement_reset_zeroes_cold_tokens():
    """It is a count of things that happened, and a warmup exists so those
    things happen unmeasured."""
    controller = make_controller(FakePool())
    controller.on_external_cache_query(
        num_tokens=1_000, num_local_hits=0, num_external_hits=0
    )
    assert controller.movement.cold_tokens == 1_000

    controller.movement.reset_measurement()
    assert controller.movement.cold_tokens == 0


def test_external_hit_tokens_counts_what_lmcache_served():
    """The complement of `cold_tokens` from the same subtraction: of the 800
    tokens HBM missed, the connector supplied 700."""
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=1_000, num_hits=200)
    controller.on_external_cache_query(
        num_tokens=1_000, num_local_hits=200, num_external_hits=700
    )

    m = controller.movement
    assert m.external_hit_tokens == 700
    assert m.hit_tokens + m.external_hit_tokens + m.cold_tokens == m.query_tokens


def test_external_hit_tokens_is_capped_by_what_hbm_missed():
    """An overlapping span must cost the external tier its excess rather
    than drive `cold_tokens` negative — the two still sum to the misses."""
    controller = make_controller(FakePool())
    controller.on_external_cache_query(
        num_tokens=100, num_local_hits=80, num_external_hits=90
    )

    m = controller.movement
    assert m.external_hit_tokens == 20
    assert m.cold_tokens == 0


def test_external_hit_tokens_is_zero_with_no_connector():
    controller = make_controller(FakePool())
    controller.on_external_cache_query(
        num_tokens=500, num_local_hits=120, num_external_hits=0
    )

    m = controller.movement
    assert m.external_hit_tokens == 0
    assert m.cold_tokens == 380


def test_a_phantom_contributes_no_external_hit_tokens():
    """Warming work the policy originated, not demand a tier served."""
    controller = make_controller(FakePool())
    controller.on_external_cache_query(
        num_tokens=1_000,
        num_local_hits=0,
        num_external_hits=900,
        request=make_request(prefetch_only=True),
    )
    assert controller.movement.external_hit_tokens == 0


def test_external_hit_tokens_is_on_the_summary_line():
    controller = make_controller(FakePool())
    controller.on_cache_query(num_tokens=1_000, num_hits=200)
    controller.on_external_cache_query(
        num_tokens=1_000, num_local_hits=200, num_external_hits=700
    )
    fields = dict(
        p.split("=", 1) for p in controller.hbm_summary().split() if "=" in p
    )
    assert fields["external_hit_tokens"] == "700"


def test_a_measurement_reset_zeroes_external_hit_tokens():
    controller = make_controller(FakePool())
    controller.on_external_cache_query(
        num_tokens=1_000, num_local_hits=0, num_external_hits=600
    )
    assert controller.movement.external_hit_tokens == 600

    controller.movement.reset_measurement()
    assert controller.movement.external_hit_tokens == 0


def test_a_first_time_prefix_is_not_movement():
    """The number must credit the policy only for work it made us redo. A
    prefix the server has never seen is not that."""
    pool = FakePool(cached=False)
    controller = make_controller(pool)
    for i in (1, 2, 3):
        cache_block(controller, pool, i, f"h{i}")
    assert controller.movement.remat_blocks == 0


def test_recaching_an_evicted_hash_counts_as_movement():
    pool = FakePool(cached=False)
    controller = make_controller(pool)
    cache_block(controller, pool, 1, "hA")
    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()

    cache_block(controller, pool, 1, "hA")
    assert controller.movement.remat_blocks == 1


def test_movement_needs_a_second_eviction_to_count_twice():
    """Otherwise every later cache of a popular prefix reads as movement and
    the number grows without anything being rebuilt."""
    pool = FakePool(cached=False)
    controller = make_controller(pool)
    cache_block(controller, pool, 1, "hA")
    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(controller, pool, 1, "hA")
    assert controller.movement.remat_blocks == 1

    pool.blocks[1].reset_hash()
    cache_block(controller, pool, 1, "hA")
    assert controller.movement.remat_blocks == 1

    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(controller, pool, 1, "hA")
    assert controller.movement.remat_blocks == 2


def test_movement_counts_blocks_from_requests_with_no_identity():
    """Movement is a property of the cache, not the workflow. The baseline
    counts anonymous blocks, so skipping them here would make the two runs'
    numbers incomparable — which is the whole point of the line."""
    pool = FakePool(cached=False)
    controller = make_controller(pool)
    anon = make_request(job_id=None)
    pool.blocks[1].block_hash = "hA"
    controller.on_blocks_cached(anon, [pool.blocks[1]], start_position=0)
    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    pool.blocks[1].block_hash = "hA"
    controller.on_blocks_cached(anon, [pool.blocks[1]], start_position=0)

    assert controller.index.num_keys == 0, "still not indexed"
    assert controller.movement.remat_blocks == 1


def test_reset_prefix_cache_is_not_movement():
    """An operator wiping the cache did not make the policy redo work."""
    pool = FakePool(cached=False)
    controller = make_controller(pool)
    cache_block(controller, pool, 1, "hA")
    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()

    controller.on_reset_prefix_cache()
    cache_block(controller, pool, 1, "hA")
    assert controller.movement.remat_blocks == 0


def test_movement_in_mb_uses_the_configured_page_size():
    pool = FakePool(cached=False)
    config = NodeEvictionConfig(enabled=True, tick_period_ms=0.0)
    controller = NodeEvictionController(
        block_pool=pool,
        config=config,
        snapshot_source=StaticSnapshotSource({}),
        block_size_bytes=2_000_000,
    )
    cache_block(controller, pool, 1, "hA")
    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(controller, pool, 1, "hA")
    assert controller.movement.remat_mb == 2.0
    assert "remat_mb=2.0 " in controller.hbm_summary()


def test_movement_in_mb_is_zero_when_the_page_size_is_unknown():
    """Reported rather than guessed: a fabricated byte count is worse than an
    absent one when the point is comparing two runs."""
    pool = FakePool(cached=False)
    controller = make_controller(pool)
    cache_block(controller, pool, 1, "hA")
    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(controller, pool, 1, "hA")
    assert controller.movement.remat_blocks == 1
    assert controller.movement.remat_mb == 0.0


def test_the_evicted_ring_is_bounded():
    """The ring is what bounds the memory, so its size has to actually bite."""
    pool = FakePool(num_blocks=32, cached=False)
    controller = make_controller(pool, remat_window_blocks=4)
    for i in range(1, 11):
        cache_block(controller, pool, i, f"h{i}")
        controller.on_block_evicted(i)
        pool.blocks[i].reset_hash()

    cache_block(controller, pool, 1, "h1")
    assert controller.movement.remat_blocks == 0, "h1 aged out of the ring"
    cache_block(controller, pool, 10, "h10")
    assert controller.movement.remat_blocks == 1


def test_remat_ratio_is_over_all_caching_work():
    pool = FakePool(cached=False)
    controller = make_controller(pool)
    for i in (1, 2, 3):
        cache_block(controller, pool, i, f"h{i}")
    controller.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(controller, pool, 1, "h1")
    # 4 blocks cached, 1 of them a rebuild.
    assert controller.movement.remat_ratio == 0.25


# -- treating a sentinel row as an absent one (opt-in) --------------------
#
# From the 2026-08-07 run: 80% of scored evictions ran on the prediction
# engine's extreme values, and the score collapsed to `constant / blocks`.
# The gate that would treat those as absent is **off by default** — the same
# run showed the low-end sentinel marks genuinely dead keys — so these tests
# opt in explicitly.

GATE_ON = dict(
    uninformative_prob_at_or_below=0.01,
    uninformative_ttnc_at_or_above_ms=3_600_000.0,
)


def test_a_sentinel_forecast_leaves_the_key_unscored():
    """`prob=0.01, ttnc=1h` is the engine saying "no idea". Acting on it
    ranks by `1/blocks` and nothing else."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {RESEARCH: dict(prob=0.01, time_to_next_call_ms=3_600_000.0)}
    )
    controller = make_controller(pool, rows, **GATE_ON)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3])
    before = pool.queue_ids()
    controller.maybe_tick()

    assert controller.get_value(RESEARCH) is None
    assert pool.queue_ids() == before, "unscored blocks keep LRU order"
    assert controller.observer.counters.blocks_spliced_total == 0


def test_a_real_forecast_is_still_scored():
    """The gate must not swallow the signal it exists to protect."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {RESEARCH: dict(prob=0.45, time_to_next_call_ms=8_000.0)}
    )
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3])
    controller.maybe_tick()
    assert controller.get_value(RESEARCH) is not None


@pytest.mark.parametrize(
    "row",
    [
        dict(prob=0.01, time_to_next_call_ms=5_000.0),      # prob gate
        dict(prob=0.9, time_to_next_call_ms=3_600_000.0),   # horizon gate
    ],
)
def test_each_half_of_the_gate_fires_on_its_own(row):
    pool = FakePool(num_blocks=16)
    controller = make_controller(pool, fresh_rows({RESEARCH: row}), **GATE_ON)
    index_prefix(controller, pool, RESEARCH, [1])
    controller.maybe_tick()
    assert controller.get_value(RESEARCH) is None


def test_the_gate_is_off_by_default():
    """Measured: the low-end sentinel marked genuinely dead keys (24.3%
    needed again vs 54.7% for the saturated class), so gating it out would
    discard real signal. Off unless a deployment shows otherwise."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {RESEARCH: dict(prob=0.01, time_to_next_call_ms=3_600_000.0)}
    )
    controller = make_controller(pool, rows)   # defaults = gate disabled
    index_prefix(controller, pool, RESEARCH, [1])
    controller.maybe_tick()
    assert controller.get_value(RESEARCH) is not None


def test_config_rejects_a_gate_above_the_probability_range():
    with pytest.raises(ValueError):
        NodeEvictionConfig(uninformative_prob_at_or_below=1.5).validate()


# -- splice churn ---------------------------------------------------------


def force_tick(controller):
    """Run a tick that would otherwise short-circuit.

    `maybe_tick` returns early when the snapshot object is unchanged *and*
    nothing was evicted since the last tick, so a test that wants a genuine
    second splice has to defeat that gate explicitly — otherwise it passes
    because no tick ran, which proves nothing about the splice.
    """
    controller._evictions_since_tick = 1
    controller.maybe_tick()


def test_a_parked_block_is_not_respliced():
    """2,054,400 relocations produced 47,363 evictions in the measured run.
    A block already at the head and untouched has nowhere better to go."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=600_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3])

    controller.maybe_tick()
    first = controller.observer.counters.blocks_spliced_total
    assert first == 3

    force_tick(controller)
    force_tick(controller)
    assert controller.observer.counters.blocks_spliced_total == first, (
        "nothing changed, so the splice must be a no-op"
    )


def test_a_touched_block_becomes_splicable_again():
    """The skip means "already parked at the head". A cache hit takes the
    block out of the queue, so that stops being true and the bookkeeping has
    to be dropped — otherwise the block is skipped forever."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=600_000.0)})
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3])
    controller.maybe_tick()
    spliced = controller.observer.counters.blocks_spliced_total
    assert 1 in controller._spliced_scores

    controller.on_prefix_hit(make_request(), [[pool.blocks[1]]])
    assert 1 not in controller._spliced_scores

    force_tick(controller)
    assert controller.observer.counters.blocks_spliced_total == spliced + 1


def test_the_staged_region_grows_deeper_than_one_ticks_cap():
    """Window 1. The policy governs exactly the prefix of the free queue it
    has staged; past that, `popleft_n` evicts by raw LRU age. With the cap
    read as a fixed K the staged region never exceeded K, so a request
    needing more blocks than that popped straight through into unordered
    territory. The cap is a catch-up rate, and depth accumulates across
    ticks."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=600_000.0)})
    controller = make_controller(
        pool, rows, splice_max_blocks=2, splice_restage_period_ms=0.0
    )
    index_prefix(controller, pool, RESEARCH, [1, 2, 3, 4, 5, 6])

    controller.maybe_tick()
    assert controller.observer.counters.splice_staged_blocks == 2

    force_tick(controller)
    force_tick(controller)
    assert controller.observer.counters.splice_staged_blocks == 6, (
        "depth must accumulate, not reset to the per-tick cap"
    )


def test_the_deficit_reports_blocks_left_in_raw_lru_order():
    """The window-1 gauge. A block the policy wanted to rank and could not
    is evicted by age, so a sustained deficit means the scores are not
    governing evictions however good they are."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=600_000.0)})
    controller = make_controller(
        pool, rows, splice_max_blocks=2, splice_restage_period_ms=0.0
    )
    index_prefix(controller, pool, RESEARCH, [1, 2, 3, 4, 5])

    controller.maybe_tick()
    assert controller.observer.counters.splice_deficit_blocks == 3

    force_tick(controller)
    force_tick(controller)
    assert controller.observer.counters.splice_deficit_blocks == 0


def test_restaging_re_sorts_the_whole_region():
    """`appendleft_n` puts each tick's batch in front of the previous one, so
    without re-staging the head is a stack of batches — newest first, each
    internally sorted, globally unsorted. At two blocks deep that is
    invisible; at thousands it inverts the policy."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.01, time_to_next_call_ms=600_000.0),
            SUPERVISOR: dict(prob=0.9, time_to_next_call_ms=1_000.0),
        }
    )
    controller = make_controller(pool, rows, splice_restage_period_ms=0.0)

    # research is the worse key and is staged first.
    index_prefix(controller, pool, RESEARCH, [1, 2])
    controller.maybe_tick()
    # supervisor is better, and lands in front of it.
    index_prefix(controller, pool, SUPERVISOR, [4, 5])
    controller.snapshot_source.set_rows(rows)
    force_tick(controller)
    assert set(pool.queue_ids()[:2]) == {4, 5}, "the inversion this fixes"

    # Now allow a re-stage: one pass re-sorts both batches together.
    controller.config.splice_restage_period_ms = 1.0
    controller._last_restage_monotonic = 0.0
    controller.snapshot_source.set_rows(rows)
    force_tick(controller)

    assert set(pool.queue_ids()[:2]) == {1, 2}, (
        "the worse key must be first out once the region is re-sorted"
    )
    assert controller.observer.counters.splice_restages_total >= 1


def test_restaging_can_be_switched_off():
    """0 restores the batch-stack behaviour, so the two can be A/B'd."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=600_000.0)})
    controller = make_controller(pool, rows, splice_restage_period_ms=0.0)
    index_prefix(controller, pool, RESEARCH, [1, 2, 3])

    controller.maybe_tick()
    spliced = controller.observer.counters.blocks_spliced_total
    for _ in range(5):
        force_tick(controller)

    assert controller.observer.counters.blocks_spliced_total == spliced
    assert controller.observer.counters.splice_restages_total == 0


def test_a_newly_worse_key_still_reaches_the_head():
    """The skip must not freeze the ranking: a key that appears after the
    first splice and scores lower still has to reach the head."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.9, time_to_next_call_ms=600_000.0),
            SUPERVISOR: dict(prob=0.9, time_to_next_call_ms=1_000_000.0),
        }
    )
    controller = make_controller(pool, rows)
    index_prefix(controller, pool, RESEARCH, [1])
    controller.maybe_tick()
    assert pool.queue_ids()[0] == 1

    index_prefix(controller, pool, SUPERVISOR, [4, 5])
    force_tick(controller)
    assert pool.queue_ids()[0] in (4, 5), "the worse key must overtake"


def _ttft_request(arrival: float, first_token: float | None, **kw):
    req = make_request(**kw)
    req.arrival_time = arrival
    req.first_token_ts = first_token
    req.ttft_recorded = False
    return req


def _fields(line: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in line.split() if "=" in p)


def test_ttft_is_recorded_once_per_request():
    """`KVCacheManager.free` also runs on preemption, and a preempted request
    keeps its original `first_token_ts`. Emitting it twice would weight slow
    requests by how often they were preempted."""
    controller = make_controller(FakePool())
    req = _ttft_request(arrival=1000.0, first_token=1000.25)

    controller.on_request_finished(req)
    controller.on_request_finished(req)

    assert controller.ttft.count == 1


def test_a_request_with_no_first_token_contributes_nothing():
    """Aborted before prefill finished — there is no prefill latency to
    attribute, and a zero would read as an instant response."""
    controller = make_controller(FakePool())
    controller.on_request_finished(_ttft_request(1000.0, None))
    assert controller.ttft.count == 0


def test_the_per_request_line_carries_the_raw_latency():
    """The whole point of the per-request line: the number in it is one
    request's own TTFT, not a mean anything was folded into."""
    controller = make_controller(FakePool())
    req = _ttft_request(1000.0, 1000.5, request_id="req-7")
    controller.on_cache_query(num_tokens=1536, num_hits=1024, request=req)

    line = controller._ttft_line(req, 500.0)
    fields = _fields(line)

    assert line.startswith("kv_hbm_ttft ")
    assert fields["req"] == "req-7"
    assert float(fields["ttft_ms"]) == 500.0
    assert fields["query_tokens"] == "1536"
    assert fields["hit_tokens"] == "1024"
    assert fields["external_hit_tokens"] == "0"
    assert fields["cold_tokens"] == "512"
    assert fields["preempted"] == "0"
    assert fields["phantom"] == "0"


def test_the_per_request_line_prefers_the_external_tier_figures():
    """`on_external_cache_query` sees the same lookup one tier later, so its
    numbers replace the local-only stash rather than adding to it — otherwise
    the local hit would be counted twice and `cold_tokens` under-reported."""
    controller = make_controller(FakePool())
    req = _ttft_request(1000.0, 1000.4)
    controller.on_cache_query(num_tokens=1000, num_hits=200, request=req)
    controller.on_external_cache_query(
        num_tokens=1000, num_local_hits=200, num_external_hits=500, request=req
    )

    fields = _fields(controller._ttft_line(req, 400.0))
    assert fields["hit_tokens"] == "200"
    assert fields["external_hit_tokens"] == "500"
    assert fields["cold_tokens"] == "300"


def test_the_per_request_line_marks_an_unmeasured_lookup():
    """A request that never reached `get_computed_blocks` has no breakdown.
    Reporting -1 keeps it from reading as a genuine all-cold prefill, which
    is what a 0 here would look like."""
    controller = make_controller(FakePool())
    fields = _fields(controller._ttft_line(_ttft_request(1000.0, 1000.1), 100.0))
    assert fields["query_tokens"] == "-1"
    assert fields["cold_tokens"] == "-1"


def test_a_phantom_prefetch_is_flagged_not_hidden():
    """Phantoms are out of every rate on the `kv_hbm` line, but their latency
    is real work the policy originated. Dropping it would make origination
    look free; the flag is what lets it be filtered out afterwards."""
    controller = make_controller(FakePool())
    req = _ttft_request(1000.0, 1000.2, prefetch_only=True)
    assert _fields(controller._ttft_line(req, 200.0))["phantom"] == "1"


def test_the_summary_line_carries_only_the_ttft_count():
    """The latencies live on their own lines. A mean here would be fixed to a
    window nobody chose."""
    controller = make_controller(FakePool(), hbm_summary_period_ms=10_000.0)
    controller.on_request_finished(_ttft_request(1000.0, 1000.5))
    line = controller._maybe_log_hbm_summary(1_000_000.0)
    fields = _fields(line)

    assert fields["ttft_n"] == "1"
    assert fields["ttft_win_n"] == "1"
    for gone in ("ttft_ms", "ttft_win_ms", "ttft_p50_ms", "ttft_p95_ms"):
        assert gone not in fields, f"{gone} should be off the kv_hbm line"


def _captured_ttft_lines(monkeypatch, controller, request) -> list[str]:
    """`vllm`'s root logger sets `propagate=False`, so `caplog` sees nothing
    here — the module logger has to be intercepted directly."""
    from vllm.v1.core.node_eviction import controller as controller_module

    lines: list[str] = []
    monkeypatch.setattr(
        controller_module.logger,
        "info",
        lambda fmt, *args: lines.append(fmt % args if args else fmt),
    )
    controller.on_request_finished(request)
    return [ln for ln in lines if ln.startswith("kv_hbm_ttft ")]


def test_finishing_a_request_writes_its_own_line(monkeypatch):
    controller = make_controller(FakePool())
    req = _ttft_request(1000.0, 1000.5, request_id="req-9")
    lines = _captured_ttft_lines(monkeypatch, controller, req)

    assert len(lines) == 1
    fields = _fields(lines[0])
    assert fields["req"] == "req-9"
    assert float(fields["ttft_ms"]) == 500.0


def test_the_per_request_line_can_be_turned_off(monkeypatch):
    """One line per request is real cost under load; the count on the
    `kv_hbm` line must not depend on the switch."""
    controller = make_controller(FakePool(), ttft_per_request_log=False)
    req = _ttft_request(1000.0, 1000.5)
    lines = _captured_ttft_lines(monkeypatch, controller, req)

    assert lines == []
    assert controller.ttft.count == 1


def test_a_backwards_clock_writes_no_line(monkeypatch):
    """A negative latency in the log is worse than a gap in it, and the
    sample must not be counted either — `ttft_n` would then stand over a
    sample nothing can be found for."""
    controller = make_controller(FakePool())
    req = _ttft_request(arrival=1000.0, first_token=999.5)
    lines = _captured_ttft_lines(monkeypatch, controller, req)

    assert lines == []
    assert controller.ttft.count == 0


# -- observe-only (the A/B baseline arm) ----------------------------------


def test_observe_only_reports_itself_as_the_baseline_variant():
    """`VLLM_NODE_EVICTION_POLICY=0` gives upstream, not a measurable
    baseline: no controller means no `kv_hbm` line to diff against. Observe
    mode keeps the line and labels it so the two arms are distinguishable in
    the field the timeline tool keys on."""
    controller = make_controller(FakePool(num_blocks=16), observe_only=True)
    fields = dict(
        part.split("=", 1)
        for part in controller.hbm_summary().split()
        if "=" in part
    )
    assert fields["variant"] == "baseline"
    assert controller.stats()["observe_only"] is True


def test_a_disabled_splice_is_labelled_baseline_too():
    """`splice_max_blocks: 0` in the JSON is behaviourally the same arm, and
    used to report itself as `node_eviction`."""
    controller = make_controller(FakePool(num_blocks=16), splice_max_blocks=0)
    assert "variant=baseline " in controller.hbm_summary()


def test_observe_only_never_reorders_the_queue():
    """The whole claim of the arm. Same rows that make the policy splice in
    `test_the_worst_scoring_block_is_moved_to_the_head`, but nothing moves."""
    pool = FakePool(num_blocks=16)
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.01, time_to_next_call_ms=600_000.0),
            SUPERVISOR: dict(prob=0.99, time_to_next_call_ms=100.0),
        }
    )
    controller = make_controller(pool, rows, observe_only=True)
    index_prefix(controller, pool, RESEARCH, [1])
    index_prefix(controller, pool, SUPERVISOR, [2])

    before = pool.queue_ids()
    controller.maybe_tick()

    assert pool.queue_ids() == before
    assert controller.observer.counters.splices_total == 0
    assert controller.observer.counters.blocks_spliced_total == 0
    # The counterfactual is still recorded: the tick scored both keys, so the
    # decision log says what the policy would have done.
    assert controller.get_value(RESEARCH) is not None
    assert controller.get_value(RESEARCH) < controller.get_value(SUPERVISOR)


def test_observe_only_forces_the_switches_that_define_the_arm():
    """Set from the env last on purpose: a JSON file that could re-enable the
    splice underneath it would produce a baseline that silently is not one."""
    import os
    from unittest.mock import patch

    from vllm.v1.core.node_eviction.config import NodeEvictionConfig

    with patch.dict(
        os.environ,
        {
            "VLLM_NODE_EVICTION_OBSERVE": "1",
            "VLLM_NODE_EVICTION_POLICY": "0",
            "VLLM_NODE_EVICTION_PREFETCH_DRAIN": "1",
        },
    ):
        cfg = NodeEvictionConfig.from_env()

    assert cfg.observe_only is True
    # Forced on despite POLICY=0, or there would be no controller to observe.
    assert cfg.enabled is True
    assert cfg.splice_max_blocks == 0
    # A baseline that originates prefills is not a baseline.
    assert cfg.prefetch_wants_enabled is False
    cfg.validate()
