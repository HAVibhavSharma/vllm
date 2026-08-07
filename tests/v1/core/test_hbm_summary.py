# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The baseline half of the HBM accounting line.

These tests pin the two things that make the branch comparison valid: the
line's *shape* (so a field-by-field diff against the policy branch lines up)
and the attribution of each eviction to a `job_id:node`.
"""

from types import SimpleNamespace

from vllm.v1.core.hbm_summary import HBMSummaryLogger, node_label_for_request
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

# The field order the policy branch emits. A baseline line that drops or
# reorders one of these breaks the comparison silently — the numbers still
# parse, they just stop lining up.
EXPECTED_FIELDS = [
    "variant",
    "total",
    "used",
    "free",
    "usage",
    "queue",
    "splices",
    "spliced_blocks",
    "evicted",
    "evicted_by_score",
    "regret",
    "hit_rate",
    "hit_rate_win",
    "hit_tokens",
    "query_tokens",
    "remat_blocks",
    "remat_mb",
    "remat_ratio",
    "blocks_cached",
    "index_keys",
    "index_blocks",
    "top_evicted",
]


class FakePool:
    def __init__(self, num_blocks: int = 16):
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.num_gpu_blocks = num_blocks
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks


def make_request(job_id="run-42", node="research"):
    extra_args = {}
    if job_id is not None:
        extra_args["job_id"] = job_id
    if node is not None:
        extra_args["langgraph_node"] = node
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))


def make_logger(pool, period_ms=10_000.0, top_keys=5):
    return HBMSummaryLogger(pool, period_ms=period_ms, top_keys=top_keys)


def parse(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def test_identity_is_read_from_extra_args():
    assert node_label_for_request(make_request()) == "run-42:research"
    assert node_label_for_request(make_request(job_id=None)) is None
    assert node_label_for_request(SimpleNamespace(sampling_params=None)) is None


def test_line_has_the_same_fields_as_the_policy_branch():
    """The whole reason this module exists: the two runs must diff."""
    pool = FakePool()
    line = make_logger(pool).summary()
    assert list(parse(line)) == EXPECTED_FIELDS


def test_policy_only_fields_are_zero_not_missing():
    """LRU never reorders the queue, but the fields still have to be there
    or a naive field-by-field diff misaligns."""
    fields = parse(make_logger(FakePool()).summary())
    assert fields["variant"] == "baseline"
    assert fields["splices"] == "0"
    assert fields["spliced_blocks"] == "0"


def test_summary_reports_the_pool_split():
    pool = FakePool(num_blocks=16)
    logger = make_logger(pool)
    pool.blocks[3].ref_cnt = 1
    pool.free_block_queue.remove(pool.blocks[3])

    fields = parse(logger.summary())
    total, used, free = (
        int(fields["total"]),
        int(fields["used"]),
        int(fields["free"]),
    )
    assert total == 16
    assert used + free == total
    assert used == 1
    assert int(fields["queue"]) == free


def test_eviction_is_attributed_to_the_node_that_owned_the_block():
    pool = FakePool()
    logger = make_logger(pool)
    logger.on_blocks_cached(
        make_request(node="research"), [pool.blocks[1], pool.blocks[2]]
    )
    logger.on_blocks_cached(make_request(node="supervisor"), [pool.blocks[4]])

    logger.on_block_evicted(1)
    logger.on_block_evicted(2)
    logger.on_block_evicted(4)

    fields = parse(logger.summary())
    assert fields["top_evicted"] == "run-42:research=2,run-42:supervisor=1"
    assert fields["evicted"] == "3"


def test_unowned_blocks_are_counted_separately_not_dropped():
    """Otherwise `evicted=` and `top_evicted=` stop adding up."""
    logger = make_logger(FakePool())
    logger.on_block_evicted(7)
    fields = parse(logger.summary())
    assert fields["top_evicted"] == "<untracked>=1"
    assert fields["evicted"] == "1"


def test_ownership_is_released_on_eviction():
    """The tracking dict must not outgrow the pool."""
    pool = FakePool()
    logger = make_logger(pool)
    logger.on_blocks_cached(make_request(), [pool.blocks[1]])
    assert logger.stats()["index_blocks"] == 1
    logger.on_block_evicted(1)
    assert logger.stats()["index_blocks"] == 0


def test_reset_prefix_cache_drops_claims_without_counting_evictions():
    """An operator resetting the cache must not be charged to whichever job
    happened to be resident."""
    pool = FakePool()
    logger = make_logger(pool)
    logger.on_blocks_cached(make_request(), [pool.blocks[1], pool.blocks[2]])
    logger.on_reset_prefix_cache()
    assert logger.stats()["index_blocks"] == 0
    assert logger.evictions_total == 0


def test_line_is_rate_limited_and_change_gated():
    pool = FakePool()
    logger = make_logger(pool, period_ms=10_000.0)

    # One line at startup, which is where `total=` comes from.
    assert logger.maybe_log(1000.0) is not None
    # Same window.
    assert logger.maybe_log(1001.0) is None
    # New window, nothing moved.
    assert logger.maybe_log(500_000.0) is None

    logger.on_blocks_cached(make_request(), [pool.blocks[1]])
    assert logger.maybe_log(600_000.0) is not None
    # New window, nothing moved.
    assert logger.maybe_log(1_000_000.0) is None

    logger.on_block_evicted(1)
    assert logger.maybe_log(2_000_000.0) is not None


def test_attribution_is_per_window():
    pool = FakePool()
    logger = make_logger(pool, period_ms=10_000.0)
    logger.on_blocks_cached(make_request(node="research"), [pool.blocks[1]])
    logger.on_block_evicted(1)
    first = logger.maybe_log(1_000_000.0)
    assert "run-42:research=1" in first

    logger.on_blocks_cached(make_request(node="supervisor"), [pool.blocks[4]])
    logger.on_block_evicted(4)
    second = logger.maybe_log(2_000_000.0)
    assert "run-42:supervisor=1" in second
    assert "run-42:research" not in second


def test_top_keys_bounds_the_list():
    pool = FakePool(num_blocks=32)
    logger = make_logger(pool, top_keys=2)
    for i in range(1, 6):
        owned = [pool.blocks[j] for j in range(i * 3, i * 3 + i)]
        logger.on_blocks_cached(make_request(node=f"n{i}"), owned)
        for j in range(i * 3, i * 3 + i):
            logger.on_block_evicted(j)
    assert parse(logger.summary())["top_evicted"] == "run-42:n5=5,run-42:n4=4"


def test_logger_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("VLLM_HBM_SUMMARY_PERIOD_MS", "0")
    assert HBMSummaryLogger.maybe_build(FakePool()) is None


def test_logger_is_on_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HBM_SUMMARY_PERIOD_MS", raising=False)
    built = HBMSummaryLogger.maybe_build(FakePool())
    assert built is not None and built.period_ms == 30_000.0


# -- hit rate ------------------------------------------------------------


def test_hit_rate_is_tokens_not_requests():
    """A 10-token request that fully hit and a 10k-token one that missed are
    not one hit and one miss — the second cost a thousand times more."""
    logger = make_logger(FakePool())
    logger.on_cache_query(num_tokens=10, num_hits=10)
    logger.on_cache_query(num_tokens=10_000, num_hits=0)
    assert logger.hit_rate == 10 / 10_010
    assert float(parse(logger.summary())["hit_rate"]) == round(10 / 10_010, 4)


def test_windowed_hit_rate_resets_but_cumulative_does_not():
    """A cumulative rate over a long run is dominated by whatever the
    workload did first, so the line carries both."""
    pool = FakePool()
    logger = make_logger(pool, period_ms=10_000.0)
    logger.on_cache_query(num_tokens=100, num_hits=0)
    logger.maybe_log(1_000_000.0)

    logger.on_cache_query(num_tokens=100, num_hits=100)
    fields = parse(logger.maybe_log(2_000_000.0))
    assert float(fields["hit_rate_win"]) == 1.0
    assert float(fields["hit_rate"]) == 0.5


def test_hit_rate_is_zero_before_any_query():
    """Not a crash, and not NaN — NaN poisons every downstream average."""
    assert float(parse(make_logger(FakePool()).summary())["hit_rate"]) == 0.0


# -- cache movement ------------------------------------------------------


def cache_block(logger, pool, block_id, tag, node="research"):
    block = pool.blocks[block_id]
    if block.block_hash is not None:
        block.reset_hash()
    block.block_hash = tag
    logger.on_blocks_cached(make_request(node=node), [block])


def test_a_first_time_prefix_is_not_movement():
    """The number must credit the policy only for work it made us redo. A
    prefix the server has never seen is not that."""
    pool = FakePool()
    logger = make_logger(pool)
    for i in (1, 2, 3):
        cache_block(logger, pool, i, f"h{i}")
    assert logger.remat_blocks == 0


def test_recaching_an_evicted_hash_counts_as_movement():
    pool = FakePool()
    logger = make_logger(pool)
    cache_block(logger, pool, 1, "hA")
    logger.on_block_evicted(1)
    pool.blocks[1].reset_hash()

    cache_block(logger, pool, 1, "hA")
    assert logger.remat_blocks == 1


def test_movement_needs_a_second_eviction_to_count_twice():
    """Otherwise every later cache of a popular prefix reads as movement and
    the number grows without anything being rebuilt."""
    pool = FakePool()
    logger = make_logger(pool)
    cache_block(logger, pool, 1, "hA")
    logger.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(logger, pool, 1, "hA")
    assert logger.remat_blocks == 1

    # Cached again with no eviction in between: not movement.
    pool.blocks[1].reset_hash()
    cache_block(logger, pool, 1, "hA")
    assert logger.remat_blocks == 1

    # A second eviction, then a second rebuild: movement again.
    logger.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(logger, pool, 1, "hA")
    assert logger.remat_blocks == 2


def test_the_evicted_ring_is_bounded():
    """The ring is what bounds the memory, so its size has to actually bite."""
    pool = FakePool(num_blocks=32)
    logger = HBMSummaryLogger(pool, period_ms=10_000.0, remat_window_blocks=4)
    for i in range(1, 11):
        cache_block(logger, pool, i, f"h{i}")
        logger.on_block_evicted(i)
        pool.blocks[i].reset_hash()

    assert len(logger._evicted_hashes) == 4
    # h1 aged out, so rebuilding it is no longer attributed to this policy.
    cache_block(logger, pool, 1, "h1")
    assert logger.remat_blocks == 0
    # h10 is still in the ring.
    cache_block(logger, pool, 10, "h10")
    assert logger.remat_blocks == 1


def test_reset_prefix_cache_is_not_movement():
    """An operator wiping the cache did not make the policy redo work."""
    pool = FakePool()
    logger = make_logger(pool)
    cache_block(logger, pool, 1, "hA")
    logger.on_block_evicted(1)
    pool.blocks[1].reset_hash()

    logger.on_reset_prefix_cache()
    cache_block(logger, pool, 1, "hA")
    assert logger.remat_blocks == 0


def test_movement_in_mb_uses_the_configured_page_size():
    pool = FakePool()
    logger = make_logger(pool)
    logger.configure(block_size_bytes=2_000_000)
    cache_block(logger, pool, 1, "hA")
    logger.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(logger, pool, 1, "hA")
    assert logger.remat_mb == 2.0
    assert parse(logger.summary())["remat_mb"] == "2.0"


def test_movement_in_mb_is_zero_when_the_page_size_is_unknown():
    """Reported rather than guessed: a fabricated byte count is worse than an
    absent one when the point is comparing two runs."""
    pool = FakePool()
    logger = make_logger(pool)
    cache_block(logger, pool, 1, "hA")
    logger.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(logger, pool, 1, "hA")
    assert logger.remat_blocks == 1
    assert logger.remat_mb == 0.0


def test_remat_ratio_is_over_all_caching_work():
    pool = FakePool()
    logger = make_logger(pool)
    for i in (1, 2, 3):
        cache_block(logger, pool, i, f"h{i}")
    logger.on_block_evicted(1)
    pool.blocks[1].reset_hash()
    cache_block(logger, pool, 1, "h1")
    # 4 blocks cached, 1 of them a rebuild.
    assert logger.remat_ratio == 0.25
