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
