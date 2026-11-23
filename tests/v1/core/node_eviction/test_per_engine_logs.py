# SPDX-License-Identifier: Apache-2.0
"""Each data-parallel engine writes its own node-eviction logs.

Every DP engine core inherits the one environment, so without a per-engine
name both would append to the same decision and access logs.
"""

from vllm.v1.core.node_eviction import per_engine_log_path


def test_rank_zero_keeps_the_configured_name():
    assert per_engine_log_path("/cell/evictions.jsonl", 0) == "/cell/evictions.jsonl"


def test_other_ranks_get_a_suffix_beside_it():
    assert per_engine_log_path("/cell/evictions.jsonl", 1) == "/cell/evictions-dp1.jsonl"
    assert per_engine_log_path("/cell/accesses.jsonl", 3) == "/cell/accesses-dp3.jsonl"


def test_unset_stays_unset():
    assert per_engine_log_path(None, 1) is None
    assert per_engine_log_path("", 1) == ""
