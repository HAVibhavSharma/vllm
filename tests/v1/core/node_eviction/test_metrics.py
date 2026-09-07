# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observability (07). The one number is the score/outcome correlation: if
it is negative the policy is anti-correlated, i.e. actively worse than LRU,
and that is a failure with no other symptom."""

import json

from vllm.v1.core.node_eviction.metrics import EvictionObserver
from vllm.v1.core.node_eviction.types import NodeKey, ScoreBreakdown

RESEARCH = NodeKey("run-42", "research", "tavily:summary")
SUPERVISOR = NodeKey("run-42", "supervisor", "plan")


def breakdown(score: float, speculative: bool = False) -> ScoreBreakdown:
    return ScoreBreakdown(
        score=score,
        prob=0.5,
        ttnc_ms=1000.0,
        decay=0.5,
        e_miss_ms=180.0,
        blocks=100,
        speculative=speculative,
    )


def test_regret_is_counted_without_an_offline_join():
    """On admission the key is already in hand, so a hit against the recent
    ring buffer is an O(1) lookup (07 §6)."""
    observer = EvictionObserver(regret_window_ms=10_000.0)
    observer.record_eviction(1, [RESEARCH], breakdown(0.5), now_ms=1000.0)
    observer.record_admission(RESEARCH, now_ms=2000.0)

    assert observer.counters.evicted_then_needed_total == 1
    assert observer.counters.evicted_then_needed_cost_ms == 180.0


def test_an_eviction_outside_the_horizon_is_not_regret():
    observer = EvictionObserver(regret_window_ms=1_000.0)
    observer.record_eviction(1, [RESEARCH], breakdown(0.5), now_ms=1000.0)
    observer.record_admission(RESEARCH, now_ms=99_000.0)
    assert observer.counters.evicted_then_needed_total == 0


def test_correlation_is_positive_when_the_policy_is_right():
    """Blocks that were needed again should have scored *higher* than blocks
    that were not."""
    observer = EvictionObserver(regret_window_ms=10_000.0)
    # A high-scoring block we evicted and immediately needed back.
    observer.record_eviction(1, [RESEARCH], breakdown(10.0), now_ms=1000.0)
    observer.record_admission(RESEARCH, now_ms=1500.0)
    # A low-scoring block nobody wanted.
    observer.record_eviction(2, [SUPERVISOR], breakdown(0.1), now_ms=1000.0)
    observer.expire(now_ms=99_000.0)

    assert observer.counters.score_outcome_correlation > 0


def test_correlation_goes_negative_when_the_sign_is_inverted():
    """The failure the whole metric exists for: the policy kept the blocks
    nobody wanted and dropped the ones it needed."""
    observer = EvictionObserver(regret_window_ms=10_000.0)
    observer.record_eviction(1, [RESEARCH], breakdown(0.1), now_ms=1000.0)
    observer.record_admission(RESEARCH, now_ms=1500.0)
    observer.record_eviction(2, [SUPERVISOR], breakdown(10.0), now_ms=1000.0)
    observer.expire(now_ms=99_000.0)

    assert observer.counters.score_outcome_correlation < 0


def test_an_eviction_is_classified_exactly_once():
    """A record counted as 'needed' must not also be counted as 'not needed'
    when it ages out, or the correlation is computed against a corrupted
    negative class."""
    observer = EvictionObserver(regret_window_ms=1_000.0)
    observer.record_eviction(1, [RESEARCH], breakdown(5.0), now_ms=1000.0)
    observer.record_admission(RESEARCH, now_ms=1500.0)
    observer.expire(now_ms=99_000.0)

    assert observer.counters.needed_count == 1
    assert observer.counters.not_needed_count == 0


def test_unscored_evictions_are_tracked_separately():
    """A broken join is otherwise completely silent (03 §5)."""
    observer = EvictionObserver()
    observer.record_eviction(1, [], None, now_ms=1000.0)
    observer.record_eviction(2, [RESEARCH], breakdown(1.0), now_ms=1000.0,
                             was_spliced=True)
    assert observer.counters.unscored_evictions_total == 1
    assert observer.counters.unscored_eviction_ratio == 0.5


def test_speculative_waste_is_reported():
    """Without this ratio, a policy that protects garbage for a full TTL
    looks identical to one that works (02 §5)."""
    observer = EvictionObserver()
    observer.counters.speculative_blocks_created = 4
    observer.record_eviction(
        1, [RESEARCH], breakdown(1.0, speculative=True), now_ms=1000.0
    )
    assert observer.counters.speculative_evicted_before_confirm == 1
    assert observer.counters.speculative_waste == 0.25


def test_waste_is_a_fraction_of_blocks_not_of_predictions():
    """Both sides must be counted in blocks. Eviction happens one block at a
    time, so dividing evicted blocks by *keys* made one fully wasted 50-block
    prefix report 5000% waste — a ratio that cannot exceed 1 by construction
    reading as though the forecast were catastrophic."""
    observer = EvictionObserver()
    observer.counters.speculative_created = 1
    observer.counters.speculative_blocks_created = 50
    for block_id in range(50):
        observer.record_eviction(
            block_id, [RESEARCH], breakdown(1.0), now_ms=1000.0, speculative=True
        )
    assert observer.counters.speculative_waste == 1.0


def test_provenance_can_be_asserted_by_the_caller():
    """The controller reads `speculative` off the ownership index, which
    knows before any tick has scored the key. Falling back to the score
    breakdown alone drops a block prefetched and evicted inside one tick
    period into the unscored bucket — exactly the case the ratio exists to
    catch."""
    observer = EvictionObserver()
    observer.counters.speculative_blocks_created = 1
    # No breakdown at all: the key has never been scored.
    observer.record_eviction(1, [RESEARCH], None, now_ms=1000.0, speculative=True)
    assert observer.counters.speculative_evicted_before_confirm == 1


def test_decision_log_carries_the_terms_and_the_snapshot_age(tmp_path):
    """Logging the components, not just the total, is what makes a bad score
    diagnosable rather than merely visible. `snapshot_age_ms` separates a
    wrong forecast from a merely late one (07 §4, §7)."""
    path = tmp_path / "decisions.jsonl"
    observer = EvictionObserver(decision_log_path=str(path))
    observer.record_eviction(
        84213,
        [RESEARCH],
        breakdown(0.0141),
        now_ms=1753980000123.0,
        prefix_pos=1487,
        run_len=1600,
        rank_in_splice=12,
        snapshot_age_ms=340.0,
        num_free_fresh=0,
    )
    observer.close()

    line = json.loads(path.read_text().strip())
    assert line["block_id"] == 84213
    assert line["keys"] == [["run-42", "research", "tavily:summary"]]
    assert line["terms"]["blocks"] == 100
    assert line["snapshot_age_ms"] == 340.0
    # prefix_pos near run_len is what says tail-first is holding.
    assert line["prefix_pos"] == 1487
    assert line["run_len"] == 1600


def test_no_decision_log_by_default(tmp_path):
    """Per-eviction lines are unbounded under pressure, so they cannot be the
    production mechanism (07 §6)."""
    observer = EvictionObserver()
    observer.record_eviction(1, [RESEARCH], breakdown(1.0), now_ms=1.0)
    assert list(tmp_path.iterdir()) == []


def test_counters_serialise():
    observer = EvictionObserver()
    payload = observer.counters.as_dict()
    for field in (
        "evictions_by_score_total",
        "evicted_then_needed_total",
        "speculative_waste",
        "score_outcome_correlation",
        "policy_enabled",
    ):
        assert field in payload


def test_ttft_counts_samples_and_keeps_no_latency():
    """The tracker is a counter now. Every latency is emitted on its own
    `kv_hbm_ttft` line as it is taken, so nothing here should be holding a
    mean, a median or a ring of recent samples for one to be computed from —
    an aggregate fixed at write time cannot be re-cut to the subset of
    requests a question is actually about."""
    from vllm.v1.core.node_eviction.metrics import TTFTTracker

    t = TTFTTracker()
    for ms in range(1, 101):
        assert t.record(float(ms)) is True

    assert t.count == 100
    assert t.as_dict() == {"ttft_n": 100}
    for gone in ("mean_ms", "window_mean_ms", "p50_ms", "p95_ms", "_recent"):
        assert not hasattr(t, gone), f"{gone} should no longer exist"


def test_ttft_window_resets_but_cumulative_does_not():
    """Same split as the hit rate: the window count says how many samples the
    line just printed stands over, the cumulative one how many the epoch
    holds."""
    from vllm.v1.core.node_eviction.metrics import TTFTTracker

    t = TTFTTracker()
    t.record(100.0)
    t.reset_window()
    t.record(200.0)

    assert t.window_count == 1
    assert t.count == 2


def test_reset_measurement_drops_both_counts():
    """The warmup boundary. A count that survived it would make the first
    line of the measured epoch claim samples that belong to the warmup."""
    from vllm.v1.core.node_eviction.metrics import TTFTTracker

    t = TTFTTracker()
    t.record(100.0)
    t.reset_measurement()

    assert t.count == 0
    assert t.window_count == 0


def test_a_negative_ttft_is_rejected_not_logged():
    """A non-monotonic wall clock must not put a negative latency in the log —
    a missing sample is the lesser corruption. The `False` is what tells the
    caller not to write the line."""
    from vllm.v1.core.node_eviction.metrics import TTFTTracker

    t = TTFTTracker()
    assert t.record(-5.0) is False
    assert t.record(10.0) is True
    assert t.count == 1
