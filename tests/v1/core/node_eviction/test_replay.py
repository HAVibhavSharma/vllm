# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The offline replay harness (08 §7).

The scenario below is 06 §2, the one LRU gets wrong: `research` is freed
*earlier* than `supervisor` and needed *sooner*, so age is anti-correlated
with need. With a forecast the policy should keep `research`; LRU cannot.
"""

import json

import pytest

from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.replay import (
    ReplayRequest,
    _Ablation,
    compare,
    load_trace,
    replay,
)
from vllm.v1.core.node_eviction.types import NodeKey

RESEARCH = NodeKey("run-42", "research", "tavily:summary")
SUPERVISOR = NodeKey("run-42", "supervisor", "plan")
INTRUDER = NodeKey("run-99", "bulk", "fill")


def request(key, arrival, duration=0.5, tokens=1600):
    return ReplayRequest(
        request_id=f"{key.node}-{arrival}",
        key=key,
        arrival_ts=arrival,
        finish_ts=arrival + duration,
        num_prompt_tokens=tokens,
    )


def walkthrough_trace(rounds: int = 6) -> list[ReplayRequest]:
    """`research` cycles every ~9s around an 8s tool call; `supervisor`
    every ~60s; an unrelated job periodically demands the whole pool."""
    trace: list[ReplayRequest] = []
    t = 0.0
    for i in range(rounds):
        trace.append(request(RESEARCH, t))
        trace.append(request(SUPERVISOR, t + 1.0))
        trace.append(request(INTRUDER, t + 5.0, tokens=1600))
        trace.append(request(RESEARCH, t + 9.0))
        t += 60.0
    trace.sort(key=lambda r: r.arrival_ts)
    return trace


def test_replay_runs_with_no_gpu_model_or_redis():
    result = replay(walkthrough_trace(), policy="lru", num_blocks=256, block_size=16)
    assert result.requests > 0
    assert result.blocks_needed > 0


def test_both_arms_run_on_the_identical_trace():
    """The counterfactual LRU arm is the only thing that can answer 'is this
    worse than doing nothing' (07 §8)."""
    trace = walkthrough_trace()
    config = NodeEvictionConfig(enabled=True, splice_max_blocks=512)
    lru, scored = compare(
        trace,
        num_blocks=256,
        block_size=16,
        config=config,
        forecast_mode="oracle",
        ablation=_Ablation(),
    )
    assert lru.policy == "lru"
    assert scored.policy == "scored"
    assert lru.requests == scored.requests
    assert lru.blocks_needed == scored.blocks_needed


def test_a_perfect_forecast_does_not_lose_to_lru():
    """The oracle arm measures the *mechanism* with a perfect forecast, i.e.
    the ceiling of the design. If it loses to LRU here, the enforcement path
    is broken and no amount of prediction work will help."""
    trace = walkthrough_trace(rounds=8)
    config = NodeEvictionConfig(enabled=True, splice_max_blocks=512)
    lru, scored = compare(
        trace,
        num_blocks=256,
        block_size=16,
        config=config,
        forecast_mode="oracle",
        ablation=_Ablation(),
    )
    assert scored.block_hit_rate >= lru.block_hit_rate


def test_ablations_parse():
    assert _Ablation.parse("none") == _Ablation(True, True, True)
    assert _Ablation.parse("prob").prob is False
    assert _Ablation.parse("prob,decay,blocks") == _Ablation(False, False, False)


def test_ablation_reaches_the_score():
    """The falsification harness: it is entirely possible that one term
    carries everything and the rest are decoration — better to learn that
    from a sweep than to ship three terms and tune all of them.

    What a unit test can assert is that the ablation is **plumbed**, i.e.
    that turning a term off actually changes the scores the splice ranks on.
    Whether a term is load-bearing *for hit rate* is the empirical question
    the sweep exists to answer, and asserting an answer here would prejudge
    it — see `test_a_uniform_workload_gives_decay_nothing_to_separate`.
    """
    trace = walkthrough_trace()
    config = NodeEvictionConfig(enabled=True, splice_max_blocks=512)
    full = replay(
        trace,
        policy="scored",
        num_blocks=256,
        block_size=16,
        config=config,
        forecast_mode="oracle",
    )
    without_decay = replay(
        trace,
        policy="scored",
        num_blocks=256,
        block_size=16,
        config=config,
        forecast_mode="oracle",
        ablation=_Ablation(decay=False),
    )
    assert full.needed_score_sum != without_decay.needed_score_sum
    assert full.not_needed_score_sum != without_decay.not_needed_score_sum


def test_a_uniform_workload_gives_decay_nothing_to_separate():
    """Records what the fixture can and cannot show, so it is not rediscovered
    as a bug.

    Under `oracle`, every key that still recurs gets `prob=1.0`; the fixture
    gives all three keys the same size, so `blocks` is constant; and with the
    default hit-class assumption `E_miss` is constant too. **Decay is the only
    term that varies here.** Ablate it and every live key collapses to one
    identical score, the splice ties everywhere, and Rule 3's tail-first
    position tie-break decides alone — which on a workload this uniform and
    this oversubscribed (3 x 100 blocks against a 256-block pool) selects the
    same blocks anyway.

    So the outcome is unchanged while the scores are not. That is a property
    of the fixture, not a broken ablation: a trace where the terms can
    actually separate needs keys of differing size or differing recurrence.
    """
    trace = walkthrough_trace()
    config = NodeEvictionConfig(enabled=True, splice_max_blocks=512)
    arms = {
        name: replay(
            trace,
            policy="scored",
            num_blocks=256,
            block_size=16,
            config=config,
            forecast_mode="oracle",
            ablation=ablation,
        )
        for name, ablation in (
            ("full", None),
            ("no_decay", _Ablation(decay=False)),
        )
    }
    full, no_decay = arms["full"], arms["no_decay"]

    # The scores moved...
    assert full.needed_score_sum != no_decay.needed_score_sum
    # ...and with decay gone the live keys are indistinguishable: every
    # regretted eviction carries the identical score, and the only other
    # value in the run is that score scaled by the 0.01 "never again" floor.
    live = no_decay.needed_score_sum / no_decay.needed_count
    dead = no_decay.not_needed_score_sum / no_decay.not_needed_count
    assert dead == pytest.approx(live * 0.01, rel=1e-9)
    # ...so the outcome cannot move. Asserting otherwise is what made this
    # test fail for a reason that was never in the production code.
    assert full.block_hit_rate == no_decay.block_hit_rate
    assert full.evictions == no_decay.evictions


def test_causal_forecast_needs_no_lookahead():
    trace = walkthrough_trace()
    result = replay(
        trace,
        policy="scored",
        num_blocks=256,
        block_size=16,
        forecast_mode="history",
    )
    assert result.requests == len(trace)


def test_load_trace_reads_filestatlogger_jsonl(tmp_path):
    path = tmp_path / "finished_requests_engine0_x.jsonl"
    rows = [
        {
            "request_id": "a",
            "job_id": "run-42",
            "langgraph_node": "research",
            "call_type": "tavily:summary",
            "arrival_ts": 100.0,
            "finish_ts": 100.5,
            "num_prompt_tokens": 23900,
            "num_local_cached_tokens": 23000,
            "num_external_cached_tokens": 0,
        },
        {
            "request_id": "b",
            "job_id": "run-42",
            "langgraph_node": "supervisor",
            "call_type": "plan",
            "arrival_ts": 50.0,
            "e2e_latency": 0.4,
            "num_prompt_tokens": 100,
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{partial")

    trace = load_trace([str(path)])
    # Sorted by arrival, partial trailing line tolerated.
    assert [r.request_id for r in trace] == ["b", "a"]
    # finish_ts derived from e2e_latency when absent.
    assert trace[0].finish_ts == 50.4
    assert trace[1].observed_class == "gpu"


def test_records_without_identity_are_skipped(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"request_id": "x", "arrival_ts": 1.0}) + "\n")
    assert load_trace([str(path)]) == []


def test_hit_class_from_the_engines_own_fields():
    """The pair that makes the L1-residency question answerable at all,
    with no new instrumentation anywhere (04 §5.4)."""
    assert request(RESEARCH, 0).observed_class == "cold"
    r = request(RESEARCH, 0)
    r.num_external_cached_tokens = 10
    assert r.observed_class == "l1"
    r.num_local_cached_tokens = 10
    assert r.observed_class == "gpu"
