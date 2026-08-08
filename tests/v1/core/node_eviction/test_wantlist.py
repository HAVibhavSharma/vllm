# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build step 6: prefetch origination (02 §4 option a).

Two halves are tested here — the want-list structure, and the controller's
decision about what goes in it. The third half (the front-end drainer) is in
`tests/v1/agent_prefetch/test_drain.py`, because it needs no engine at all.

The bounds get as much attention as the happy path on purpose: a want costs a
real prefill whenever LMCache misses the prefix, so a duplicating or unbounded
want-list does not merely fail to help — it competes with the requests it was
meant to serve.
"""

from vllm.v1.core.node_eviction.types import ImportanceRow, NodeKey
from vllm.v1.core.node_eviction.wantlist import PrefetchWant, PrefetchWantList

from .test_controller import (
    RESEARCH,
    SUPERVISOR,
    FakePool,
    fresh_rows,
    make_controller,
    make_request,
)


def make_want(key: NodeKey, score: float = 1.0, created_at: float = 0.0):
    return PrefetchWant(
        key=key,
        agent_id=f"langgraph:{key.node}",
        score=score,
        prob=1.0,
        time_to_next_call_ms=1_000.0,
        created_at=created_at,
    )


# -- the structure -------------------------------------------------------


def test_a_key_is_not_offered_twice_while_pending():
    """The tick runs every 250ms and the drain every 1s, so without dedup
    every want would be offered four times before anyone acted on it."""
    wants = PrefetchWantList()
    assert wants.offer(make_want(RESEARCH)) is True
    assert wants.offer(make_want(RESEARCH)) is False
    assert wants.num_pending == 1


def test_a_key_is_not_re_offered_while_its_phantom_is_in_flight():
    """The window between handing a want over and its blocks appearing in the
    index is the whole LMCache load. Re-offering across it would submit the
    same phantom on every tick of that window."""
    wants = PrefetchWantList()
    wants.offer(make_want(RESEARCH))
    wants.drain(1, now=0.0)
    assert wants.num_pending == 0
    assert wants.num_outstanding == 1
    assert wants.offer(make_want(RESEARCH)) is False


def test_max_outstanding_bounds_pending_and_inflight_together():
    """The bound is on speculation in total, not per state — otherwise
    draining would free up slots to speculate again immediately."""
    wants = PrefetchWantList(max_outstanding=2)
    assert wants.offer(make_want(NodeKey("j", "a", ""))) is True
    assert wants.offer(make_want(NodeKey("j", "b", ""))) is True
    assert wants.offer(make_want(NodeKey("j", "c", ""))) is False

    wants.drain(1, now=0.0)
    assert wants.offer(make_want(NodeKey("j", "c", ""))) is False


def test_drain_takes_the_most_valuable_first():
    """A partial drain must take the wants worth the most expected ms, not
    whichever the dict happened to yield first."""
    wants = PrefetchWantList()
    wants.offer(make_want(NodeKey("j", "cheap", ""), score=1.0))
    wants.offer(make_want(NodeKey("j", "rich", ""), score=99.0))
    wants.offer(make_want(NodeKey("j", "mid", ""), score=50.0))

    taken = wants.drain(2, now=0.0)
    assert [w.key.node for w in taken] == ["rich", "mid"]
    assert wants.num_pending == 1


def test_satisfied_clears_both_states_and_allows_a_new_want():
    wants = PrefetchWantList()
    wants.offer(make_want(RESEARCH))
    wants.drain(1, now=0.0)

    assert wants.note_satisfied(RESEARCH) is True
    assert wants.num_outstanding == 0
    # A later job may legitimately want the same key again.
    assert wants.offer(make_want(RESEARCH)) is True


def test_a_pending_want_nobody_drained_goes_stale():
    """Handing a minute-old prediction to a drainer that just woke up would
    prefetch for a call that has already happened."""
    wants = PrefetchWantList(want_ttl_ms=10_000.0)
    wants.offer(make_want(RESEARCH, created_at=0.0))

    assert wants.expire(now=5.0) == (0, 0)
    assert wants.num_pending == 1
    assert wants.expire(now=11.0) == (1, 0)
    assert wants.num_pending == 0


def test_an_outstanding_want_is_released_after_the_backoff():
    """A phantom that was dropped, failed, or never landed must not hold one
    of `max_outstanding` slots for the process lifetime."""
    wants = PrefetchWantList(resubmit_backoff_ms=10_000.0)
    wants.offer(make_want(RESEARCH))
    wants.drain(1, now=0.0)

    assert wants.expire(now=5.0) == (0, 0)
    assert wants.expire(now=11.0) == (0, 1)
    assert wants.num_outstanding == 0
    assert wants.offer(make_want(RESEARCH)) is True


def test_the_wire_form_is_flat_scalars():
    """The front end submits phantoms and has no reason to import engine-core
    types, so nothing structured crosses the call_utility hop."""
    payload = make_want(RESEARCH, score=12.5).as_dict()
    assert payload == {
        "job_id": "run-42",
        "node": "research",
        "call_type": "tavily:summary",
        "agent_id": "langgraph:research",
        "score": 12.5,
        "prob": 1.0,
        "time_to_next_call_ms": 1_000.0,
    }


# -- the decision --------------------------------------------------------


def wants_controller(pool, rows=None, **overrides):
    overrides.setdefault("prefetch_wants_enabled", True)
    return make_controller(pool, rows, **overrides)


def test_origination_is_off_unless_asked_for():
    """Reordering the free queue is free; originating a prefetch buys prefill
    work. The two switches are separate and this one defaults off."""
    controller = make_controller(FakePool())
    controller.maybe_tick()
    assert controller.drain_prefetch_wants(4) == []
    assert controller.stats()["prefetch_origination_enabled"] is False


def test_a_predicted_node_absent_from_hbm_becomes_a_want():
    """The diagram's rule: assess the value, check whether it is in HBM, and
    if not, issue a prefetch call."""
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(FakePool(), rows)

    controller.maybe_tick()

    drained = controller.drain_prefetch_wants(4)
    assert len(drained) == 1
    assert drained[0]["agent_id"] == "langgraph:research"
    assert drained[0]["node"] == "research"
    assert controller.observer.counters.prefetch_wants_created == 1


def test_a_prefix_already_in_hbm_is_not_prefetched():
    """`get_entry(key) is not None` *is* the HBM presence check — the index
    holds exactly the keys with blocks resident right now."""
    pool = FakePool()
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(pool, rows)
    controller.on_blocks_cached(make_request(), pool.blocks[:3], 0)

    controller.maybe_tick()

    assert controller.drain_prefetch_wants(4) == []


def test_an_arriving_prefix_retires_an_outstanding_want():
    """Retirement keys off the index rather than the phantom's completion, so
    it works no matter who put the prefix there."""
    pool = FakePool()
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(pool, rows)
    controller.maybe_tick()
    assert len(controller.drain_prefetch_wants(4)) == 1

    controller.on_blocks_cached(make_request(prefetch_only=True), pool.blocks[:3], 0)
    controller.snapshot_source.set_rows(rows)  # force a fresh snapshot object
    controller.maybe_tick()

    counters = controller.observer.counters
    assert counters.prefetch_wants_satisfied == 1
    assert controller.stats()["prefetch_wants_outstanding"] == 0


def test_an_unlikely_call_is_not_worth_a_prefill():
    """The gate still works — it is just switched off by default now."""
    rows = fresh_rows({RESEARCH: dict(prob=0.1, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(FakePool(), rows, prefetch_min_prob=0.5)
    controller.maybe_tick()
    assert controller.drain_prefetch_wants(4) == []


def test_a_distant_call_is_real_but_not_imminent():
    """Warming a node predicted ten minutes out just evicts something needed
    sooner. The diagram gates admission on the need being *imminent*.

    Also opt-in now: `time_to_next_call_ms` is 60s or 3600s and nothing
    between (12 §2), so as a default the gate splits on the forecast's broken
    binary rather than on imminence."""
    rows = fresh_rows({RESEARCH: dict(prob=0.99, time_to_next_call_ms=600_000.0)})
    controller = wants_controller(FakePool(), rows, prefetch_horizon_ms=60_000.0)
    controller.maybe_tick()
    assert controller.drain_prefetch_wants(4) == []


def test_neither_gate_fires_by_default():
    """The reason the first real run produced 96 wants in 74 minutes. With a
    forecast that is 0.01 or 1.0 and nothing in between, a `prob >= 0.5` gate
    is not selecting likely rows — it is selecting one arm of a binary
    signal, and the other arm is where the come-back rate is highest."""
    rows = fresh_rows(
        {RESEARCH: dict(prob=0.01, time_to_next_call_ms=3_600_000.0)}
    )
    controller = wants_controller(FakePool(), rows)
    controller.maybe_tick()
    assert len(controller.drain_prefetch_wants(4)) == 1


def test_a_stale_forecast_buys_no_prefill_when_asked():
    """The eviction half degrades to LRU on staleness; this half can be told
    to degrade to doing nothing. Same conservative direction."""
    rows = {RESEARCH: ImportanceRow(prob=0.99, time_to_next_call_ms=1_000.0)}
    controller = wants_controller(
        FakePool(), rows, prefetch_ignore_staleness=False
    )
    controller.maybe_tick()
    assert controller.drain_prefetch_wants(4) == []


def test_a_stale_forecast_still_warms_by_default():
    """Asymmetric with the eviction half on purpose: a stale row there
    destroys a block, here it costs at worst one redundant prefill of a
    prefix this process has already served. A quiet forecast must not
    silently switch prefetching off."""
    rows = {RESEARCH: ImportanceRow(prob=0.99, time_to_next_call_ms=1_000.0)}
    controller = wants_controller(FakePool(), rows)
    controller.maybe_tick()
    assert len(controller.drain_prefetch_wants(4)) == 1


# -- residency -----------------------------------------------------------


def test_a_mostly_evicted_prefix_is_wanted_again():
    """The bug that made origination structurally dead. `remove_block` only
    deletes an index entry when its *last* block goes, so `get_entry(key) is
    not None` reported a key holding 1 of 4 blocks as resident and never
    re-warmed it. A prefix is worth something only as a run from position 0,
    so 1-of-4 is absent for hit-rate purposes."""
    pool = FakePool()
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(pool, rows)
    controller.on_blocks_cached(make_request(), pool.blocks[:4], 0)

    controller.maybe_tick()
    assert controller.drain_prefetch_wants(4) == [], "fully resident"

    # Evict all but the first block: run_len stays 4, num_blocks drops to 1.
    for block in pool.blocks[1:4]:
        controller.index.remove_block(block.block_id)
    controller.snapshot_source.set_rows(rows)
    controller.maybe_tick()

    assert len(controller.drain_prefetch_wants(4)) == 1


def test_partial_residency_counts_as_resident_when_coverage_is_off():
    """`prefetch_min_coverage=0` restores the original all-or-nothing test,
    so the two can be A/B'd without a code change."""
    pool = FakePool()
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(pool, rows, prefetch_min_coverage=0.0)
    controller.on_blocks_cached(make_request(), pool.blocks[:4], 0)
    for block in pool.blocks[1:4]:
        controller.index.remove_block(block.block_id)

    controller.maybe_tick()
    assert controller.drain_prefetch_wants(4) == []


def test_the_most_valuable_want_is_handed_over_first():
    rows = fresh_rows(
        {
            RESEARCH: dict(prob=0.99, time_to_next_call_ms=100.0),
            SUPERVISOR: dict(prob=0.51, time_to_next_call_ms=50_000.0),
        }
    )
    controller = wants_controller(FakePool(), rows, prefetch_max_per_drain=1)
    controller.maybe_tick()

    drained = controller.drain_prefetch_wants(4)
    assert len(drained) == 1
    assert drained[0]["node"] == "research"


# -- visibility ----------------------------------------------------------


def test_the_summary_reports_both_halves_of_the_question():
    """Wants answer "did the instruction reach HBM"; speculative answers "was
    what landed used". Either alone hides a distinct failure, so the one line
    everyone will actually read has to carry both."""
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(FakePool(), rows)
    controller.maybe_tick()
    controller.drain_prefetch_wants(4)

    summary = controller.prefetch_summary()
    assert "wants created=1" in summary
    assert "drained=1" in summary
    assert "hit_rate=" in summary
    assert "speculative" in summary
    assert "waste=" in summary


def test_there_is_no_summary_when_origination_is_off():
    """Flag-off must stay silent, not print a line of zeroes."""
    assert make_controller(FakePool()).prefetch_summary() is None


def test_an_idle_server_does_not_repeat_the_summary():
    """Rate-limited *and* change-gated. An identical line every 30s forever
    trains everyone to filter it out, and the one time it matters is the time
    it changed.

    Asserted on the return value rather than through `caplog`: vLLM's root
    logger sets `propagate=False`, so caplog captures nothing and the test
    would pass without the behaviour existing.
    """
    import time

    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(
        FakePool(), rows, prefetch_summary_period_ms=0.001
    )

    controller.maybe_tick()  # creates a want -> counters move
    assert controller.observer.counters.prefetch_wants_created == 1
    # The tick above already emitted; nothing has changed since.
    assert controller._maybe_log_prefetch_summary(time.monotonic()) is None

    controller.drain_prefetch_wants(4)
    assert controller._maybe_log_prefetch_summary(time.monotonic()) is not None


def test_a_reset_drops_every_want():
    """`reset_prefix_cache` invalidates every block hash, so an outstanding
    want's prefix is gone whether or not its phantom landed. Holding it would
    block a re-offer exactly when everything needs re-warming."""
    rows = fresh_rows({RESEARCH: dict(prob=0.9, time_to_next_call_ms=1_000.0)})
    controller = wants_controller(FakePool(), rows)
    controller.maybe_tick()
    controller.drain_prefetch_wants(4)

    controller.on_reset_prefix_cache()

    assert controller.stats()["prefetch_wants_outstanding"] == 0
    assert controller.stats()["prefetch_wants_pending"] == 0
