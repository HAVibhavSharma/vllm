# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared value types for the node-aware KV eviction policy.

Everything here is immutable. The importance snapshot in particular is
published to the scheduler thread by a single attribute assignment and read
without a lock, which is only safe because nothing mutates it after
construction (see plan/new-eviction/01-redis-integration.md §2).
"""

from dataclasses import dataclass, field
from typing import NamedTuple


class NodeKey(NamedTuple):
    """The join key between vLLM's block index and the Redis forecast.

    `(job_id, langgraph_node, call_type)`, per
    plan/new-eviction/01-redis-integration.md §4 and §5:

    - `job_id` scopes one workflow run. Two concurrent jobs running the same
      node have different forecasts, so collapsing them averages unrelated
      signals.
    - `langgraph_node` is the bare node name, which joins directly with the
      Redis `node_name` field. `agent_id` is namespaced (`langgraph:research`)
      and is deliberately not used here.
    - `call_type` is the flattened leaf name (`summary`, `tavily:summary`),
      which makes the join with `PROB`/`HISTORY` exact rather than aggregated.
    """

    job_id: str
    node: str
    call_type: str


@dataclass(frozen=True, slots=True)
class ImportanceRow:
    """One `(node, call_type)` forecast row, joined from `PROB` + `HISTORY`.

    Units are the ones pinned in plan/new-eviction/03-workflow-contract.md §4:
    `prob` is a fraction in [0, 1] and every duration is in milliseconds.
    """

    # From PROB.
    prob: float = 0.0
    time_to_next_call_ms: float = 0.0
    update_ts_ms: float = 0.0

    # What `prob` was asked. None (the default) means "will this fire again at
    # all, this job" — time-free, leaving `decay(ttnc)` to supply the whole
    # time preference. An integer N means `prob = P(fires within the next N
    # calls)`, which already carries that preference, and applying the decay
    # on top would discount time twice.
    #
    # It rides on the row rather than on a vLLM-side config flag on purpose.
    # The publisher is the only component that knows what it computed, and two
    # independently-set flags that disagree would produce a policy that is
    # quietly wrong with a healthy-looking log on both sides — the failure mode
    # this design is least able to detect (03 §5).
    prob_horizon: int | None = None

    # From HISTORY. p_l1 / p_cold default to the "assume L1 resident"
    # position of 01 §5: every miss is recoverable from LMCache. Publishing a
    # measured p_cold changes the ranking without any code change here.
    p_l1: float = 1.0
    p_cold: float = 0.0
    delta_l1_ms: float | None = None
    delta_cold_ms: float | None = None

    # Number of samples behind the HISTORY statistics, if published. Carried
    # through to the decision log so a mean over 2 samples is not read like a
    # mean over 200 (04 §5.3).
    num_samples: int = 0


@dataclass(frozen=True, slots=True)
class ImportanceSnapshot:
    """An immutable point-in-time view of the Redis forecast.

    Published by reference swap from the subscriber thread; the scheduler
    thread takes one reference per tick and never mutates it.
    """

    rows: dict[NodeKey, ImportanceRow] = field(default_factory=dict)
    # Wall-clock epoch ms at which this snapshot was built, used only for
    # reporting staleness. Row-level freshness comes from `update_ts_ms`.
    built_at_ms: float = 0.0
    # Monotonic reference for the same instant, so age can be computed
    # without trusting wall-clock jumps.
    built_at_monotonic: float = 0.0
    # Incremented on every rebuild. The tick's cheap "did anything change?"
    # check is an identity comparison on the snapshot object; this is for
    # logging and tests.
    revision: int = 0

    def get(self, key: NodeKey) -> ImportanceRow | None:
        return self.rows.get(key)


EMPTY_SNAPSHOT = ImportanceSnapshot()


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """A score together with the terms that produced it.

    The terms are what make a bad score diagnosable rather than merely
    visible (07 §4); they are carried into the eviction decision log.
    """

    score: float
    prob: float
    ttnc_ms: float
    decay: float
    e_miss_ms: float
    blocks: int
    speculative: bool = False
    floor_applied: float | None = None
    scored: bool = True

    def as_terms(self) -> dict[str, float | int]:
        return {
            "prob": self.prob,
            "ttnc_ms": self.ttnc_ms,
            "decay": self.decay,
            "e_miss_ms": self.e_miss_ms,
            "blocks": self.blocks,
        }


# Sentinel for a block that no scored key owns. Unscored blocks are never
# selected by the splice, so they keep their LRU position; this is only ever
# used as a lookup miss marker (08 §4).
UNSCORED = ScoreBreakdown(
    score=0.0,
    prob=0.0,
    ttnc_ms=0.0,
    decay=0.0,
    e_miss_ms=0.0,
    blocks=0,
    scored=False,
)
