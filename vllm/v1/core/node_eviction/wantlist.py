# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prefetch want-list — build step 6 (02 §4, option a).

The diagram asks *"can we not push the prefetching decision to the KV CACHE
MANAGER?"*. The decision, yes; the submission, no. `PhantomPrefetchSubmitter`
holds an `EngineClient` and lives in the **API-server** process, while this
manager lives in **engine core**, which has no channel for originating a
request — `EngineCoreOutputs` carries per-request output, not requests.

So the two halves are split (02 §4 option a):

- **here** — the manager decides, and publishes a want-list: "these keys are
  about to be needed and are not in HBM."
- **front end** — `vllm/v1/agent_prefetch/drain.py` polls the list over the
  existing `call_utility` channel and submits phantoms through the existing
  submitter. No new transport, no new message type.

Option (b) (a new engine→front-end message type) was rejected as invasive, and
(c) (the manager driving the connector directly) needs a connector API to load
a prefix by hash outside a request lifecycle, which `LMCacheMPConnector` does
not expose.

**This structure is a bounded, deduplicating queue, and every bound matters.**
A want costs a real prefill if LMCache misses (see
`lmcache_mp_connector.py:1032`), so an unbounded or duplicating want-list does
not degrade the cache — it competes with the requests it was meant to serve.
"""

import heapq
from dataclasses import dataclass

from vllm.v1.core.node_eviction.types import NodeKey


@dataclass(frozen=True, slots=True)
class PrefetchWant:
    """One "pull this prefix into HBM" instruction.

    `agent_id` is resolved here rather than in the front end because the
    `{namespace}:{node}` convention is the engine's own join with the
    LangGraph fork's `derive_agent_id()`; keeping it in one place means the
    drainer needs no knowledge of the naming scheme.
    """

    key: NodeKey
    agent_id: str
    # Expected ms of latency saved if this prefix is resident when the call
    # lands: `prob * decay(ttnc) * E_miss`. Deliberately *not* divided by
    # blocks — the eviction score is a density because it ranks blocks
    # already held, while this ranks whole prefixes not yet fetched, and the
    # prefix's block count is unknown until it lands (08 §2).
    score: float
    prob: float
    time_to_next_call_ms: float
    # Monotonic seconds at creation. Drives both TTLs below.
    created_at: float

    def as_dict(self) -> dict[str, str | float]:
        """Flat, msgpack-safe form for the `call_utility` hop.

        `NodeKey` is a `NamedTuple` and would survive encoding, but the
        front end has no reason to import engine-core types, so the wire
        form is plain scalars.
        """
        return {
            "job_id": self.key.job_id,
            "node": self.key.node,
            "call_type": self.key.call_type,
            "agent_id": self.agent_id,
            "score": self.score,
            "prob": self.prob,
            "time_to_next_call_ms": self.time_to_next_call_ms,
        }


class PrefetchWantList:
    """Bounded set of outstanding prefetch wants, deduplicated by key.

    Two states, because "decided" and "acted on" are separated by a poll
    interval:

    - **pending** — offered by the tick, not yet handed to the front end.
    - **outstanding** — handed over, phantom presumably in flight, not yet
      observed in the index.

    A key in either state is not re-offered. Without that, the tick (every
    250 ms by default) would re-emit the same want on every pass while the
    phantom for it was still loading, and the drain interval (1 s) would
    submit each of them.
    """

    def __init__(
        self,
        max_outstanding: int = 8,
        want_ttl_ms: float = 30_000.0,
        resubmit_backoff_ms: float = 60_000.0,
    ) -> None:
        self.max_outstanding = max_outstanding
        self.want_ttl_ms = want_ttl_ms
        self.resubmit_backoff_ms = resubmit_backoff_ms
        self._pending: dict[NodeKey, PrefetchWant] = {}
        # key -> monotonic seconds at which it was drained.
        self._outstanding: dict[NodeKey, float] = {}

    # -- production --------------------------------------------------------

    def offer(self, want: PrefetchWant) -> bool:
        """Add a want unless it is a duplicate or the list is full.

        Returns True if it was accepted. A rejection is not an error — it is
        the normal steady state once the forecast is stable, and it is what
        keeps the phantom rate bounded by `max_outstanding` per TTL rather
        than by the tick rate.
        """
        key = want.key
        if key in self._pending or key in self._outstanding:
            return False
        if len(self._pending) + len(self._outstanding) >= self.max_outstanding:
            return False
        self._pending[key] = want
        return True

    # -- consumption -------------------------------------------------------

    def drain(self, max_items: int, now: float) -> list[PrefetchWant]:
        """Take up to `max_items` pending wants, highest score first.

        Called on the engine-core loop thread via `call_utility`, which is
        the same thread that runs the tick, so there is no locking here.
        """
        if max_items <= 0 or not self._pending:
            return []
        taken = heapq.nlargest(
            max_items, self._pending.values(), key=lambda w: w.score
        )
        for want in taken:
            del self._pending[want.key]
            self._outstanding[want.key] = now
        return taken

    def note_satisfied(self, key: NodeKey) -> bool:
        """The prefix is in HBM — retire any want for it.

        Driven from the tick by the key's presence in the ownership index,
        which is true whether the phantom landed it, a real request did, or
        it was never actually absent. All three mean the same thing to a
        prefetcher: stop asking.
        """
        found = self._pending.pop(key, None) is not None
        return self._outstanding.pop(key, None) is not None or found

    def expire(self, now: float) -> tuple[int, int]:
        """Drop wants that have aged out. Returns `(pending, outstanding)`.

        Both TTLs exist to fail *open*:

        - a **pending** want nobody drained is stale intent; handing a
          minute-old prediction to a drainer that just woke up would prefetch
          for a call that has already happened.
        - an **outstanding** want that never showed up in the index means the
          phantom was dropped, missed, or failed. Releasing it after the
          backoff is what allows a retry; holding it forever would silently
          consume one of `max_outstanding` slots for the process lifetime.
        """
        pending_ttl = self.want_ttl_ms / 1000.0
        outstanding_ttl = self.resubmit_backoff_ms / 1000.0
        dropped_pending = 0
        for key in [
            k
            for k, w in self._pending.items()
            if now - w.created_at >= pending_ttl
        ]:
            del self._pending[key]
            dropped_pending += 1
        dropped_outstanding = 0
        for key in [
            k
            for k, ts in self._outstanding.items()
            if now - ts >= outstanding_ttl
        ]:
            del self._outstanding[key]
            dropped_outstanding += 1
        return dropped_pending, dropped_outstanding

    def clear(self) -> None:
        self._pending.clear()
        self._outstanding.clear()

    # -- reads -------------------------------------------------------------

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_outstanding(self) -> int:
        return len(self._outstanding)

    def is_tracked(self, key: NodeKey) -> bool:
        return key in self._pending or key in self._outstanding
