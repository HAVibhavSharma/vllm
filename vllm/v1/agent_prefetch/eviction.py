# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agent-aware early-eviction policy.

This module implements a cross-request, in-process policy that lets the
GPU block pool evict cached blocks **before** they reach the LRU head if
their owning agent is unlikely to be the next one to fire.

Mental model::

    Each live ``/v1/agents/chat/completions`` request can carry an
    ``agent_probabilities`` dict: ``agent_id -> P(agent fires soon)``.
    The policy maintains a global view of these per-agent probabilities
    aggregated (max) across every live request.

    When the GPU block pool runs short, it asks the policy for eviction
    candidates. The policy ranks every non-active agent by aggregated
    probability ascending and drains blocks from the lowest-probability
    agent first, walking up the ranking until the pool's request is
    satisfied.

The policy is a process-wide singleton because the scheduler /
block-pool live in a single process; the API endpoint feeds info in via
``register_request`` / ``unregister_request`` and the engine's block
pool queries it from ``get_new_blocks`` / ``cache_full_blocks``.

Thread-safety: every public method takes an ``RLock``.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Mapping

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ActiveRequestInfo:
    """Per-live-request data feeding the global aggregator."""

    request_id: str
    agent_id: str
    # Agent id -> P(agent fires soon). An agent missing from this map
    # is treated as probability 0 by the aggregator (i.e. its blocks
    # sit at the head of the eviction ranking unless some other live
    # request lifts them higher).
    agent_probabilities: dict[str, float] = field(default_factory=dict)


class AgentEvictionPolicy:
    """In-process registry of agent probabilities + block ownership.

    Holds three correlated indices:

    * ``_active``: live-request info (agent id, probabilities).
      Populated when the scheduler admits a request, removed when the
      request finishes.
    * ``_block_owner``: which agent owns a given cached block. Set when
      a block becomes cached (`cache_full_blocks`); cleared when the
      block leaves the prefix cache (`_maybe_evict_cached_block` or
      `reset_prefix_cache`).
    * ``_agent_blocks``: reverse index ``agent_id -> set[block_id]``,
      ordered LRU so we can hand the LRU-first set of an agent's blocks
      back to the pool.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[str, ActiveRequestInfo] = {}
        # request_id -> agent_id, kept as a fast lookup for tag_block
        # paths where we only have the request_id at hand.
        self._request_to_agent: dict[str, str] = {}
        self._block_owner: dict[int, str] = {}
        # Use OrderedDict-as-set so we can keep LRU-ish order of when a
        # block was tagged; popping from the head gives the oldest first.
        self._agent_blocks: dict[str, OrderedDict[int, None]] = {}
        # Per-request tag bookkeeping for diagnostic logging. Keyed by
        # request_id; cleared on unregister after the SUMMARY line is
        # emitted. Tracks fresh tags, re-tags (block changed agent),
        # LRU refreshes (same agent re-touches its block), and the
        # number of unmapped tag attempts for requests we never saw a
        # register_request() for.
        self._tag_stats: dict[str, dict[str, int]] = {}
        self._unmapped_tag_counts: dict[str, int] = {}

    # -- live-request registry -------------------------------------------

    def register_request(
        self,
        request_id: str,
        agent_id: str,
        agent_probabilities: Mapping[str, float] | None = None,
    ) -> None:
        """Register a live agent request with the policy.

        Idempotent: re-registering the same request_id overwrites the
        previous info (useful when probabilities are refined mid-flight,
        though the current API doesn't do this).

        The vote lives in ``_active`` only for the duration of the
        request -- ``unregister_request`` removes it. Multiple requests
        that overlap in time contribute their votes simultaneously,
        and the aggregator takes the ``max`` per agent. Sequential
        requests do **not** leak state into one another.
        """
        if not agent_id:
            return
        probs = dict(agent_probabilities) if agent_probabilities else {}
        info = ActiveRequestInfo(
            request_id=request_id,
            agent_id=agent_id,
            agent_probabilities=probs,
        )
        with self._lock:
            self._active[request_id] = info
            self._request_to_agent[request_id] = agent_id
            self._agent_blocks.setdefault(agent_id, OrderedDict())
        logger.debug(
            "agent_eviction: register request=%s agent=%s probs=%s",
            request_id, agent_id, probs,
        )

    def unregister_request(self, request_id: str) -> None:
        """Drop the live-request entry. Block ownership is left intact
        so any cached blocks the request produced can still be targeted
        for low-probability eviction after the request finishes."""
        import sys
        with self._lock:
            agent_id = self._request_to_agent.get(request_id)
            stats = self._tag_stats.pop(request_id, None)
            self._active.pop(request_id, None)
            self._request_to_agent.pop(request_id, None)
            agent_total = (
                len(self._agent_blocks.get(agent_id, ()))
                if agent_id is not None else 0
            )
            global_tagged = len(self._block_owner)
            unmapped = self._unmapped_tag_counts.pop(request_id, 0)
        if stats is not None:
            print(
                f"DBG tag_block SUMMARY req={request_id} agent={agent_id} "
                f"new={stats['new']} retag={stats['retag']} "
                f"refresh={stats['refresh']} "
                f"agent_owns_now={agent_total} "
                f"global_tagged={global_tagged}",
                file=sys.stderr, flush=True,
            )
        if unmapped:
            print(
                f"DBG tag_block SUMMARY req={request_id} agent=<UNMAPPED> "
                f"unmapped_attempts={unmapped}",
                file=sys.stderr, flush=True,
            )
        logger.debug("agent_eviction: unregister request=%s", request_id)

    # -- block-ownership tagging -----------------------------------------

    def tag_block(self, block_id: int, request_id: str) -> None:
        """Mark ``block_id`` as owned by the agent that owns
        ``request_id``. No-op if the request isn't agent-tagged.

        Logging: instead of sampling every Nth block (which masks
        contention) we keep a per-request counter and emit:

        * ``tag_block START`` -- first time we see this request tagging,
        * ``tag_block RETAG`` -- a block changes owner (signals
          cross-agent cache contention; always logged, never sampled),
        * ``tag_block SUMMARY`` -- on ``unregister_request`` with the
          final per-request totals and the agent's current bucket size.
        """
        import sys
        with self._lock:
            agent_id = self._request_to_agent.get(request_id)
            if agent_id is None:
                count = self._unmapped_tag_counts.get(request_id, 0) + 1
                self._unmapped_tag_counts[request_id] = count
                if count == 1:
                    print(
                        f"DBG tag_block UNMAPPED req={request_id} "
                        f"first_block={block_id} "
                        f"known_reqs={list(self._request_to_agent.keys())[:3]}",
                        file=sys.stderr, flush=True,
                    )
                return
            stats = self._tag_stats.get(request_id)
            if stats is None:
                stats = {"new": 0, "retag": 0, "refresh": 0}
                self._tag_stats[request_id] = stats
                print(
                    f"DBG tag_block START req={request_id} agent={agent_id} "
                    f"first_block={block_id}",
                    file=sys.stderr, flush=True,
                )
            prev_agent = self._block_owner.get(block_id)
            if prev_agent == agent_id:
                # Refresh LRU position.
                bucket = self._agent_blocks.setdefault(agent_id, OrderedDict())
                if block_id in bucket:
                    bucket.move_to_end(block_id)
                stats["refresh"] += 1
                return
            if prev_agent is not None:
                # Remove from previous owner's bucket and log the steal --
                # this is the signal that cache pressure is real.
                prev_bucket = self._agent_blocks.get(prev_agent)
                prev_remaining = 0
                if prev_bucket is not None:
                    prev_bucket.pop(block_id, None)
                    prev_remaining = len(prev_bucket)
                stats["retag"] += 1
                print(
                    f"DBG tag_block RETAG block={block_id} "
                    f"req={request_id} agent={agent_id} "
                    f"prev_agent={prev_agent} "
                    f"prev_agent_owns_now={prev_remaining}",
                    file=sys.stderr, flush=True,
                )
            else:
                stats["new"] += 1
            self._block_owner[block_id] = agent_id
            self._agent_blocks.setdefault(agent_id, OrderedDict())[block_id] = None

    def tag_block_explicit(self, block_id: int, agent_id: str) -> None:
        """Tag a block directly with an agent id, without going through
        a live request. Used by tests and by paths that have the agent
        id but not the request id at hand."""
        if not agent_id:
            return
        with self._lock:
            prev_agent = self._block_owner.get(block_id)
            if prev_agent == agent_id:
                bucket = self._agent_blocks.setdefault(agent_id, OrderedDict())
                if block_id in bucket:
                    bucket.move_to_end(block_id)
                return
            if prev_agent is not None:
                prev_bucket = self._agent_blocks.get(prev_agent)
                if prev_bucket is not None:
                    prev_bucket.pop(block_id, None)
            self._block_owner[block_id] = agent_id
            self._agent_blocks.setdefault(agent_id, OrderedDict())[block_id] = None

    def untag_block(self, block_id: int) -> None:
        """Drop an agent->block association. Called when the prefix
        cache evicts a block or when ``reset_prefix_cache`` runs.

        Logging: every untag is a real eviction event, so we log it.
        To keep the line useful we include the agent's remaining bucket
        size and the global tagged-block count after the drop.
        """
        import sys
        with self._lock:
            agent_id = self._block_owner.pop(block_id, None)
            if agent_id is None:
                # Block wasn't tagged -- pure LRU eviction of an
                # untagged block. Not interesting, stay quiet.
                return
            bucket = self._agent_blocks.get(agent_id)
            bucket_remaining = 0
            if bucket is not None:
                bucket.pop(block_id, None)
                bucket_remaining = len(bucket)
                # Keep the bucket even if empty -- the agent may still
                # be referenced by a live request.
            global_tagged = len(self._block_owner)
        print(
            f"DBG untag_block block={block_id} agent={agent_id} "
            f"agent_owns_now={bucket_remaining} "
            f"global_tagged={global_tagged}",
            file=sys.stderr, flush=True,
        )

    def clear_all_blocks(self) -> None:
        """Drop every block-ownership association. Used by
        ``reset_prefix_cache``."""
        import sys
        with self._lock:
            dropped = len(self._block_owner)
            per_agent = {a: len(b) for a, b in self._agent_blocks.items() if b}
            self._block_owner.clear()
            for bucket in self._agent_blocks.values():
                bucket.clear()
        print(
            f"DBG clear_all_blocks dropped={dropped} "
            f"per_agent_before={per_agent}",
            file=sys.stderr, flush=True,
        )

    # -- queries ---------------------------------------------------------

    def aggregated_probability(self, agent_id: str) -> float:
        """Return the max probability assigned to ``agent_id`` across
        every live request.

        Max (rather than mean/sum) is the most generous aggregator: if
        any live request thinks an agent is likely, we protect its
        blocks. Returns 0.0 when no live request mentions the agent --
        which sits it at the head of the eviction ranking.

        Each live request also implicitly votes ``1.0`` for **its own**
        ``agent_id``: an agent that's currently running cannot
        sensibly be classified as "low probability." Explicit votes
        still win if higher (capped at 1.0 anyway).
        """
        with self._lock:
            if not self._active:
                return 0.0
            best = 0.0
            for info in self._active.values():
                p = info.agent_probabilities.get(agent_id, 0.0)
                if info.agent_id == agent_id and p < 1.0:
                    p = 1.0
                if p > best:
                    best = p
            return best

    def _agents_by_probability_locked(self) -> list[tuple[str, float]]:
        """Return candidate agents sorted by ascending aggregated
        probability. Caller must hold ``self._lock``.

        Currently-active agents are excluded entirely: a request
        implicitly self-votes 1.0 for its own ``agent_id``, and we
        never evict the agent that's actively generating tokens. Every
        other agent (block-owning or merely referenced by some live
        request's forecast) is ranked by the highest probability any
        live request assigns it, lowest first.
        """
        if not self._active:
            return []
        active_agents: set[str] = {
            info.agent_id for info in self._active.values()
        }
        candidates: set[str] = set(self._agent_blocks.keys())
        for info in self._active.values():
            candidates.update(info.agent_probabilities.keys())
        candidates -= active_agents
        ranked: list[tuple[str, float]] = []
        for agent_id in candidates:
            best = 0.0
            for info in self._active.values():
                p = info.agent_probabilities.get(agent_id, 0.0)
                if p > best:
                    best = p
            ranked.append((agent_id, best))
        ranked.sort(key=lambda x: x[1])
        return ranked

    def evictable_blocks(self, num_needed: int) -> list[int]:
        """Return up to ``num_needed`` block ids in lowest-probability
        order: drain the agent with the smallest aggregated probability
        first, walk up the ranking, and keep taking until the request is
        satisfied or every non-active agent has been drained.

        There is no threshold gate -- if the request needs blocks, the
        policy hands over whatever it has, starting from the agents
        least likely to fire next. Currently-active agents are never
        touched (they self-vote 1.0).

        The pool then has to verify each block is actually a valid
        eviction target (``ref_cnt == 0`` and in the free queue) --
        a block we tagged may have been reallocated to a busy request
        since we last looked. The pool ignores any block that fails
        that check and falls back to the LRU head for the remainder.
        """
        import sys
        if num_needed <= 0:
            return []
        with self._lock:
            # Throttle: this fires every eviction attempt; sample 1/50.
            self._evictable_call_count = getattr(
                self, "_evictable_call_count", 0) + 1
            do_print = (self._evictable_call_count % 50 == 0)
            if not self._active:
                if do_print:
                    print(
                        f"DBG evictable_blocks(num_needed={num_needed}) "
                        f"-> [] : _active is EMPTY",
                        file=sys.stderr, flush=True,
                    )
                return []
            ranked = self._agents_by_probability_locked()
            if not ranked:
                if do_print:
                    print(
                        f"DBG evictable_blocks(num_needed={num_needed}) "
                        f"-> [] : no candidate agents "
                        f"(active={len(self._active)}, "
                        f"tagged={list(self._agent_blocks.keys())})",
                        file=sys.stderr, flush=True,
                    )
                return []
            picks: list[int] = []
            drained: list[tuple[str, float, int]] = []
            for agent_id, prob in ranked:
                bucket = self._agent_blocks.get(agent_id)
                if not bucket:
                    continue
                taken = 0
                for block_id in bucket.keys():
                    picks.append(block_id)
                    taken += 1
                    if len(picks) >= num_needed:
                        drained.append((agent_id, prob, taken))
                        if do_print:
                            print(
                                f"DBG evictable_blocks(num_needed={num_needed}) "
                                f"-> {len(picks)} blocks "
                                f"(drained_lowest_first={drained})",
                                file=sys.stderr, flush=True,
                            )
                        return picks
                drained.append((agent_id, prob, taken))
            if do_print:
                print(
                    f"DBG evictable_blocks(num_needed={num_needed}) "
                    f"-> {len(picks)} blocks "
                    f"(drained_lowest_first={drained}, short of need)",
                    file=sys.stderr, flush=True,
                )
            return picks

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "active_requests": len(self._active),
                "tagged_blocks": len(self._block_owner),
                "tracked_agents": len(self._agent_blocks),
            }


# Module-level singleton -------------------------------------------------

_POLICY_LOCK = threading.Lock()
_POLICY_SINGLETON: AgentEvictionPolicy | None = None


def get_eviction_policy() -> AgentEvictionPolicy:
    """Return the process-wide ``AgentEvictionPolicy`` instance,
    creating it on first access."""
    global _POLICY_SINGLETON
    if _POLICY_SINGLETON is None:
        with _POLICY_LOCK:
            if _POLICY_SINGLETON is None:
                _POLICY_SINGLETON = AgentEvictionPolicy()
    return _POLICY_SINGLETON


def reset_eviction_policy_for_tests() -> None:
    """Drop the singleton so unit tests can start from a clean state."""
    global _POLICY_SINGLETON
    with _POLICY_LOCK:
        _POLICY_SINGLETON = None
