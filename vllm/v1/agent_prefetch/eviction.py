# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agent-aware early-eviction policy.

This module implements a cross-request, in-process policy that lets the
GPU block pool evict cached blocks **before** they reach the LRU head if
their owning agent is unlikely to be the next one to fire.

Mental model::

    Each live ``/v1/agents/chat/completions`` request can carry an
    ``agent_probabilities`` dict: ``agent_id -> P(agent fires in the
    next N turns)``. The policy maintains a global view of these
    per-agent probabilities aggregated across every live request.

    A block in state S3 (free-cached -- see ``plan/block_lifecycle.md``)
    that is owned by an agent whose aggregated probability is below
    ``eviction_threshold`` becomes liable for *early* eviction: the
    block pool prefers it over arbitrary LRU victims when satisfying a
    ``get_new_blocks`` request. This frees up cached GPU memory for
    prefixes that are actually likely to be reused soon.

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

DEFAULT_EVICTION_WINDOW = 3
"""Default `next-N-turns` window used when the request doesn't override it."""

DEFAULT_EVICTION_THRESHOLD = 0.5
"""Default probability threshold below which an agent's blocks are
liable for early eviction."""


@dataclass(frozen=True)
class ActiveRequestInfo:
    """Per-live-request data feeding the global aggregator."""

    request_id: str
    agent_id: str
    # Agent id -> P(agent fires in the next ``window`` turns).
    # An agent missing from this map is treated as probability 0 by the
    # aggregator (i.e. eligible for eviction unless some other live
    # request lifts it above the threshold).
    agent_probabilities: dict[str, float] = field(default_factory=dict)
    window: int = DEFAULT_EVICTION_WINDOW
    threshold: float = DEFAULT_EVICTION_THRESHOLD


class AgentEvictionPolicy:
    """In-process registry of agent probabilities + block ownership.

    Holds three correlated indices:

    * ``_active``: live-request info (agent id, probabilities, window,
      threshold). Populated when the scheduler admits a request,
      removed when the request finishes.
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

    # -- live-request registry -------------------------------------------

    def register_request(
        self,
        request_id: str,
        agent_id: str,
        agent_probabilities: Mapping[str, float] | None = None,
        window: int = DEFAULT_EVICTION_WINDOW,
        threshold: float = DEFAULT_EVICTION_THRESHOLD,
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
            window=max(1, int(window)) if window is not None else DEFAULT_EVICTION_WINDOW,
            threshold=float(threshold) if threshold is not None else DEFAULT_EVICTION_THRESHOLD,
        )
        with self._lock:
            self._active[request_id] = info
            self._request_to_agent[request_id] = agent_id
            self._agent_blocks.setdefault(agent_id, OrderedDict())
        logger.debug(
            "agent_eviction: register request=%s agent=%s probs=%s "
            "window=%d threshold=%.3f",
            request_id, agent_id, probs, info.window, info.threshold,
        )

    def unregister_request(self, request_id: str) -> None:
        """Drop the live-request entry. Block ownership is left intact
        so any cached blocks the request produced can still be targeted
        for low-probability eviction after the request finishes."""
        with self._lock:
            self._active.pop(request_id, None)
            self._request_to_agent.pop(request_id, None)
        logger.debug("agent_eviction: unregister request=%s", request_id)

    # -- block-ownership tagging -----------------------------------------

    def tag_block(self, block_id: int, request_id: str) -> None:
        """Mark ``block_id`` as owned by the agent that owns
        ``request_id``. No-op if the request isn't agent-tagged."""
        with self._lock:
            agent_id = self._request_to_agent.get(request_id)
            if agent_id is None:
                # Sample heavily: an un-mapped request emits this for
                # ~1.5K blocks; one line per ~100 blocks is plenty.
                if block_id % 100 == 0:
                    import sys
                    print(
                        f"DBG tag_block: block={block_id} req={request_id} "
                        f"-> NO AGENT MAPPED (known reqs: {list(self._request_to_agent.keys())[:3]})",
                        file=sys.stderr, flush=True,
                    )
                return
            # Only print every ~100th tag to avoid spam (one prompt = ~1500 blocks)
            if block_id % 100 == 0:
                import sys
                print(
                    f"DBG tag_block: block={block_id} req={request_id} agent={agent_id} "
                    f"(total tagged so far: {len(self._block_owner)})",
                    file=sys.stderr, flush=True,
                )
            prev_agent = self._block_owner.get(block_id)
            if prev_agent == agent_id:
                # Refresh LRU position.
                bucket = self._agent_blocks.setdefault(agent_id, OrderedDict())
                if block_id in bucket:
                    bucket.move_to_end(block_id)
                return
            if prev_agent is not None:
                # Remove from previous owner's bucket.
                prev_bucket = self._agent_blocks.get(prev_agent)
                if prev_bucket is not None:
                    prev_bucket.pop(block_id, None)
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
        cache evicts a block or when ``reset_prefix_cache`` runs."""
        with self._lock:
            agent_id = self._block_owner.pop(block_id, None)
            if agent_id is None:
                return
            bucket = self._agent_blocks.get(agent_id)
            if bucket is not None:
                bucket.pop(block_id, None)
                if not bucket:
                    # Keep the bucket even if empty -- the agent may
                    # still be referenced by a live request. Small.

                    pass

    def clear_all_blocks(self) -> None:
        """Drop every block-ownership association. Used by
        ``reset_prefix_cache``."""
        with self._lock:
            self._block_owner.clear()
            for bucket in self._agent_blocks.values():
                bucket.clear()

    # -- queries ---------------------------------------------------------

    def aggregated_probability(self, agent_id: str) -> float:
        """Return the max probability assigned to ``agent_id`` across
        every live request.

        Max (rather than mean/sum) is the most generous aggregator: if
        any live request thinks an agent is likely, we protect its
        blocks. Returns 0.0 when no live request mentions the agent --
        which makes its blocks immediately eligible for eviction once a
        threshold check is applied.

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

    def effective_threshold(self) -> float:
        """Compute the threshold to apply for the current eviction
        decision. When multiple live requests disagree on the
        threshold, we take the **min** -- the most lenient one. This
        means a block is liable for early eviction only when *every*
        live request is willing to evict at that probability.
        """
        with self._lock:
            if not self._active:
                return DEFAULT_EVICTION_THRESHOLD
            return min(info.threshold for info in self._active.values())

    def _low_probability_agents_locked(self) -> set[str]:
        """Return the set of agent ids whose aggregated probability is
        strictly below the effective threshold. Caller must hold
        ``self._lock``."""
        if not self._active:
            return set()
        threshold = min(info.threshold for info in self._active.values())
        # Union of every agent we have block ownership for and every
        # agent any live request talks about.
        candidates: set[str] = set(self._agent_blocks.keys())
        for info in self._active.values():
            candidates.update(info.agent_probabilities.keys())
            candidates.add(info.agent_id)
        low: set[str] = set()
        for agent_id in candidates:
            best = 0.0
            for info in self._active.values():
                p = info.agent_probabilities.get(agent_id, 0.0)
                if info.agent_id == agent_id and p < 1.0:
                    # A request implicitly votes 1.0 for its own
                    # agent_id -- the agent is actively running and
                    # must never be classified as low-probability.
                    p = 1.0
                if p > best:
                    best = p
            if best < threshold:
                low.add(agent_id)
        return low

    def low_probability_agents(self) -> set[str]:
        """Public wrapper around ``_low_probability_agents_locked``."""
        with self._lock:
            return self._low_probability_agents_locked()

    def evictable_blocks(self, num_needed: int) -> list[int]:
        """Return up to ``num_needed`` block ids whose owning agent is
        below the eviction threshold, in LRU-tag order (oldest first).

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
            low = self._low_probability_agents_locked()
            if not low:
                if do_print:
                    print(
                        f"DBG evictable_blocks(num_needed={num_needed}) "
                        f"-> [] : no low-prob agents "
                        f"(active={len(self._active)}, tagged_agents={list(self._agent_blocks.keys())})",
                        file=sys.stderr, flush=True,
                    )
                return []
            picks: list[int] = []
            for agent_id in low:
                bucket = self._agent_blocks.get(agent_id)
                if not bucket:
                    continue
                for block_id in bucket.keys():
                    picks.append(block_id)
                    if len(picks) >= num_needed:
                        if do_print:
                            print(
                                f"DBG evictable_blocks(num_needed={num_needed}) "
                                f"-> {len(picks)} blocks from low-prob agents {low}",
                                file=sys.stderr, flush=True,
                            )
                        return picks
            if do_print:
                print(
                    f"DBG evictable_blocks(num_needed={num_needed}) "
                    f"-> {len(picks)} blocks (low={low}, short of need)",
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
