# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-agent LRU registry of recently-seen prefix descriptors.

Tracks, per ``agent_id``, the last N chunk-aligned prefixes the agent
used. When a new request arrives the API layer pulls the top-K
descriptors and fires phantom prefetches for them, so vLLM's APC is
warm by the time the real call reaches prefill.

The registry is in-memory only and bounded along two axes:

* ``max_per_agent`` -- LRU depth within a single agent.
* ``max_agents``    -- across-agent LRU eviction once too many agents
  are tracked.

Thread-safety: all public methods take an ``RLock``. The HTTP server
calls in from many async workers, and engine callbacks (e.g. response
post-processing) may also touch the registry.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class PrefixDescriptor:
    """A single recordable prefix entry."""

    # Chunk-aligned token sequence covering this prefix. Length is a
    # multiple of the LMCache chunk size by construction.
    token_ids: tuple[int, ...]
    # Stable identity hash; see ``hashing.compute_prefix_hash``.
    prefix_hash: bytes
    # Per-agent isolation salt forwarded to LMCache key derivation
    # for the phantom prefetch.
    cache_salt: str
    # Monotonic ns timestamp of the most recent record / promote.
    last_used_ns: int = field(default_factory=time.monotonic_ns)

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("PrefixDescriptor.token_ids must be non-empty")
        if not self.prefix_hash:
            raise ValueError("PrefixDescriptor.prefix_hash must be non-empty")


class AgentPrefixRegistry:
    """Bounded per-agent LRU of :class:`PrefixDescriptor` entries.

    Layout: ``OrderedDict[agent_id, OrderedDict[prefix_hash, descriptor]]``.
    The outer dict is LRU-ordered on whichever agent was touched most
    recently; the inner dict is LRU-ordered on which prefix the agent
    used most recently.

    All operations are O(1) amortized (``OrderedDict.move_to_end``).
    """

    def __init__(
        self,
        default_top_k: int = 20,
        max_agents: int = 10_000,
        max_per_agent: int = 64,
    ) -> None:
        if default_top_k < 0:
            raise ValueError(
                f"default_top_k must be >= 0, got {default_top_k}"
            )
        if max_agents <= 0:
            raise ValueError(f"max_agents must be positive, got {max_agents}")
        if max_per_agent <= 0:
            raise ValueError(
                f"max_per_agent must be positive, got {max_per_agent}"
            )
        if default_top_k > max_per_agent:
            raise ValueError(
                "default_top_k cannot exceed max_per_agent "
                f"({default_top_k} > {max_per_agent})"
            )

        self.default_top_k = default_top_k
        self.max_agents = max_agents
        self.max_per_agent = max_per_agent

        self._lock = threading.RLock()
        self._by_agent: OrderedDict[
            str, OrderedDict[bytes, PrefixDescriptor]
        ] = OrderedDict()

    # -- mutators ---------------------------------------------------------

    def record(self, agent_id: str, desc: PrefixDescriptor) -> None:
        """Insert or promote ``desc`` for ``agent_id``.

        If a descriptor with the same ``prefix_hash`` already exists
        for the agent, it is replaced (last-used timestamp refreshed)
        and moved to the MRU position. Otherwise the descriptor is
        inserted at MRU and the oldest entry is evicted if the
        per-agent cap is exceeded.
        """
        if not agent_id:
            raise ValueError("agent_id must be non-empty")

        with self._lock:
            agent_map = self._by_agent.get(agent_id)
            if agent_map is None:
                agent_map = OrderedDict()
                self._by_agent[agent_id] = agent_map
                self._maybe_evict_agent_locked()
            else:
                # Promote the agent itself to MRU in the outer dict.
                self._by_agent.move_to_end(agent_id)

            if desc.prefix_hash in agent_map:
                # Replace stale descriptor (token_ids etc. may differ
                # if cache_salt changed) and promote.
                agent_map[desc.prefix_hash] = desc
                agent_map.move_to_end(desc.prefix_hash)
            else:
                agent_map[desc.prefix_hash] = desc
                while len(agent_map) > self.max_per_agent:
                    evicted_hash, _ = agent_map.popitem(last=False)
                    logger.debug(
                        "agent_prefetch: evicted prefix %s from agent %s "
                        "(per-agent LRU)",
                        evicted_hash.hex()[:12],
                        agent_id,
                    )

    def evict_agent(self, agent_id: str) -> bool:
        """Drop all entries for an agent. Returns True if anything was
        removed."""
        with self._lock:
            return self._by_agent.pop(agent_id, None) is not None

    def clear(self) -> None:
        """Drop the entire registry."""
        with self._lock:
            self._by_agent.clear()

    # -- readers ----------------------------------------------------------

    def top_k(self, agent_id: str, k: int | None = None) -> list[PrefixDescriptor]:
        """Return up to ``k`` most-recently-used descriptors for the agent.

        ``k`` defaults to ``self.default_top_k``. Returns an empty list
        if the agent is unknown or ``k <= 0``.

        Does **not** promote the agent or its descriptors -- reading is
        side-effect free so a phantom-prefetch fanout can't itself
        thrash LRU order.
        """
        if k is None:
            k = self.default_top_k
        if k <= 0:
            return []

        with self._lock:
            agent_map = self._by_agent.get(agent_id)
            if not agent_map:
                return []
            # OrderedDict is iterated oldest -> newest; we want the
            # newest k entries, in newest-first order.
            descriptors = list(agent_map.values())
            return list(reversed(descriptors[-k:]))

    def agent_size(self, agent_id: str) -> int:
        with self._lock:
            agent_map = self._by_agent.get(agent_id)
            return len(agent_map) if agent_map is not None else 0

    def num_agents(self) -> int:
        with self._lock:
            return len(self._by_agent)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "num_agents": len(self._by_agent),
                "total_descriptors": sum(
                    len(m) for m in self._by_agent.values()
                ),
                "max_agents": self.max_agents,
                "max_per_agent": self.max_per_agent,
                "default_top_k": self.default_top_k,
            }

    # -- internal ---------------------------------------------------------

    def _maybe_evict_agent_locked(self) -> None:
        """Evict the LRU agent if we've exceeded ``max_agents``.

        Called with ``self._lock`` held. The agent we just inserted is
        at MRU, so ``popitem(last=False)`` cannot evict it.
        """
        while len(self._by_agent) > self.max_agents:
            evicted_agent, _ = self._by_agent.popitem(last=False)
            logger.debug(
                "agent_prefetch: evicted agent %s (cross-agent LRU)",
                evicted_agent,
            )
