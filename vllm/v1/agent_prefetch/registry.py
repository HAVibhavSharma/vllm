# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-agent registry of recently-seen prefix descriptors.

Tracks, per ``agent_id``, every distinct chunk-aligned prefix the
agent has used. When a new prefetch request arrives the API layer
pulls **all** of an agent's descriptors and fires one phantom request
per prefix, so vLLM's APC is warm by the time the real call reaches
prefill.

The registry is in-memory only. By default it is bounded along one
axis:

* ``max_agents`` -- across-agent LRU eviction once too many distinct
  agents are tracked.

A second optional axis -- ``max_per_agent`` -- exists for tests and
for operators who want to bound memory per agent. It defaults to
``None`` (no cap): the agent's inner map grows without per-agent
eviction.

Thread-safety: all public methods take an ``RLock``. The HTTP server
calls in from many async workers, and engine callbacks (e.g. response
post-processing) may also touch the registry.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


# A key containing one of these is a pattern, not an agent. Real ids come from
# `derive_agent_id`, which normalizes every segment, so neither can occur in
# one by accident.
_WILDCARD = "*"

# Guards the per-lookup scan. Patterns come from a seeding phase with one entry
# per node, so this is orders of magnitude above what a run creates; a config
# that blows past it is a bug, not a workload.
_MAX_PATTERNS = 256


def _compile_agent_pattern(pattern: str) -> re.Pattern[str] | None:
    """Translate a segment glob over agent ids into an anchored regex.

    Two wildcards, both segment-scoped -- never a raw regex from a caller,
    because these keys arrive over HTTP and a caller-supplied pattern is both
    a ReDoS surface and a way to match buckets it should not see.

    ``*``   exactly one segment  (``langgraph:*:researcher``)
    ``**``  zero or more segments (``langgraph:*:**:researcher``)

    The second is what makes a seed reachable: it spans the graph path, whose
    depth the seeding phase does not know. Returns None for a key with no
    wildcard, which is an ordinary agent id and must stay an exact-match key.
    """
    if _WILDCARD not in pattern:
        return None
    segments = pattern.split(":")
    out: list[str] = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            # Absorbs its own trailing separator so it can match zero segments.
            out.append(r"[^:]+(?::[^:]+)*" if last else r"(?:[^:]+:)*")
            continue
        if segment == _WILDCARD:
            out.append(r"[^:]+")
        else:
            out.append(re.escape(segment))
        if not last:
            out.append(":")
    try:
        return re.compile("".join(out) + r"\Z")
    except re.error:
        return None


# Separates the parallel-unit suffix from the graph path in an agent id, e.g.
# `langgraph:1:research_supervisor:supervisor_tools:researcher#2`. Mirrors
# `AGENT_UNIT_SEP` on the client that mints these ids; kept as a literal here
# rather than imported, since the client is not a dependency of the engine.
AGENT_UNIT_SEP = "#"


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
        max_per_agent: int | None = None,
    ) -> None:
        if default_top_k < 0:
            raise ValueError(
                f"default_top_k must be >= 0, got {default_top_k}"
            )
        if max_agents <= 0:
            raise ValueError(f"max_agents must be positive, got {max_agents}")
        if max_per_agent is not None and max_per_agent <= 0:
            raise ValueError(
                f"max_per_agent must be positive or None, got {max_per_agent}"
            )
        if max_per_agent is not None and default_top_k > max_per_agent:
            raise ValueError(
                "default_top_k cannot exceed max_per_agent "
                f"({default_top_k} > {max_per_agent})"
            )

        self.default_top_k = default_top_k
        self.max_agents = max_agents
        # ``None`` ≡ no per-agent cap; the inner map grows unbounded.
        self.max_per_agent = max_per_agent

        self._lock = threading.RLock()
        self._by_agent: OrderedDict[
            str, OrderedDict[bytes, PrefixDescriptor]
        ] = OrderedDict()
        # Compiled form of every wildcard key in `_by_agent`, kept alongside
        # rather than recompiled per lookup. Empty in the common case, which
        # is the one branch every read pays.
        self._patterns: dict[str, re.Pattern[str]] = {}

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
                compiled = _compile_agent_pattern(agent_id)
                if compiled is not None and len(self._patterns) < _MAX_PATTERNS:
                    self._patterns[agent_id] = compiled
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
                if self.max_per_agent is not None:
                    while len(agent_map) > self.max_per_agent:
                        evicted_hash, _ = agent_map.popitem(last=False)
                        logger.debug(
                            "agent_prefetch: evicted prefix %s from "
                            "agent %s (per-agent LRU)",
                            evicted_hash.hex()[:12],
                            agent_id,
                        )

    def evict_agent(self, agent_id: str) -> bool:
        """Drop all entries for an agent. Returns True if anything was
        removed.

        Exact key only. A `non-react` seed dropping every pattern that happens
        to cover it would delete another node's seeds as a side effect of
        recording its own.
        """
        with self._lock:
            self._patterns.pop(agent_id, None)
            return self._by_agent.pop(agent_id, None) is not None

    def clear(self) -> None:
        """Drop the entire registry."""
        with self._lock:
            self._by_agent.clear()
            self._patterns.clear()

    # -- readers ----------------------------------------------------------

    def _matching_agents_locked(self, agent_id: str) -> list[str]:
        """Pattern keys that cover ``agent_id``, exact key excluded.

        Matched against the id with its ``#unit`` suffix stripped as well as
        whole, so a seed written before any unit existed still covers the
        per-unit buckets a fan-out will ask for.
        """
        if not self._patterns:
            return []
        bare = agent_id.split(AGENT_UNIT_SEP, 1)[0]
        return [
            key
            for key, compiled in self._patterns.items()
            if key != agent_id
            and (compiled.match(agent_id) or compiled.match(bare))
        ]

    def _unit_siblings_locked(self, agent_id: str) -> list[str]:
        """Keys naming the same graph path as ``agent_id`` at a different unit.

        Two callers name the same predicted target differently, and neither is
        wrong. A predictor that fires *before* a fan-out has assigned its unit
        can only say ``…:researcher``; one that fires after, off a request
        that already carries ``…:researcher#1``, says that. Chat traffic is
        recorded under the id the request carried, so without this the
        unit-less caller looks into a bucket nothing ever writes and sees only
        whatever pattern seed happens to cover it -- never the conversation
        prefix the node has actually been accumulating.

        So the bare id is treated as the pooled parent of its units: asking
        for it sees every ``…#unit`` bucket, and asking for one unit also sees
        the parent. Pooling siblings is the price -- a bare lookup during a
        fan-out mixes ``#1`` and ``#2`` -- and it is the right price, because
        a caller that cannot name the unit is asking exactly that question.
        A caller that can name it still gets its own bucket first.
        """
        bare, sep, _ = agent_id.partition(AGENT_UNIT_SEP)
        if sep:
            # Asking as a unit: the pooled parent, if anything wrote it.
            return [bare] if bare in self._by_agent else []
        # Asking as the parent: every unit under it.
        prefix = agent_id + AGENT_UNIT_SEP
        return [key for key in self._by_agent if key.startswith(prefix)]

    def _descriptors_locked(self, agent_id: str) -> list[PrefixDescriptor]:
        """Every descriptor visible to ``agent_id``, newest-first.

        The agent's own bucket, any pattern bucket that covers it, and the
        unit siblings of its graph path, merged on recency rather than
        concatenated -- a caller asking for the k most recent prefixes must
        get the k most recent, not k from whichever bucket happened to be
        scanned first.
        """
        own = self._by_agent.get(agent_id)
        extra = self._matching_agents_locked(agent_id)
        extra.extend(
            key for key in self._unit_siblings_locked(agent_id) if key not in extra
        )
        if not extra:
            return list(reversed(list(own.values()))) if own else []
        pooled: list[PrefixDescriptor] = list(own.values()) if own else []
        # Deduped on prefix hash: the same prefix can sit in a unit bucket and
        # in a pattern bucket that covers it, and returning it twice would
        # spend two of the caller's k slots on one prefix.
        seen: set[bytes] = {d.prefix_hash for d in pooled}
        for key in extra:
            for desc in self._by_agent[key].values():
                if desc.prefix_hash in seen:
                    continue
                seen.add(desc.prefix_hash)
                pooled.append(desc)
        pooled.sort(key=lambda d: d.last_used_ns, reverse=True)
        return pooled

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
            return self._descriptors_locked(agent_id)[:k]

    def get_all(self, agent_id: str) -> list[PrefixDescriptor]:
        """Return *every* descriptor registered for ``agent_id``,
        newest-first.

        Side-effect-free (does not promote LRU). Returns an empty list
        for unknown agents. Use this in place of ``top_k`` when the
        caller wants no truncation -- the prefetch endpoint uses this
        whenever ``prefetch_top_k`` is omitted, so every stored prefix
        gets warmed.
        """
        with self._lock:
            return self._descriptors_locked(agent_id)

    def agent_size(self, agent_id: str) -> int:
        """How many prefixes a prefetch for ``agent_id`` would find.

        Counts pattern buckets too, because this is what the endpoint reports
        as `available_prefixes` and gates its "nothing to warm" warning on --
        a seed reachable only through a pattern must not read as nothing.
        """
        with self._lock:
            return len(self._descriptors_locked(agent_id))

    def num_agents(self) -> int:
        with self._lock:
            return len(self._by_agent)

    def stats(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "num_agents": len(self._by_agent),
                "total_descriptors": sum(
                    len(m) for m in self._by_agent.values()
                ),
                "max_agents": self.max_agents,
                # ``None`` is JSON-serialized as ``null`` ≡ unlimited.
                "max_per_agent": self.max_per_agent,
                "default_top_k": self.default_top_k,
                "num_patterns": len(self._patterns),
            }

    # -- internal ---------------------------------------------------------

    def _maybe_evict_agent_locked(self) -> None:
        """Evict the LRU agent if we've exceeded ``max_agents``.

        Called with ``self._lock`` held. The agent we just inserted is
        at MRU, so ``popitem(last=False)`` cannot evict it.
        """
        while len(self._by_agent) > self.max_agents:
            evicted_agent, _ = self._by_agent.popitem(last=False)
            self._patterns.pop(evicted_agent, None)
            logger.debug(
                "agent_prefetch: evicted agent %s (cross-agent LRU)",
                evicted_agent,
            )
