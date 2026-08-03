# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Front-end half of prefetch origination — 02 §4 option a, build step 6.

Engine core decides *what* to prefetch (`node_eviction/wantlist.py`) but
cannot submit it: `EngineCoreOutputs` carries per-request output, not
requests, and the only object that can originate one is the `EngineClient`
living in this process. So a background task polls the engine's want-list
over the existing `call_utility` channel and fans each want out through the
existing `PhantomPrefetchSubmitter`.

Nothing here decides anything. It resolves `agent_id` to the prefixes the
registry has seen for it and submits phantoms. If the registry has never seen
that agent, the want is dropped — the engine can want a prefix that this
process has no tokens for, and inventing one would prefetch the wrong bytes.

**Every failure path is a no-op.** A drain that raises, an engine that does
not implement the utility method, an agent with no prefixes: all of them mean
"no prefetch this round", never a broken server. Prefetching is an
optimisation, and an optimisation that can take down the serving path is a
regression whatever it does to hit rate.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.v1.agent_prefetch.registry import AgentPrefixRegistry
    from vllm.v1.agent_prefetch.submitter import PhantomPrefetchSubmitter

logger = init_logger(__name__)


class PrefetchWantDrainer:
    """Polls engine core's want-list and submits phantoms for it."""

    def __init__(
        self,
        engine_client: "EngineClient",
        registry: "AgentPrefixRegistry",
        submitter: "PhantomPrefetchSubmitter",
        *,
        interval_s: float = 1.0,
        max_per_drain: int = 4,
        top_k_per_want: int | None = None,
    ) -> None:
        self._engine_client = engine_client
        self._registry = registry
        self._submitter = submitter
        self._interval_s = max(interval_s, 0.05)
        self._max_per_drain = max_per_drain
        # None means "every prefix the registry holds for the agent", the
        # same default the /v1/agents/prefetch endpoint uses.
        self._top_k_per_want = top_k_per_want
        self._task: asyncio.Task | None = None
        self._stats = {
            "polls": 0,
            "wants_received": 0,
            "wants_unknown_agent": 0,
            "phantoms_submitted": 0,
            "errors": 0,
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="prefetch_want_drainer")
        logger.info(
            "agent_prefetch: want-list drainer started "
            "(interval=%.2fs, max_per_drain=%d)",
            self._interval_s,
            self._max_per_drain,
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # -- the loop ----------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad poll must not kill the loop: the engine may be
                # mid-shutdown, or a want may be malformed. Log and keep
                # going, because silently stopping would look exactly like
                # "the forecast predicted nothing".
                self._stats["errors"] += 1
                logger.exception("agent_prefetch: want-list drain failed")

    async def drain_once(self) -> int:
        """One poll. Returns the number of phantoms submitted."""
        self._stats["polls"] += 1
        drain = getattr(self._engine_client, "drain_prefetch_wants", None)
        if drain is None:
            return 0
        wants = await drain(self._max_per_drain)
        if not wants:
            return 0
        self._stats["wants_received"] += len(wants)

        submitted = 0
        for want in wants:
            submitted += await self._submit_want(want)
        if submitted:
            # INFO, not DEBUG: this is the only line that says origination is
            # actually working, and it is rare by construction -- a want is
            # deduplicated until it lands or its backoff expires, so this
            # fires on the order of once per node per minute, not per tick.
            logger.info(
                "agent_prefetch: drained %d want(s) %s, submitted %d phantom(s)",
                len(wants),
                [
                    f"{w.get('agent_id')}@{w.get('time_to_next_call_ms')}ms"
                    for w in wants
                ],
                submitted,
            )
        return submitted

    async def _submit_want(self, want: dict[str, Any]) -> int:
        agent_id = str(want.get("agent_id") or "")
        if not agent_id:
            return 0

        if self._top_k_per_want is None:
            descriptors = self._registry.get_all(agent_id)
        else:
            descriptors = self._registry.top_k(agent_id, k=self._top_k_per_want)
        if not descriptors:
            # The engine wants a node this process has never served, so there
            # is no prefix to warm. Expected during the first turn of a job
            # and on any node whose preamble has not been recorded yet.
            self._stats["wants_unknown_agent"] += 1
            # Also INFO: this is the single most likely reason origination
            # looks switched on and does nothing, and it is bounded by the
            # want backoff. Expected on a job's first turn through a node.
            logger.info(
                "agent_prefetch: want for %s has no registered prefixes yet "
                "-- nothing to warm",
                agent_id,
            )
            return 0

        # Carried into the phantom's extra_args so the blocks it caches are
        # indexed under the key the policy predicted, and therefore stamped
        # speculative and protected by the decaying floor (02 §5 part 1).
        # Without this the phantom is anonymous and the floor never engages.
        identity = {
            "job_id": str(want.get("job_id", "")),
            "langgraph_node": str(want.get("node", "")),
            "call_type": str(want.get("call_type", "")),
        }

        submitted = 0
        for desc in descriptors:
            task = await self._submitter.submit(
                agent_id=agent_id,
                token_ids=desc.token_ids,
                prefix_hash=desc.prefix_hash,
                cache_salt=desc.cache_salt,
                identity=identity,
            )
            if task is not None:
                submitted += 1
        self._stats["phantoms_submitted"] += submitted
        return submitted

    def stats(self) -> dict[str, int]:
        return dict(self._stats)


def maybe_start_prefetch_drainer(app) -> PrefetchWantDrainer | None:
    """Start the drainer if `VLLM_NODE_EVICTION_PREFETCH_DRAIN` is set.

    One env var gates both halves: engine core only builds a want-list when
    it is set, and this process only drains when it is set. Splitting them
    would let the engine accumulate wants nobody submits, which reads as a
    broken forecast rather than a configuration mistake.

    Returns None (having logged why) whenever it cannot start, so the server
    comes up either way.
    """
    from vllm import envs

    if not envs.VLLM_NODE_EVICTION_PREFETCH_DRAIN:
        return None

    state = app.state
    chat_handler = getattr(state, "openai_serving_chat", None)
    if chat_handler is None:
        logger.warning(
            "VLLM_NODE_EVICTION_PREFETCH_DRAIN is set but this server has no "
            "chat handler; prefetch origination is off."
        )
        return None

    from vllm.entrypoints.openai.agent_chat.api_router import (
        get_or_init_agent_prefetch_state,
    )

    registry, submitter = get_or_init_agent_prefetch_state(
        state, chat_handler.engine_client
    )
    drainer = PrefetchWantDrainer(
        chat_handler.engine_client,
        registry,
        submitter,
        interval_s=envs.VLLM_NODE_EVICTION_PREFETCH_DRAIN_INTERVAL_S,
    )
    drainer.start()
    state.prefetch_want_drainer = drainer
    return drainer
