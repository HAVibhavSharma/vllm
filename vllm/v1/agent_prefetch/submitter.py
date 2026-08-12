# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phantom-prefetch submitter.

Fires fire-and-forget prefetch requests into the engine. Each phantom
request:

* Carries ``kv_transfer_params={"prefetch_only": True, "cache_salt": ...}``.
* Uses ``max_tokens=1`` and ``temperature=0`` -- the smallest legal
  generation request. The scheduler's prefetch_only hook in
  ``_update_from_kv_xfer_finished`` finalizes the request and emits a
  terminal output as soon as the LMCache load completes, so no
  sampling step actually runs.
* Has a request id of the form ``prefetch::<agent_id>::<short_hash>``
  for idempotent dedup -- if a phantom for the same prefix is already
  in flight, a second submit is a no-op.

The submitter does **not** await the generator; it spawns an asyncio
task that drains it. This way the caller (the agent-chat endpoint) is
not blocked on the prefetch finishing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)


def _short_hex(prefix_hash: bytes, n: int = 16) -> str:
    return prefix_hash.hex()[:n]


def build_prefetch_request_id(agent_id: str, prefix_hash: bytes) -> str:
    """Build a deterministic request id for a phantom prefetch.

    Deterministic so concurrent calls for the same (agent, prefix)
    collapse to one in-flight request via the engine's request id
    uniqueness check.
    """
    return f"prefetch::{agent_id}::{_short_hex(prefix_hash)}"


class PhantomPrefetchSubmitter:
    """Submits fire-and-forget prefetch requests into the engine."""

    def __init__(
        self,
        engine_client: "EngineClient",
        *,
        max_inflight_per_agent: int = 64,
    ) -> None:
        self._engine_client = engine_client
        self._max_inflight_per_agent = max_inflight_per_agent
        # Per-agent in-flight tracking. Used only for the local
        # idempotency guard and observability; the engine itself
        # rejects duplicate request_ids if a true collision occurs.
        self._inflight: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        *,
        agent_id: str,
        token_ids: Sequence[int],
        prefix_hash: bytes,
        cache_salt: str,
        identity: dict[str, str] | None = None,
    ) -> asyncio.Task | None:
        """Submit one phantom prefetch.

        Returns the spawned ``asyncio.Task`` on success, or ``None`` if
        the request was skipped (already in flight or quota exceeded).
        Callers can either ignore the task (fire-and-forget) or
        ``await`` it to block until the phantom completes -- at which
        point the prefix is guaranteed to be registered in APC.

        ``identity`` carries ``job_id`` / ``langgraph_node`` / ``call_type``
        into ``sampling_params.extra_args``. Without it the blocks a phantom
        caches are **invisible to the node-eviction policy**:
        ``node_key_for_request`` returns None for a request with no identity,
        so ``on_blocks_cached`` returns early and the entry is never stamped
        speculative -- leaving the decaying protection floor (02 §5) inert for
        exactly the blocks it exists to protect. Callers that know which node
        they are warming for should always pass it.

        Never raises; all failures inside the spawned task are logged
        and absorbed -- prefetch is best-effort and must not break the
        real call.
        """
        request_id = build_prefetch_request_id(agent_id, prefix_hash)

        async with self._lock:
            inflight_for_agent = self._inflight.setdefault(agent_id, set())
            if request_id in inflight_for_agent:
                return None
            if len(inflight_for_agent) >= self._max_inflight_per_agent:
                # `warning_once` per agent, not `debug`: silently dropping
                # prefetches is indistinguishable from a forecast that
                # predicted nothing, and the cap being hit at all means the
                # fan-out is outrunning the engine. Once per agent keeps a
                # saturated agent from flooding the log with the same fact.
                logger.warning_once(
                    "agent_prefetch: in-flight cap (%d) reached for agent %s "
                    "-- dropping prefetches for it. Further drops for this "
                    "agent are not logged.",
                    self._max_inflight_per_agent,
                    agent_id,
                )
                logger.debug(
                    "agent_prefetch: dropped prefetch %s (in-flight cap)",
                    request_id,
                )
                return None
            inflight_for_agent.add(request_id)

        extra_args: dict[str, Any] = {
            "kv_transfer_params": {
                "prefetch_only": True,
                "cache_salt": cache_salt,
            },
        }
        if identity:
            # Top-level, beside kv_transfer_params -- the same shape a real
            # request carries, which is what lets one `node_key_for_request`
            # serve both.
            extra_args.update(identity)

        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extra_args=extra_args,
        )

        task = asyncio.create_task(
            self._run_one(
                request_id=request_id,
                token_ids=list(token_ids),
                params=params,
                agent_id=agent_id,
            ),
            name=f"agent_prefetch:{request_id}",
        )
        # Avoid the task being silently GC'd before completion.
        task.add_done_callback(lambda _t: None)
        return task

    async def _run_one(
        self,
        *,
        request_id: str,
        token_ids: list[int],
        params: SamplingParams,
        agent_id: str,
    ) -> None:
        """Drain the engine generator. Exceptions are logged, never raised."""
        try:
            prompt: dict[str, Any] = {"prompt_token_ids": token_ids}
            # The success path used to be silent, which left `submitted` /
            # `completed` from the HTTP layer as the only evidence a phantom
            # existed — and those count asyncio tasks, not engine requests. A
            # phantom that never reaches the scheduler is then
            # indistinguishable from one that ran: both report
            # `submitted=1 completed=1`. Debug, not info: at real want volume
            # this is one line per phantom.
            logger.debug(
                "agent_prefetch: submitting phantom %s (%d prompt tokens, "
                "identity=%s)",
                request_id,
                len(token_ids),
                params.extra_args,
            )
            gen = self._engine_client.generate(prompt, params, request_id)
            # Consume the generator until the engine finishes. We don't
            # use the outputs; the engine handles APC registration in
            # _finalize_prefetch_only_request before sending the final
            # event, so once the generator completes, the prefix is
            # cached.
            async for _ in gen:
                pass
            logger.debug(
                "agent_prefetch: phantom %s finished in engine", request_id
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Best-effort: prefetch must never break the host process.
            logger.exception(
                "agent_prefetch: phantom prefetch %s failed", request_id
            )
        finally:
            async with self._lock:
                inflight = self._inflight.get(agent_id)
                if inflight is not None:
                    inflight.discard(request_id)
                    if not inflight:
                        self._inflight.pop(agent_id, None)

    def inflight_size(self, agent_id: str | None = None) -> int:
        if agent_id is None:
            return sum(len(s) for s in self._inflight.values())
        return len(self._inflight.get(agent_id, ()))
