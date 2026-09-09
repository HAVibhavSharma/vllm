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
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)

# A phantom's *end* is otherwise unobservable. The HTTP access line is the
# submit (the endpoint answers with `wait=False`), and `kv_hbm_ttft` skips
# phantoms on purpose -- `NodeEvictionController.on_request_finished` gates on
# `ttft_chat_completions_only` so the policy's own warming traffic cannot
# flatter its TTFT average. So without this there is no way to tell a warm
# that landed before its request from one the request waited behind, which is
# exactly the difference between "the prefetch bought lead" and "the prefetch
# moved the prefill 90ms earlier and the request queued behind it".
#
# On by default because the question above is the one these runs are for; set
# `VLLM_PREFETCH_LOG_SPANS=0` to fall back to debug at real want volume, where
# this is one line per phantom.
_LOG_SPANS = os.getenv("VLLM_PREFETCH_LOG_SPANS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
_span_log = logger.info if _LOG_SPANS else logger.debug


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
        prefill_on_miss: bool = True,
    ) -> asyncio.Task | None:
        """Submit one phantom prefetch.

        Returns the spawned ``asyncio.Task`` on success, or ``None`` if
        the request was skipped (already in flight or quota exceeded).
        Callers can either ignore the task (fire-and-forget) or
        ``await`` it to block until the phantom completes -- at which
        point the prefix is guaranteed to be registered in APC.

        ``prefill_on_miss`` lets this phantom fall through to a real prefill
        when LMCache has nothing for the prefix, rather than being aborted.
        On by default: the scheduler holds such a phantom out of any step
        with real work running, so its prefill lands in idle time or not at
        all. See ``AgentPrefetchRequest.prefill_on_miss``.

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
                # Read by `Scheduler._prefetch_may_prefill_on_miss`. True
                # lets a miss prefill once, which the store path writes into
                # LMCache so later prefetches have something to promote --
                # and the scheduler admits it only into an idle step. False
                # restores the older contract, where an LMCache miss aborts
                # before ever reaching the model.
                "prefill_on_miss": bool(prefill_on_miss),
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
            # `submitted=1 completed=1`. Level is `_span_log` -- see
            # `VLLM_PREFETCH_LOG_SPANS`.
            started = time.perf_counter()
            _span_log(
                "agent_prefetch_start req=%s prompt_tokens=%d agent=%s "
                "identity=%s",
                request_id,
                len(token_ids),
                agent_id,
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
            # The line's timestamp is the phantom's end, and `elapsed_ms`
            # carries its start, so a single line pairs against the target's
            # access line without joining two. `_finalize_prefetch_only_request`
            # has already registered the prefix in APC by the time the
            # generator completes, so this instant is also when the blocks
            # became hittable.
            _span_log(
                "agent_prefetch_end req=%s prompt_tokens=%d agent=%s "
                "elapsed_ms=%.1f",
                request_id,
                len(token_ids),
                agent_id,
                (time.perf_counter() - started) * 1000.0,
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
