# SPDX-License-Identifier: Apache-2.0
"""Which data-parallel engine a question's requests go to, and when seeds run.

Under ``--data-parallel-size`` the front-end's load balancer places every
request on its own. For a dense model it never hears the engines' queue
lengths (only the MoE engine class publishes them), so it balances the number
of requests it assigned in the last few seconds, not the load. And phantoms
count there like real calls, which moves the calls of an arm that sends them.

With ``VLLM_AGENT_DP_PIN=1`` a question keeps one engine instead:

* the harness names each job's engine when the question starts
  (``POST /v1/agents/dp_rank``), and sends its chat calls with
  ``X-data-parallel-rank``;
* a phantom of that job goes to the same engine (``phantom_rank``), since
  what it warms is that engine's cache;
* a prompt seed (request id ``...::seed-...``) is prefill done early for a
  later call, and it should only use GPU time no agent call needs. So it waits
  (``place_seed``) until some engine is idle -- no agent call in flight and no
  other seed -- and runs there, on its own question's engine if both are idle.
  A seed still waiting when its target call arrives, or when its question
  ends, is dropped: the call computes its prompt itself, as without seeds.

A seed that started in an idle gap still runs to its end if an agent call
arrives on that engine meanwhile; one seed per engine at a time keeps that
overlap to one seed's remaining prefill.

The map lives in the API server's process, which is the only place a request
is placed; run it with ``--api-server-count 1``.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

SEED_MARK = "::seed-"
# How often a waiting seed looks at the engines again.
POLL_S = 0.05
# A seed placed on an engine counts as running there until vLLM lists it in
# flight, or for this long if it never gets that far (rejected while rendering).
RESERVE_S = 2.0

_lock = threading.RLock()
_rank_of_job: dict[str, int] = {}
# agent_id -> wall-clock time the latest agent call carrying it arrived.
_last_call_at: dict[str, float] = {}
# engine index -> (seed request id, monotonic time it was placed there).
_seed_placed: dict[int, tuple[str, float]] = {}


def enabled() -> bool:
    return os.getenv("VLLM_AGENT_DP_PIN", "0") == "1"


def set_job_rank(job_id: str, rank: int) -> None:
    """Pin ``job_id`` to engine ``rank``; a negative rank removes the pin."""
    with _lock:
        if rank < 0:
            _rank_of_job.pop(str(job_id), None)
        else:
            _rank_of_job[str(job_id)] = int(rank)


def job_rank(job_id: str | None) -> int | None:
    if job_id is None:
        return None
    with _lock:
        return _rank_of_job.get(str(job_id))


def job_of(request_or_agent_id: str) -> str | None:
    """The job a LangGraph agent id names (``langgraph:<job>:...``), wherever
    it sits in a request id (``chatcmpl-prefetch::langgraph:5:...::seed-...``)."""
    marker = "langgraph:"
    at = request_or_agent_id.find(marker)
    if at < 0:
        return None
    job = request_or_agent_id[at + len(marker):].split(":", 1)[0]
    return job if job and job != "*" else None


def num_engines(engine_client: Any) -> int | None:
    core = getattr(engine_client, "engine_core", None)
    engines = getattr(core, "core_engines", None)
    return len(engines) if engines else None


def phantom_rank(job_id: str | None) -> int | None:
    """The engine a phantom of ``job_id`` goes to, or None to leave it to the
    load balancer (pinning off, or a job nobody named)."""
    return job_rank(job_id) if enabled() else None


def is_seed(request_id: str) -> bool:
    return SEED_MARK in request_id


def seed_agent_id(request_id: str) -> str | None:
    """The target's agent id in ``chatcmpl-prefetch::<agent_id>::seed-<hex>``."""
    head, mark, _ = request_id.rpartition(SEED_MARK)
    _, sep, agent_id = head.partition("prefetch::")
    return agent_id if mark and sep and agent_id else None


def note_call(agent_id: Any) -> None:
    """An agent call carrying ``agent_id`` arrived: seeds for it issued
    before now are stale."""
    if enabled() and isinstance(agent_id, str) and agent_id:
        with _lock:
            _last_call_at[agent_id] = time.time()


def _is_agent_call(request_id: str) -> bool:
    return request_id.startswith("chatcmpl-") and "prefetch::" not in request_id


def _load(engines: list[Any], in_flight: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Agent calls and seeds in flight per engine (arrived, not finished, so a
    call still waiting for its KV counts). Phantoms compute nothing and count
    as neither."""
    index = {engine: i for i, engine in enumerate(engines)}
    calls = [0] * len(engines)
    seeds = [0] * len(engines)
    for rid, engine in list(in_flight.items()):
        i = index.get(engine)
        if i is None:
            continue
        if is_seed(rid):
            seeds[i] += 1
        elif _is_agent_call(rid):
            calls[i] += 1
    return calls, seeds


def _seed_running(i: int, seeds_in_flight: list[int], now: float) -> bool:
    if seeds_in_flight[i]:
        _seed_placed.pop(i, None)
        return True
    placed = _seed_placed.get(i)
    if placed is not None and now - placed[1] < RESERVE_S:
        return True
    _seed_placed.pop(i, None)
    return False


async def place_seed(
    engine_client: Any, request_id: str, issued_at: float | None = None
) -> tuple[int | None, str | None]:
    """Wait for an idle engine for the seed ``request_id``.

    Returns ``(rank, None)`` to send it to engine ``rank``; ``(None, reason)``
    to drop it (its target call arrived, or its question ended, while it
    waited); ``(None, None)`` for anything this does not handle (not a seed,
    pinning off, one engine), which the load balancer then places.
    ``issued_at`` is the wall-clock time the harness issued the seed: a target
    call arriving after that makes it stale, even one that arrived before the
    seed reached the server (seeds of one question are sent one at a time).
    """
    if not enabled() or not is_seed(request_id):
        return None, None
    core = getattr(engine_client, "engine_core", None)
    engines = getattr(core, "core_engines", None)
    in_flight = getattr(core, "reqs_in_flight", None)
    if not engines or in_flight is None or len(engines) < 2:
        return None, None
    since = issued_at if issued_at is not None else time.time()
    waited_from = time.monotonic()
    agent_id = seed_agent_id(request_id)
    job = job_of(request_id)
    had_job = job_rank(job) is not None
    while True:
        with _lock:
            own = _rank_of_job.get(job) if job is not None else None
            if agent_id is not None and _last_call_at.get(agent_id, float("-inf")) >= since:
                reason = "its call arrived"
            elif had_job and own is None:
                reason = "its question ended"
            else:
                reason = None
                calls, seeds = _load(engines, in_flight)
                now = time.monotonic()
                idle = [i for i in range(len(engines))
                        if calls[i] == 0 and not _seed_running(i, seeds, now)]
                if idle:
                    rank = own if own in idle else idle[0]
                    _seed_placed[rank] = (request_id, now)
                    logger.info(
                        "agent_dp: seed %s -> engine %d after waiting %.2f s "
                        "(agent calls in flight %s, its job's engine %s)",
                        request_id, rank, now - waited_from, calls, own,
                    )
                    return rank, None
        if reason is not None:
            logger.info(
                "agent_dp: seed %s dropped after waiting %.2f s: %s",
                request_id, time.monotonic() - waited_from, reason,
            )
            return None, reason
        await asyncio.sleep(POLL_S)
