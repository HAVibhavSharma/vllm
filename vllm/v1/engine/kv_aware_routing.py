# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-aware placement of requests between data-parallel engines.

vLLM's internal load balancer (``DPLBAsyncMPClient``) sends each request to
the engine with the least load and does not know which engine's GPU holds the
request's prefix. An agent's next call shares most of its prompt with its
previous one, so a call placed on the other engine reloads that prefix from
LMCache, and beside another call's long prefill the reload waits out its
steps: in pitlane run swe_lite10_dp2_lbfix_4q_20261126_181834, Q4's last call
in arm A left a 28k-token prefix on GPU 1, landed on GPU 0 during a 24k-token
prefill and waited 14.6 s, against 0.5 s where it stayed.

With ``VLLM_DP_KV_ROUTING=1`` the router remembers, per engine, the prompts it
sent there, as chained hashes of ``CHUNK_TOKENS``-token chunks, forgetting the
least recently sent once they exceed that engine's KV cache. A request goes to
the least loaded engine; among equally loaded ones, to the one holding the
longest prefix of it, when that prefix is at least
``VLLM_DP_KV_ROUTING_MIN_TOKENS`` long (a system prompt every question shares
is shorter). A busier engine is never picked for its prefix: a call sharing a
GPU decodes slower for its whole run, while its prefix reloads from LMCache's
CPU memory, where arm A's prefetch puts it, in about 0.3 s for 28k tokens.
``VLLM_DP_KV_ROUTING_SLACK`` (default 0) lets the prefix's engine be that many
calls busier.

Load is the greater of this client's agent calls in flight and the engine's
waiting + running, as in the load balancer, but phantoms (``prefetch_only``)
are not counted: they compute nothing, and a staged one stays in flight until
its call arrives, so counting them pushes calls away from the engine that
holds their prefix. Phantoms are placed by the same rule and their prompts
are remembered too, so a call follows the phantom that warmed its prefix.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLE_ENV = "VLLM_DP_KV_ROUTING"
MIN_TOKENS_ENV = "VLLM_DP_KV_ROUTING_MIN_TOKENS"
SLACK_ENV = "VLLM_DP_KV_ROUTING_SLACK"
MIN_TOKENS_DEFAULT = 4096
SLACK_DEFAULT = 0
CHUNK_TOKENS = 256


def is_phantom(request: Any) -> bool:
    """A prefetch-only request (the fork's phantoms), by the flag the
    scheduler's ``_is_prefetch_only_request`` reads."""
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    kv = extra.get("kv_transfer_params") if isinstance(extra, dict) else None
    return bool(isinstance(kv, dict) and kv.get("prefetch_only"))


def chunk_hashes(
    token_ids: Sequence[int] | None, chunk: int = CHUNK_TOKENS
) -> list[int]:
    """Chained hashes of the full ``chunk``-token chunks of a prompt: the
    i-th names the prompt's first (i + 1) * chunk tokens."""
    if not token_ids:
        return []
    out: list[int] = []
    h = 0
    for start in range(0, len(token_ids) - chunk + 1, chunk):
        h = hash((h, tuple(token_ids[start:start + chunk])))
        out.append(h)
    return out


class KVAwareRouter:
    """Per-engine memory of placed prompts, and the placement rule."""

    def __init__(
        self,
        num_engines: int,
        capacity_tokens: Callable[[], int],
        min_tokens: int = MIN_TOKENS_DEFAULT,
        slack: int = SLACK_DEFAULT,
        chunk: int = CHUNK_TOKENS,
    ):
        self.num_engines = num_engines
        # Per engine, read when needed: the KV cache size reaches the
        # front-end with the engines' ready responses.
        self.capacity_tokens = capacity_tokens
        self.min_tokens = min_tokens
        self.slack = slack
        self.chunk = chunk
        self.prefixes: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(num_engines)
        ]
        self.calls = [0] * num_engines
        self.call_engine: dict[str, int] = {}

    @classmethod
    def from_env(
        cls, num_engines: int, capacity_tokens: Callable[[], int]
    ) -> KVAwareRouter | None:
        if os.getenv(ENABLE_ENV, "0") != "1":
            return None
        router = cls(
            num_engines,
            capacity_tokens,
            min_tokens=int(os.getenv(MIN_TOKENS_ENV, str(MIN_TOKENS_DEFAULT))),
            slack=int(os.getenv(SLACK_ENV, str(SLACK_DEFAULT))),
        )
        logger.info(
            "kv_route: on for %d engines (prefix of at least %d tokens, load "
            "slack %d, %d-token chunks)",
            num_engines, router.min_tokens, router.slack, router.chunk,
        )
        return router

    def matched_tokens(self, engine: int, hashes: Sequence[int]) -> int:
        held = self.prefixes[engine]
        n = 0
        for h in hashes:
            if h not in held:
                break
            n += 1
        return n * self.chunk

    def loads(
        self, counts: Sequence[Sequence[float]], client_count: int
    ) -> list[int]:
        """Agent calls per engine: this client's in flight, or the engine's
        waiting + running if more (phantoms are in neither)."""
        return [
            max(client_count * self.calls[i], int(counts[i][0]) + int(counts[i][1]))
            for i in range(self.num_engines)
        ]

    def choose(
        self,
        request: Any,
        counts: Sequence[Sequence[float]],
        client_count: int,
        fallback: int,
    ) -> tuple[int, list[int]]:
        """The engine for ``request``, and its prompt's chunk hashes.
        ``fallback`` is the load balancer's own pick, kept when it is among
        the least loaded."""
        hashes = chunk_hashes(request.prompt_token_ids, self.chunk)
        loads = self.loads(counts, client_count)
        least = min(loads)
        matched = [self.matched_tokens(i, hashes) for i in range(self.num_engines)]
        best = max(range(self.num_engines), key=lambda i: (matched[i], -loads[i]))
        if matched[best] >= self.min_tokens and loads[best] <= least + self.slack:
            engine, why = best, "prefix"
        else:
            engine = fallback if loads[fallback] == least else loads.index(least)
            why = "load"
        (logger.debug if is_phantom(request) else logger.info)(
            "kv_route: %s -> engine %d by %s (prompt %d tokens, prefix held "
            "per engine %s, agent calls per engine %s)",
            request.request_id, engine, why,
            len(request.prompt_token_ids or ()), matched, loads,
        )
        return engine, hashes

    def placed(
        self, request: Any, engine: int, hashes: list[int] | None = None
    ) -> None:
        """Remember ``request``'s prompt on ``engine``, and count it in flight
        there unless it is a phantom."""
        if hashes is None:
            hashes = chunk_hashes(request.prompt_token_ids, self.chunk)
        held = self.prefixes[engine]
        # Head last, so it is the most recent: a cache keeps a shared prefix
        # longer than the tails behind it.
        for h in reversed(hashes):
            held[h] = None
            held.move_to_end(h)
        capacity = self.capacity_tokens() // self.chunk
        if capacity > 0:
            while len(held) > capacity:
                held.popitem(last=False)
        if not is_phantom(request):
            self.calls[engine] += 1
            self.call_engine[request.request_id] = engine

    def finished(self, request_id: str) -> None:
        engine = self.call_engine.pop(request_id, None)
        if engine is not None:
            self.calls[engine] -= 1
