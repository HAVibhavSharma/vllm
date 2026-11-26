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

With ``VLLM_DP_KV_ROUTING=1`` a request still goes to the least loaded engine
(``VLLM_DP_KV_ROUTING_SLACK``, default 0, lets an engine that many calls
busier count as tied): a call sharing a GPU decodes slower for its whole run,
while its prefix reloads from LMCache's CPU memory, where arm A's prefetch puts
it, in about 0.3 s for 28k tokens. Among the tied engines:

* one whose in-flight calls still have more than
  ``VLLM_DP_KV_ROUTING_MAX_PREFILL`` uncached prompt tokens to prefill is left
  out, since a call beside a long prefill waits for its steps and decodes
  slower (36 to 50 ms a token against 29);
* so is one whose free KV cache, from the usage it reports, cannot hold what
  the request still needs;
* of the rest, the one holding the longest prefix of the request gets it, when
  that prefix is at least ``VLLM_DP_KV_ROUTING_MIN_TOKENS`` long (a system
  prompt every question shares is shorter); otherwise the one with the least
  prefill ahead.

The router remembers, per engine, the prompts it sent there, as chained hashes
of ``CHUNK_TOKENS``-token chunks, forgetting the least recently sent past that
engine's KV cache. Load leaves out phantoms (``prefetch_only``): they compute
nothing, and a staged one stays in flight until its call arrives. A phantom
that warms the GPU is remembered like a call, so a call follows it; a staged
one (``stage_l1``) only reads into LMCache's L1, shared by every engine, and
is not.

A staged phantom ends when its call reaches the engine it waits on; a call
placed on another engine left it holding L1 read locks until its question
ended (7 of 15 in the run above, 63k tokens). So a call releases its
question's phantoms of the same node waiting on other engines:
``take_stranded`` hands their ids to ``AsyncLLM``, which aborts them.
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
MAX_PREFILL_ENV = "VLLM_DP_KV_ROUTING_MAX_PREFILL"
MIN_TOKENS_DEFAULT = 4096
SLACK_DEFAULT = 0
MAX_PREFILL_DEFAULT = 8192
CHUNK_TOKENS = 256


def _kv_params(request: Any) -> dict:
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    kv = extra.get("kv_transfer_params") if isinstance(extra, dict) else None
    return kv if isinstance(kv, dict) else {}


def is_phantom(request: Any) -> bool:
    """A prefetch-only request (the fork's phantoms), by the flag the
    scheduler's ``_is_prefetch_only_request`` reads."""
    return bool(_kv_params(request).get("prefetch_only"))


def is_staged(request: Any) -> bool:
    """A phantom that only reads its prefix into LMCache's L1 (the
    scheduler's ``_is_staged_phantom``)."""
    kv = _kv_params(request)
    return bool(kv.get("prefetch_only") and kv.get("stage_l1"))


def agent_key(request: Any) -> tuple[str, str] | None:
    """``(job_id, agent_id)``, as the scheduler's ``_agent_key``: the same for
    a node's call and the phantoms warming it."""
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    if not isinstance(extra, dict):
        return None
    job_id, agent_id = extra.get("job_id"), extra.get("agent_id")
    if job_id is None or not agent_id or "*" in str(agent_id):
        return None
    return str(job_id), str(agent_id)


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
        h = hash((h, tuple(token_ids[start : start + chunk])))
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
        max_prefill: int = MAX_PREFILL_DEFAULT,
        chunk: int = CHUNK_TOKENS,
    ):
        self.num_engines = num_engines
        # Per engine, read when needed: the KV cache size reaches the
        # front-end with the engines' ready responses.
        self.capacity_tokens = capacity_tokens
        self.min_tokens = min_tokens
        self.slack = slack
        self.max_prefill = max_prefill
        self.chunk = chunk
        self.prefixes: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(num_engines)
        ]
        self.calls = [0] * num_engines
        self.call_engine: dict[str, int] = {}
        # Calls with no token back yet: request id -> (engine, uncached
        # prompt tokens when placed).
        self.prefilling: dict[str, tuple[int, int]] = {}
        # Phantoms in flight: request id -> (engine, agent key).
        self.phantoms: dict[str, tuple[int, tuple[str, str] | None]] = {}
        self.stranded: list[str] = []

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
            max_prefill=int(os.getenv(MAX_PREFILL_ENV, str(MAX_PREFILL_DEFAULT))),
        )
        logger.info(
            "kv_route: on for %d engines (prefix of at least %d tokens, load "
            "slack %d, prefill limit %d tokens, %d-token chunks)",
            num_engines, router.min_tokens, router.slack, router.max_prefill,
            router.chunk,
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

    def prefill_ahead(self, engine: int) -> int:
        """Uncached prompt tokens of the calls on ``engine`` with no token
        back yet."""
        return sum(t for e, t in self.prefilling.values() if e == engine)

    def free_tokens(
        self, engine: int, counts: Sequence[Sequence[float]]
    ) -> int | None:
        """KV cache tokens ``engine`` reports free, or None if unknown."""
        capacity = self.capacity_tokens()
        row = counts[engine]
        if capacity <= 0 or len(row) < 3:
            return None
        return int((1.0 - float(row[2])) * capacity)

    def choose(
        self,
        request: Any,
        counts: Sequence[Sequence[float]],
        client_count: int,
        fallback: int,
    ) -> tuple[int, list[int]]:
        """The engine for ``request``, and its prompt's chunk hashes.
        ``fallback`` is the load balancer's own pick, kept when nothing above
        tells the tied engines apart."""
        hashes = chunk_hashes(request.prompt_token_ids, self.chunk)
        prompt = len(request.prompt_token_ids or ())
        n = self.num_engines
        loads = self.loads(counts, client_count)
        least = min(loads)
        tied = [i for i in range(n) if loads[i] <= least + self.slack]
        matched = [self.matched_tokens(i, hashes) for i in range(n)]
        ahead = [self.prefill_ahead(i) for i in range(n)]
        free = [self.free_tokens(i, counts) for i in range(n)]
        fits = [free[i] is None or prompt - matched[i] <= free[i] for i in range(n)]
        calm = [ahead[i] <= self.max_prefill for i in range(n)]
        # Room first: without it the request waits for GPU space.
        ok = (
            [i for i in tied if fits[i] and calm[i]]
            or [i for i in tied if fits[i]]
            or [i for i in tied if calm[i]]
            or tied
        )
        best = max(ok, key=lambda i: (matched[i], -loads[i], -ahead[i]))
        if matched[best] >= self.min_tokens:
            engine, why = best, "prefix"
        else:
            rank = {i: (loads[i], ahead[i]) for i in ok}
            top = min(rank.values())
            engine = fallback if rank.get(fallback) == top else min(
                (i for i in ok if rank[i] == top)
            )
            why = "load"
        log = logger.debug if is_phantom(request) else logger.info
        log(
            "kv_route: %s -> engine %d by %s (prompt %d tokens, prefix held per "
            "engine %s, agent calls %s, prefill ahead %s, free KV %s)",
            request.request_id, engine, why, prompt, matched, loads, ahead, free,
        )
        return engine, hashes

    def placed(
        self, request: Any, engine: int, hashes: list[int] | None = None
    ) -> None:
        """Remember ``request`` on ``engine``. A call is counted in flight
        there and releases its node's phantoms on other engines; a phantom is
        remembered for that release, and its prompt only if it warms the GPU."""
        if hashes is None:
            hashes = chunk_hashes(request.prompt_token_ids, self.chunk)
        phantom = is_phantom(request)
        if not phantom:
            uncached = len(request.prompt_token_ids or ()) - self.matched_tokens(
                engine, hashes
            )
            self.prefilling[request.request_id] = (engine, max(0, uncached))
        if not is_staged(request):
            held = self.prefixes[engine]
            # Head last, so it is the most recent: a cache keeps a shared
            # prefix longer than the tails behind it.
            for h in reversed(hashes):
                held[h] = None
                held.move_to_end(h)
            capacity = self.capacity_tokens() // self.chunk
            if capacity > 0:
                while len(held) > capacity:
                    held.popitem(last=False)
        key = agent_key(request)
        if phantom:
            self.phantoms[request.request_id] = (engine, key)
            return
        self.calls[engine] += 1
        self.call_engine[request.request_id] = engine
        if key is None:
            return
        away = [
            rid for rid, (e, k) in self.phantoms.items() if k == key and e != engine
        ]
        for rid in away:
            del self.phantoms[rid]
        if away:
            self.stranded.extend(away)
            logger.info(
                "kv_route: call %s on engine %d releases %d phantom(s) of its "
                "node on other engines: %s",
                request.request_id, engine, len(away), away,
            )

    def take_stranded(self) -> list[str]:
        """Phantoms to abort, once each."""
        out, self.stranded = self.stranded, []
        return out

    def produced(self, request_id: str) -> None:
        """A token came back for ``request_id``: its prefill is done."""
        self.prefilling.pop(request_id, None)

    def finished(self, request_id: str) -> None:
        self.prefilling.pop(request_id, None)
        self.phantoms.pop(request_id, None)
        engine = self.call_engine.pop(request_id, None)
        if engine is not None:
            self.calls[engine] -= 1
