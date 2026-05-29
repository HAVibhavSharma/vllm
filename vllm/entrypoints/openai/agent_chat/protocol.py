# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request schemas for the agent-scoped chat completion + prefetch endpoints.

* :class:`AgentChatCompletionRequest` is a strict superset of the
  OpenAI ``ChatCompletionRequest`` with extra fields that scope the
  call to a named agent and control whether its prompt gets recorded
  in the per-agent prefix registry. **It does not fire phantom
  prefetches** -- that responsibility now lives in the separate
  :class:`AgentPrefetchRequest` endpoint.

* :class:`AgentPrefetchRequest` is the body for the new
  ``POST /v1/agents/prefetch`` endpoint. The caller supplies an
  ``agent_id`` and the server fans out phantom prefetches for that
  agent's most recent prefixes so APC is warm before the next real
  chat call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)


class AgentChatCompletionRequest(ChatCompletionRequest):
    """Agent-scoped chat completion.

    Same body as ``ChatCompletionRequest`` plus an ``agent_id`` so the
    server can record the prompt's chunk-aligned prefix in the
    per-agent registry. Phantom prefetches are *not* triggered here --
    call ``POST /v1/agents/prefetch`` first if you want APC warmed.

    Three optional fields drive the cross-request, agent-aware **early
    eviction** policy implemented in :mod:`vllm.v1.agent_prefetch.eviction`:

    * ``agent_probabilities`` -- caller-supplied probabilities that each
      named agent will be the next-N-turns originator. The engine
      aggregates these across every live request and lets the GPU block
      pool evict any cached block whose owning agent's aggregated
      probability is below ``eviction_threshold``.
    * ``eviction_window`` -- the ``N`` the probabilities were computed
      over. Carried for diagnostics; the policy itself does not derive
      behavior from it currently.
    * ``eviction_threshold`` -- minimum aggregated probability an agent
      needs to keep its blocks safe. Below this, its blocks become
      first-class eviction candidates ahead of LRU.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    agent_cache_salt: str | None = Field(default=None, max_length=256)
    record_in_registry: bool = Field(default=True)

    agent_probabilities: dict[str, float] | None = Field(
        default=None,
        description="Optional map of agent_id -> P(agent fires in next "
        "`eviction_window` turns). Used by the agent-aware early "
        "eviction policy: blocks owned by agents whose aggregated "
        "probability falls below `eviction_threshold` are evicted "
        "ahead of LRU when `get_new_blocks` runs short on free "
        "capacity. Omit (or set to null) to opt this request out -- "
        "the policy will then aggregate other live requests' votes "
        "without contribution from this one.",
    )
    eviction_window: int | None = Field(
        default=None, ge=1, le=1024,
        description="Window (number of upcoming turns) the probabilities "
        "in `agent_probabilities` were computed over. Defaults to 3.",
    )
    eviction_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Probability cutoff below which an agent's blocks "
        "are liable for early eviction. Defaults to 0.5.",
    )
    probability_ttl_seconds: float | None = Field(
        default=None, ge=0.0,
        description="TTL (seconds) on this request's probability vote. "
        "After it elapses, the vote stops contributing to the global "
        "max-aggregation and the entry is opportunistically removed. "
        "Block-ownership tags are unaffected; only the vote ages out. "
        "Defaults to 60s; pass 0 (or omit) to fall back to the "
        "server default. Cap is intentionally loose so long-running "
        "sessions can opt out by setting it large.",
    )

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        """Strip agent fields and return a plain ChatCompletionRequest
        the existing serving_chat handler can consume.

        Agent-aware eviction fields are forwarded into
        ``kv_transfer_params`` so the engine ``Request`` can read them
        on the other side of the API/engine boundary."""
        data = self.model_dump()
        agent_id = data.pop("agent_id", None)
        agent_probabilities = data.pop("agent_probabilities", None)
        eviction_window = data.pop("eviction_window", None)
        eviction_threshold = data.pop("eviction_threshold", None)
        probability_ttl_seconds = data.pop("probability_ttl_seconds", None)
        data.pop("agent_cache_salt", None)
        data.pop("record_in_registry", None)

        kv_params = dict(data.get("kv_transfer_params") or {})
        if agent_id is not None:
            kv_params["agent_id"] = agent_id
        if agent_probabilities:
            kv_params["agent_probabilities"] = dict(agent_probabilities)
        if eviction_window is not None:
            kv_params["eviction_window"] = int(eviction_window)
        if eviction_threshold is not None:
            kv_params["eviction_threshold"] = float(eviction_threshold)
        if probability_ttl_seconds is not None:
            kv_params["probability_ttl_seconds"] = float(probability_ttl_seconds)
        if kv_params:
            data["kv_transfer_params"] = kv_params

        return ChatCompletionRequest.model_validate(data)


class AgentPrefetchRequest(BaseModel):
    """Body for ``POST /v1/agents/prefetch``.

    Triggers phantom-prefetch requests for the agent's most recently
    recorded prefixes. The phantoms drive LMCache -> GPU loads so the
    next real chat completion finds the relevant blocks already in
    APC.
    """

    agent_id: str = Field(
        ..., min_length=1, max_length=128,
        description="Caller identifier; scopes the prefix registry."
    )
    prefetch_top_k: int | None = Field(
        default=None, ge=0,
        description="Cap on how many of this agent's most recent "
        "prefixes to warm. **Omit (or leave null) to warm every prefix "
        "the registry has stored for this agent** -- there is no "
        "implicit top-K limit any more. Pass an explicit integer only "
        "when you want to artificially truncate the warm set."
    )
    agent_cache_salt: str | None = Field(
        default=None, max_length=256,
        description="Optional override for the LMCache cache_salt "
        "attached to phantom prefetches. Defaults to "
        "``\"agent::<agent_id>\"``."
    )
    wait: bool = Field(
        default=True,
        description="If True (default) the endpoint awaits all phantom "
        "tasks before responding -- so APC is guaranteed warm on "
        "return. Set to False for fire-and-forget."
    )
