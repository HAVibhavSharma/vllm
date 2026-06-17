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

    ``agent_probabilities`` is the only knob that drives the
    cross-request, agent-aware eviction policy implemented in
    :mod:`vllm.v1.agent_prefetch.eviction`. Callers supply the
    probability that each named agent will fire in the next few turns;
    the engine aggregates the max across every live request and, when
    the GPU block pool runs short, drains blocks from the
    lowest-probability agent first.
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    agent_cache_salt: str | None = Field(default=None, max_length=256)
    record_in_registry: bool = Field(default=True)

    agent_probabilities: dict[str, float] | None = Field(
        default=None,
        description="Optional map of agent_id -> P(agent fires soon). "
        "Used by the agent-aware eviction policy: when the GPU block "
        "pool runs short, blocks are drained starting from the agent "
        "with the lowest aggregated probability across all live "
        "requests. Omit (or set to null) to opt this request out -- "
        "the policy will then aggregate other live requests' votes "
        "without contribution from this one.",
    )

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        """Strip agent fields and return a plain ChatCompletionRequest
        the existing serving_chat handler can consume.

        ``agent_id`` and ``agent_probabilities`` are forwarded into
        ``kv_transfer_params`` so the engine ``Request`` can read them
        on the other side of the API/engine boundary."""
        data = self.model_dump()
        agent_id = data.pop("agent_id", None)
        agent_probabilities = data.pop("agent_probabilities", None)
        data.pop("agent_cache_salt", None)
        data.pop("record_in_registry", None)

        kv_params = dict(data.get("kv_transfer_params") or {})
        if agent_id is not None:
            kv_params["agent_id"] = agent_id
        if agent_probabilities:
            kv_params["agent_probabilities"] = dict(agent_probabilities)
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
    wait: bool = Field(
        default=True,
        description="If True (default) the endpoint awaits all phantom "
        "tasks before responding -- so APC is guaranteed warm on "
        "return. Set to False for fire-and-forget."
    )
    agent_probabilities: dict[str, float] | None = Field(
        default=None,
        description="Optional vote that every phantom request issued by "
        "this call carries into the engine. Forwarded as "
        "`agent_probabilities` in the phantom's `kv_transfer_params`, "
        "where it both (a) flips the eviction policy's 'opted out' "
        "gate so the phantom's blocks are tagged with `agent_id`, and "
        "(b) contributes to the cross-request probability aggregator "
        "for as long as the phantom is in flight. Omit (or leave null) "
        "to fall back to a self-vote of `{agent_id: 1.0}` -- the "
        "minimum needed for the phantom to register at all."
    )
