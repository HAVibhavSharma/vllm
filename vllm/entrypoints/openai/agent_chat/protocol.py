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
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    agent_cache_salt: str | None = Field(default=None, max_length=256)
    record_in_registry: bool = Field(default=True)

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        """Strip agent fields and return a plain ChatCompletionRequest
        the existing serving_chat handler can consume."""
        data = self.model_dump()
        for key in (
            "agent_id",
            "agent_cache_salt",
            "record_in_registry",
        ):
            data.pop(key, None)
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
