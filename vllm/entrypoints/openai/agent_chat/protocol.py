# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request schema for the agent-scoped chat completion endpoint.

Strict superset of the OpenAI ChatCompletionRequest: same fields plus

* ``agent_id``        -- caller identifier used to scope the prefix
                         registry. Required.
* ``prefetch_top_k``  -- how many of this agent's most recent prefixes
                         to phantom-prefetch before the real call.
                         Defaults to the server-side registry default.
* ``agent_cache_salt``-- optional override for the LMCache cache_salt
                         attached to phantom prefetches. Defaults to
                         ``"agent::<agent_id>"``.
* ``record_in_registry`` -- whether to record this call's prefix into
                         the registry for future prefetches. Defaults
                         to True.
"""

from __future__ import annotations

from pydantic import Field

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)


class AgentChatCompletionRequest(ChatCompletionRequest):
    agent_id: str = Field(..., min_length=1, max_length=128)
    prefetch_top_k: int | None = Field(default=None, ge=0, le=64)
    agent_cache_salt: str | None = Field(default=None, max_length=256)
    record_in_registry: bool = Field(default=True)

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        """Strip agent fields and return a plain ChatCompletionRequest
        the existing serving_chat handler can consume."""
        data = self.model_dump()
        for key in (
            "agent_id",
            "prefetch_top_k",
            "agent_cache_salt",
            "record_in_registry",
        ):
            data.pop(key, None)
        return ChatCompletionRequest.model_validate(data)
