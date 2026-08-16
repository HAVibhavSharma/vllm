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

from typing import Any, Literal

from pydantic import BaseModel, Field

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

AgentKind = Literal["react", "non-react"]


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
        """Strip agent-routing fields and return a plain
        ChatCompletionRequest the existing serving_chat handler can
        consume.

        ``agent_id`` is intentionally **kept** so it lands in
        ``model_extra`` on the inner request and gets promoted into
        ``SamplingParams.extra_args`` — that's how the per-request CSV
        in ``FileStatLogger`` ends up with the agent attribution.
        """
        # by_alias=True preserves wire-format keys (e.g. `schema` on
        # JsonSchemaResponseFormat) so the round-trip through
        # model_validate doesn't silently drop aliased fields.
        data = self.model_dump(by_alias=True, exclude_none=True)
        for key in (
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
    agent_kind: AgentKind = Field(
        default="react",
        description="Agent shape. ``react`` (default) keeps the existing "
        "behaviour: multiple prefixes accumulate in the registry and "
        "prefetch warms up to ``prefetch_top_k`` of them. ``non-react`` "
        "assumes a single static prefix per agent: the agent's existing "
        "registry entries are dropped before recording the new seed "
        "(if ``text`` is supplied), and the effective prefetch fan-out "
        "is forced to 1 regardless of ``prefetch_top_k``."
    )
    text: str | None = Field(
        default=None, min_length=1, max_length=1_048_576,
        description="Optional raw prefix text. When provided, the server "
        "wraps it as a system message, applies the served model's chat "
        "template (with ``add_generation_prompt=False`` so the tokens are "
        "a clean prefix of any real chat that starts with the same "
        "system content), tokenizes, and records the chunk-aligned "
        "result in the registry under ``agent_id`` before fanning out "
        "phantom prefetches. Omit to use whatever the registry already "
        "holds for the agent."
    )
    messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional seed as a full chat array, rendered exactly "
        "like ``text`` but without the system-message wrap. Takes "
        "precedence over ``text``.\n\n"
        "``text`` can only ever warm a system-message head, because that is "
        "what it renders to. A caller that already knows the upcoming "
        "request verbatim -- a pinned trace replay, say -- can warm the "
        "whole conversation prefix instead by sending that request's own "
        "``messages`` here. Rendering still forces "
        "``add_generation_prompt=False``, so the tokens stop where the "
        "assistant turn would begin and stay a strict prefix of the real "
        "call. Requests whose prompt carries no system message at all "
        "(single-user-message calls) are unreachable through ``text`` and "
        "are the reason this field exists."
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Template kwargs the upcoming real request will send, "
        "for templates whose output depends on them (Qwen3's "
        "``enable_thinking``, for instance). Merged under the server's own "
        "``add_generation_prompt=False``, which always wins -- the seed has "
        "to stop before the assistant turn to stay a prefix. Ignored when "
        "seeding from ``text``."
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Tool definitions to render alongside ``messages``. "
        "Required for a tool-calling request: chat templates splice the "
        "tool JSON into the system block, so a seed rendered without them "
        "produces different tokens than the real call and warms blocks "
        "nothing will ever hit. Ignored when seeding from ``text``."
    )
    # --- node-eviction identity -------------------------------------
    #
    # `(job_id, langgraph_node, call_type)` is the key the node-aware
    # eviction policy indexes blocks under (`node_key_for_request`). Without
    # it a phantom's blocks are cached but never indexed: `on_blocks_cached`
    # returns early on `key is None`, so they carry no score, are never
    # stamped speculative, and are evicted in plain LRU order — the opposite
    # of what warming them was for.
    #
    # These must be the identity the *upcoming real request* will present,
    # not the prefetch caller's own. `agent_id` is namespace-scoped and its
    # node segment can be a graph path (`langgraph:research_supervisor:
    # supervisor_tools:researcher`) while the real request sends the bare
    # runtime node (`researcher`), so the node is NOT derived from it here —
    # a derived-but-wrong key indexes the phantom under a name no real
    # request ever presents, which is worse than leaving it unscored.
    job_id: str | None = Field(
        default=None, max_length=256,
        description="Job the warmed prefix belongs to. First element of "
        "the node-eviction index key. Omit only if you accept unscored "
        "blocks."
    )
    langgraph_node: str | None = Field(
        default=None, max_length=256,
        description="Runtime node name the warmed prefix will be used by "
        "— the bare value the real request sends as `langgraph_node`, not "
        "the graph path embedded in `agent_id`. Second element of the "
        "index key."
    )
    call_type: str | None = Field(
        default=None, max_length=256,
        description="Flattened leaf label for the call, third element of "
        "the index key. Defaults server-side to the empty string, which is "
        "what `use_call_type=False` keying uses."
    )
