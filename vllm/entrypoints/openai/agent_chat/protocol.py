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

from pydantic import BaseModel, Field, model_validator

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
        "wraps it as a message of role ``text_role``, applies the served "
        "model's chat template (with ``add_generation_prompt=False`` so the "
        "tokens are a clean prefix of any real chat that starts with the same "
        "first message), tokenizes, and records the chunk-aligned "
        "result in the registry under ``agent_id`` before fanning out "
        "phantom prefetches. Omit to use whatever the registry already "
        "holds for the agent."
    )
    messages: list[dict[str, Any]] | None = Field(
        default=None, max_length=4096,
        description="Optional multi-turn prefix, as an OpenAI-shaped message "
        "list. Rendered through the served model's chat template with "
        "``add_generation_prompt=False``, so the tokens are a strict prefix "
        "of any real chat whose leading messages are the same list.\n\n"
        "This is what ``text`` cannot express: ``text`` wraps its content as "
        "one message of one role, so a conversation pushed through it renders "
        "as a single block and is a prefix of nothing. A caller that knows the "
        "*conversation* the next request will send -- an agent loop that has "
        "just appended its own reply to the turn it sent -- passes it here.\n\n"
        "Mutually exclusive with ``text``. ``text_role`` does not apply: each "
        "message carries its own role."
    )
    text_role: Literal["system", "user"] = Field(
        default="system",
        description="Role to wrap ``text`` in before rendering. The chat "
        "template emits different control tokens per role, so a seed "
        "rendered as ``system`` is not a prefix of a real request whose "
        "first message is a ``user`` one — it warms blocks nothing hits. "
        "Set this to the role the upcoming real request actually sends: "
        "``user`` for a single-prompt call such as a tool-internal "
        "summarization, ``system`` (the default) for an agent whose prompt "
        "opens with a system block."
    )
    prefill_on_miss: bool = Field(
        default=True,
        description="Let this call's phantoms run a real prefill when "
        "LMCache has nothing for the prefix, instead of being aborted.\n\n"
        "Default True. The scheduler bounds what such a phantom can take: it "
        "is admitted only into a step running at most "
        "`VLLM_PREFETCH_PREFILL_MAX_RUNNING` real requests (2 by default), "
        "held in the skipped-waiting queue otherwise, and dropped without "
        "prefilling if no such step arrives before "
        "`VLLM_PREFETCH_PREFILL_DEFER_TIMEOUT_S`. At 1 a mispredicted warm "
        "costs a decoding request one longer step and nothing else; the "
        "default of 2 also lets it join a lightly loaded batch, where it can "
        "take budget a real prefill wanted, so the cost there is another "
        "request's TTFT. Set that variable to 0 for the stricter "
        "contract where a phantom waits for a fully idle step -- which also "
        "means it will not prefill until the previous request has finished, "
        "usually too late to help the next one. Set "
        "False to restore the older contract, where a phantom is strictly a "
        "promotion from LMCache L1 into HBM and a prefix the cache does not "
        "hold is finished without ever reaching the model.\n\n"
        "Set this to True for a **seeding** call -- one that has just "
        "recorded a prefix via `text` that no request has ever computed, so "
        "it is guaranteed absent from LMCache. The phantom then prefills "
        "once, and the store path writes the result into LMCache as a side "
        "effect, so every later prefetch for that prefix is a real L1->HBM "
        "promotion. Without it, seeding fills the registry but leaves the "
        "cache empty and each prefix is still prefilled by the first real "
        "request that wants it.\n\n"
        "Costs one prefill per warmed prefix, and that cost is real whether "
        "or not the guess was right: at the default cap it lengthens a step "
        "a decoding request was in, and at 0 it spends an idle step a later "
        "real request could have had to itself. Either way it materialises "
        "blocks that push others out of HBM.\n\n"
        "No effect when LMCache *does* hold the prefix: that path never "
        "reaches the abort, and the phantom terminates before prefill as "
        "usual."
    )
    @model_validator(mode="after")
    def _one_seed_source(self) -> "AgentPrefetchRequest":
        """``text`` and ``messages`` are two spellings of the same slot.

        Accepting both would leave the precedence up to handler order, and a
        caller that sent the wrong one would get a silently-warmed prefix that
        no request hits. Refusing is the only outcome that surfaces it.
        """
        if self.text is not None and self.messages is not None:
            raise ValueError(
                "text and messages are mutually exclusive; send one seed"
            )
        if self.messages is not None and not self.messages:
            raise ValueError("messages must be non-empty when provided")
        return self

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
