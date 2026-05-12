# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import Field, model_validator

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class ChunkedChatCompletionRequest(ChatCompletionRequest):
    """Request body for `/v1/chunked_chat/completions`.

    Mirrors :class:`ChatCompletionRequest` and adds two fields:

    - ``chunks``: ordered list of input pieces. The serving layer is
      responsible for combining them into the prompt that the engine sees.
    - ``anchor_indices``: indices into ``chunks`` that are "static" and may
      therefore be reused / cached across requests.

    NOTE: shape of ``chunks`` and ``anchor_indices`` is a placeholder — the
    final logic will be filled in later. ``messages`` is made optional so a
    request can be specified purely via ``chunks``.
    """

    # Override: messages becomes optional when chunks are provided.
    messages: list[ChatCompletionMessageParam] = Field(default_factory=list)

    chunks: list[str] = Field(
        ...,
        description=(
            "Ordered list of input chunks. Concatenation / templating "
            "behavior is defined by the serving layer."
        ),
    )

    anchor_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Indices into `chunks` marking chunks that are static "
            "(anchors). Must be a subset of [0, len(chunks))."
        ),
    )

    agent_id: str | None = Field(
        default=None,
        description=(
            "Optional unique id (e.g. agent id) that namespaces the "
            "anchor pools for this request's chunks. Two requests with "
            "the same chunk text but different `agent_id` will not share "
            "pools. Leave unset for a global namespace."
        ),
    )

    @model_validator(mode="after")
    def _validate_chunks_and_anchors(self) -> "ChunkedChatCompletionRequest":
        if not self.chunks and not self.messages:
            raise ValueError(
                "ChunkedChatCompletionRequest requires either `chunks` "
                "or `messages` to be non-empty."
            )
        n = len(self.chunks)
        for idx in self.anchor_indices:
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"anchor_indices entry {idx} is out of range "
                    f"for chunks of length {n}."
                )
        if len(set(self.anchor_indices)) != len(self.anchor_indices):
            raise ValueError("anchor_indices must not contain duplicates.")
        return self


# Reuse the standard chat completion response shape.
ChunkedChatCompletionResponse = ChatCompletionResponse
