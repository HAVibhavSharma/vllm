# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncGenerator

from fastapi import Request

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.chunked_chat.protocol import (
    ChunkedChatCompletionRequest,
    ChunkedChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

logger = init_logger(__name__)


class OpenAIServingChunkedChat(OpenAIServingChat):
    """Serving class for `/v1/chunked_chat/completions`.

    Inherits all of :class:`OpenAIServingChat`'s infrastructure (renderer,
    tokenizer, tool/reasoning parsers, etc.). The only new entry point is
    :meth:`create_chunked_chat_completion`, which currently delegates to a
    placeholder pipeline. The real chunk/anchor handling will be added later.
    """

    async def create_chunked_chat_completion(
        self,
        request: ChunkedChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> (
        AsyncGenerator[str, None] | ChunkedChatCompletionResponse | ErrorResponse
    ):
        # ------------------------------------------------------------------
        # PLACEHOLDER: chunk + anchor handling goes here.
        #
        # Expected pipeline (to be implemented):
        #   1. Validate `request.chunks` against `request.anchor_indices`.
        #   2. Build the prompt / engine input from the chunks, applying
        #      whatever anchor-aware logic we land on (e.g. prefix-cache
        #      reuse for static chunks, recomputation for dynamic ones).
        #   3. Call into the existing chat-completion generation path with
        #      the assembled input.
        # ------------------------------------------------------------------
        logger.debug(
            "ChunkedChat request: %d chunks, %d anchors",
            len(request.chunks),
            len(request.anchor_indices),
        )

        # TODO(chunked_chat): replace with chunk-aware prompt assembly.
        # For now we just forward the underlying ChatCompletionRequest fields
        # to the standard chat path so the endpoint is wired end-to-end.
        return await self.create_chat_completion(request, raw_request)
