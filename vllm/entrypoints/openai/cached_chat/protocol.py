# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)

# The cached_chat endpoint takes exactly the same body as /v1/chat/completions
# and returns exactly the same response shape. Aliased here so the route can
# be typed independently if we ever want to diverge.
CachedChatCompletionRequest = ChatCompletionRequest
CachedChatCompletionResponse = ChatCompletionResponse
CachedChatCompletionStreamResponse = ChatCompletionStreamResponse
