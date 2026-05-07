# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.cached_chat.store import (
    ManualKVHandleEntry,
    ManualKVStore,
)
from vllm.entrypoints.openai.cached_chat.worker_ext import (
    EXTRA_KEY_CAPTURE,
    EXTRA_KEY_HANDLE,
    EXTRA_KEY_PREFILLED_TOKENS,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

logger = init_logger(__name__)


class OpenAIServingCachedChat(OpenAIServingChat):
    """Manual KV-cache reuse on top of /v1/chat/completions.

    On miss: stamps ``manual_kv_capture_after_prefill=True`` plus a fresh
    ``manual_kv_handle`` into the request's ``vllm_xargs``. The worker
    then captures the prefix KV blocks into pinned host memory after
    prefill (see ``cached_chat/worker_ext.py``).

    On hit: stamps ``manual_kv_handle`` plus
    ``manual_kv_prefilled_tokens=N`` so the scheduler treats the first N
    tokens as already-computed (skipping prefill) and the worker copies
    the saved blob into the freshly-allocated blocks before forward.

    Wire-compatible with the standard chat endpoint.

    Required server flags:
        --no-enable-prefix-caching
        --worker-extension-cls \
            vllm.entrypoints.openai.cached_chat.worker_ext.ManualKVWorkerExtension
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Force prompt-token details on so cache reuse is visible to the
        # client via usage.prompt_tokens_details.cached_tokens. (Engine
        # prefix cache is expected to be disabled; this surfaces our own
        # accounting via the same field.)
        kwargs["enable_prompt_tokens_details"] = True
        super().__init__(*args, **kwargs)

        self._block_size: int | None = None
        self._block_size_lock = asyncio.Lock()

        self.kv_store = ManualKVStore(
            max_entries=64,
            on_evict=self._on_evict_handle,
        )

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------
    @staticmethod
    def _fingerprint(request: ChatCompletionRequest) -> str:
        payload = {
            "model": request.model,
            "messages": [
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in request.messages
            ],
            "tools": [t.model_dump() for t in (request.tools or [])],
            "tool_choice": (
                request.tool_choice.model_dump()
                if hasattr(request.tool_choice, "model_dump")
                else request.tool_choice
            ),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ------------------------------------------------------------------
    # block_size discovery (cached)
    # ------------------------------------------------------------------
    async def _get_block_size(self) -> int:
        if self._block_size is not None:
            return self._block_size
        async with self._block_size_lock:
            if self._block_size is not None:
                return self._block_size
            try:
                results = await self.engine_client.collective_rpc(
                    "manual_kv_install"
                )
            except Exception as e:
                logger.warning(
                    "[cached_chat] collective_rpc(manual_kv_install) failed: %s "
                    "— is the server running with --worker-extension-cls "
                    "vllm.entrypoints.openai.cached_chat.worker_ext."
                    "ManualKVWorkerExtension ?",
                    e,
                )
                raise
            # collective_rpc returns a list (one per worker rank); TP=1 so
            # one entry, but tolerate more by taking the first.
            if isinstance(results, list) and results:
                bs = int(results[0])
            else:
                bs = int(results)
            self._block_size = bs
            logger.info("[cached_chat] worker block_size = %d", bs)
            return bs

    # ------------------------------------------------------------------
    # Handle eviction notification
    # ------------------------------------------------------------------
    def _on_evict_handle(self, handle: str) -> None:
        # Fire-and-forget; we can't await here.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._drop_handle_on_workers(handle))

    async def _drop_handle_on_workers(self, handle: str) -> None:
        try:
            await self.engine_client.collective_rpc(
                "manual_kv_drop", args=(handle,)
            )
            logger.info("[cached_chat] evicted handle=%s on workers", handle[:12])
        except Exception as e:
            logger.warning(
                "[cached_chat] failed to drop handle=%s on workers: %s",
                handle[:12],
                e,
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        try:
            fp = self._fingerprint(request)
        except Exception:
            fp = None

        entry: ManualKVHandleEntry | None = None
        if fp is not None:
            entry = self.kv_store.get(fp)

        # Make sure block_size is known before we issue anything that
        # might rely on it. This also installs the worker-side state.
        try:
            block_size = await self._get_block_size()
        except Exception:
            # If discovery fails, fall through to a plain chat call so
            # the endpoint never strictly worse than /v1/chat/completions.
            return await super().create_chat_completion(request, raw_request)

        xargs = dict(request.vllm_xargs) if request.vllm_xargs else {}

        if entry is not None and entry.populated and entry.prefix_len >= block_size:
            # HIT: configure inject. The capture flag is intentionally NOT
            # re-set; the blob is already on the worker.
            xargs[EXTRA_KEY_HANDLE] = entry.handle
            xargs[EXTRA_KEY_PREFILLED_TOKENS] = int(entry.prefix_len)
            request.vllm_xargs = xargs
            logger.info(
                "[cached_chat] HIT  fp=%s handle=%s prefix_len=%d hits=%d",
                fp[:12] if fp else "?",
                entry.handle[:12],
                entry.prefix_len,
                entry.hits,
            )
        elif fp is not None:
            # MISS: reserve a handle, ask the worker to capture after prefill.
            new_entry = self.kv_store.reserve(fp)
            xargs[EXTRA_KEY_HANDLE] = new_entry.handle
            xargs[EXTRA_KEY_CAPTURE] = 1
            request.vllm_xargs = xargs
            logger.info(
                "[cached_chat] MISS fp=%s handle=%s — capture-after-prefill armed",
                fp[:12],
                new_entry.handle[:12],
            )

        result = await super().create_chat_completion(request, raw_request)

        # Mark the store entry populated once we've seen prefill complete.
        # We use the response's prompt_tokens to compute the actual prefix
        # length captured (rounded down to a whole-block boundary) so the
        # next request asks for exactly what's available.
        if (
            fp is not None
            and entry is None
            and isinstance(result, ChatCompletionResponse)
        ):
            prompt_tokens = (
                result.usage.prompt_tokens if result.usage is not None else 0
            )
            captured_tokens = (prompt_tokens // block_size) * block_size
            if captured_tokens > 0:
                self.kv_store.populate(fp, captured_tokens)
                logger.info(
                    "[cached_chat] POPULATED fp=%s prefix_len=%d "
                    "(prompt_tokens=%d, block_size=%d)",
                    fp[:12],
                    captured_tokens,
                    prompt_tokens,
                    block_size,
                )
            else:
                # Prompt was shorter than one block — nothing to reuse.
                # Drop the placeholder so we don't leak handles on the
                # worker for a blob that was never written.
                handle = self.kv_store.drop(fp)
                if handle is not None:
                    await self._drop_handle_on_workers(handle)

        return result
