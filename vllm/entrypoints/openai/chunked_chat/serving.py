# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for `/v1/chunked_chat/completions`.

Pipeline:
    1. Render the request via the standard chat template, treating the
       chunks (in order) as the contents of a single user message.
    2. Tokenize the rendered prompt; walk the chunks in order and locate
       each chunk's character offset, then count tokens up to that
       offset to derive per-chunk `(t_start, num_tokens)` spans.
    3. For each chunk listed in `anchor_indices`, attach an entry to
       `vllm_xargs["anchor_pool_spans"]` with that span and a
       content-stable `chunk_hash` (sha256 of the chunk text).
    4. Forward to the standard chat completion path. The worker-side
       anchor-pool extension reads the xargs in its engine-loop hook,
       and after prefill completes runs:
           - capture_placeholder_base   (first observation per chunk)
           - predict_as_anchor + capture_anchor (subsequent observations)
    5. Return the response unchanged.

The chunked_chat endpoint does NOT yet skip prefill on cache hits — see
`vllm/entrypoints/openai/cached_chat/ARCHITECTURE.md` for the gap on
scheduler integration. This serving layer is the orchestration scaffold
that exercises the capture / admission pipeline so the pool actually
grows over a benchmark run.

Required server flags:
    --worker-extension-cls \\
        vllm.entrypoints.openai.cached_chat.anchor_pool_worker_ext.\\
        AnchorPoolWorkerExtension
"""

import asyncio
import hashlib
import os
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.chunked_chat.protocol import (
    ChunkedChatCompletionRequest,
    ChunkedChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

logger = init_logger(__name__)


_XARG_ANCHOR_SPANS = "anchor_pool_spans"
_XARG_ENTROPY_THRESHOLD = "anchor_pool_entropy_threshold"
_XARG_TOP_P = "anchor_pool_top_p"
_XARG_BLEND_ON_NO_ADMIT = "anchor_pool_blend_on_no_admit"
_XARG_BLEND_TEMPERATURE = "anchor_pool_blend_temperature"


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


class OpenAIServingChunkedChat(OpenAIServingChat):
    """Chat completion handler with chunk-aware anchor-pool capture."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Forces the worker extension to attach its state on first hit.
        self._anchor_pool_installed: bool = False
        self._anchor_pool_install_lock = asyncio.Lock()
        self._anchor_pool_threshold: float = _env_float(
            "ANCHOR_POOL_ENTROPY_THRESHOLD", 0.3
        )
        self._anchor_pool_top_p: float = _env_float(
            "ANCHOR_POOL_TOP_P", 0.9
        )
        # When True, the worker extension's admit=False branch blends
        # anchor deltas into the candidate K/V and writes the corrected
        # tensors back into the live KV cache. This is the "Option X"
        # accuracy-validation path — no compute saved, but exercises the
        # reference reuse formula end-to-end so its quality vs. raw
        # dense prefill can be measured. Controlled by env var
        # `ANCHOR_POOL_BLEND` (default on).
        self._anchor_pool_blend_on_no_admit: bool = _env_bool(
            "ANCHOR_POOL_BLEND", True
        )
        self._anchor_pool_blend_temperature: float = _env_float(
            "ANCHOR_POOL_BLEND_TEMPERATURE", 1.0
        )
        logger.info(
            "[chunked_chat] anchor pool config: threshold=%.2f top_p=%.2f "
            "blend=%s temperature=%.2f",
            self._anchor_pool_threshold,
            self._anchor_pool_top_p,
            self._anchor_pool_blend_on_no_admit,
            self._anchor_pool_blend_temperature,
        )

    # ------------------------------------------------------------------
    # Lazy install on first request
    # ------------------------------------------------------------------
    async def _ensure_anchor_pool_installed(self) -> None:
        if self._anchor_pool_installed:
            return
        async with self._anchor_pool_install_lock:
            if self._anchor_pool_installed:
                return
            try:
                results = await self.engine_client.collective_rpc(
                    "anchor_pool_install"
                )
                info = results[0] if isinstance(results, list) else results
                logger.info(
                    "[chunked_chat] anchor pool installed: %s", info
                )
                self._anchor_pool_installed = True
            except Exception as e:
                logger.warning(
                    "[chunked_chat] anchor_pool_install RPC failed: %s "
                    "— continuing without anchor pool",
                    e,
                )

    # ------------------------------------------------------------------
    # Chunk-to-token-span resolution
    # ------------------------------------------------------------------
    def _compute_chunk_spans(
        self,
        chunks: list[str],
        anchor_indices: list[int],
    ) -> list[dict[str, Any]]:
        """Return a list of `{chunk_hash, t_start, num_tokens}` for each
        chunk whose index is in `anchor_indices`.

        Spans are token offsets within the chat-templated rendered
        prompt, derived by tokenizing prefix strings up to each chunk
        boundary. This is the simplest robust algorithm; tokenizers
        without sub-substring stability may produce off-by-1 spans, but
        for templates that put chunks on their own lines (the benchmark
        case) the boundaries are exact.
        """
        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            logger.warning(
                "[chunked_chat] tokenizer unavailable; skipping anchor "
                "span computation"
            )
            return []

        # Render the prompt via the chat template using the concatenated
        # chunks as a single user message. This matches what the engine
        # will actually tokenize after we forward the request.
        joined = "".join(chunks)
        # Build a one-shot conversation that mirrors what the standard
        # chat completion path will see if `messages` is empty.
        messages = [{"role": "user", "content": joined}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(
                "[chunked_chat] apply_chat_template failed: %s — "
                "falling back to raw concatenation for span calc",
                e,
            )
            rendered = joined

        # Locate the joined-chunks substring within the rendered prompt.
        # If the chat template wraps content, the offset shifts uniformly
        # for all chunks; we just need the cumulative char offset of the
        # very first chunk to anchor positions correctly.
        try:
            content_start_char = rendered.index(joined)
        except ValueError:
            # The chat template mangled whitespace such that the literal
            # joined string doesn't appear. Fall back to assuming chunks
            # appear in order at increasing offsets.
            content_start_char = 0
            logger.warning(
                "[chunked_chat] joined chunks not found verbatim in "
                "rendered prompt; spans may be approximate"
            )

        # Walk chunks in order, accumulating char offsets and tokenizing
        # the rendered prefix-up-to-each-boundary.
        spans: list[dict[str, Any]] = []
        cursor = content_start_char
        for idx, chunk in enumerate(chunks):
            chunk_start = cursor
            chunk_end = cursor + len(chunk)
            cursor = chunk_end

            if idx not in anchor_indices:
                continue

            try:
                t_start = len(
                    tokenizer.encode(
                        rendered[:chunk_start], add_special_tokens=False
                    )
                )
                t_end = len(
                    tokenizer.encode(
                        rendered[:chunk_end], add_special_tokens=False
                    )
                )
            except Exception as e:
                logger.warning(
                    "[chunked_chat] tokenizer.encode failed for chunk %d: "
                    "%s — dropping span",
                    idx,
                    e,
                )
                continue
            num_tokens = t_end - t_start
            if num_tokens <= 0:
                logger.warning(
                    "[chunked_chat] chunk %d resolved to zero tokens; "
                    "dropping span",
                    idx,
                )
                continue
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            spans.append(
                {
                    "chunk_hash": chunk_hash,
                    "t_start": t_start,
                    "num_tokens": num_tokens,
                }
            )
            logger.info(
                "[chunked_chat] anchor span chunk_idx=%d hash=%s "
                "t_start=%d num_tokens=%d",
                idx,
                chunk_hash[:12],
                t_start,
                num_tokens,
            )
        return spans

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def create_chunked_chat_completion(
        self,
        request: ChunkedChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> (
        AsyncGenerator[str, None] | ChunkedChatCompletionResponse | ErrorResponse
    ):
        await self._ensure_anchor_pool_installed()

        chunks = list(request.chunks or [])
        anchor_indices = list(request.anchor_indices or [])

        # Optional: move all anchor chunks to the front so the connector's
        # leading-prefix skip covers every static span (otherwise a suffix
        # anchor goes through dense prefill because it isn't contiguous
        # with the prefix anchor). Preserves original order within each
        # group: [anchor_0, anchor_1, ..., dynamic_0, dynamic_1, ...].
        #
        # CAVEAT: this changes the token sequence the model sees. Safe
        # for templates whose suffix is order-independent (style /
        # formatting instructions); breaks templates where the suffix
        # references prior content (e.g. "the conversation above").
        # A/B compare with the env var off to verify quality.
        if (
            chunks
            and anchor_indices
            and _env_bool("ANCHOR_POOL_REORDER_ANCHORS_FRONT", False)
        ):
            anchor_set = set(anchor_indices)
            anchor_chunks = [
                c for i, c in enumerate(chunks) if i in anchor_set
            ]
            dynamic_chunks = [
                c for i, c in enumerate(chunks) if i not in anchor_set
            ]
            chunks = anchor_chunks + dynamic_chunks
            anchor_indices = list(range(len(anchor_chunks)))
            logger.info(
                "[chunked_chat] reordered: %d anchors → front, %d "
                "dynamic chunks → back",
                len(anchor_chunks),
                len(dynamic_chunks),
            )

        # If messages weren't supplied, build a single user message from
        # the concatenated chunks. The standard chat path will tokenize
        # this through the same chat template we just used for span
        # computation, so positions line up.
        if not request.messages and chunks:
            request.messages = [
                {"role": "user", "content": "".join(chunks)}
            ]

        # Compute spans for anchor chunks and stamp them into xargs so
        # the worker extension can act on them after prefill.
        if chunks and anchor_indices and self._anchor_pool_installed:
            try:
                spans = self._compute_chunk_spans(chunks, anchor_indices)
            except Exception as e:
                logger.warning(
                    "[chunked_chat] span computation failed: %s", e
                )
                spans = []
            if spans:
                xargs = dict(request.vllm_xargs) if request.vllm_xargs else {}
                xargs[_XARG_ANCHOR_SPANS] = spans
                xargs[_XARG_ENTROPY_THRESHOLD] = self._anchor_pool_threshold
                xargs[_XARG_TOP_P] = self._anchor_pool_top_p
                xargs[_XARG_BLEND_ON_NO_ADMIT] = (
                    self._anchor_pool_blend_on_no_admit
                )
                xargs[_XARG_BLEND_TEMPERATURE] = (
                    self._anchor_pool_blend_temperature
                )
                request.vllm_xargs = xargs
                logger.info(
                    "[chunked_chat] dispatching with %d anchor spans "
                    "(threshold=%.2f top_p=%.2f blend=%s temp=%.2f)",
                    len(spans),
                    self._anchor_pool_threshold,
                    self._anchor_pool_top_p,
                    self._anchor_pool_blend_on_no_admit,
                    self._anchor_pool_blend_temperature,
                )

        return await self.create_chat_completion(request, raw_request)
