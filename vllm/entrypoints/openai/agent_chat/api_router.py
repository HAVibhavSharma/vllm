# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP routes for the agent-scoped chat completion + prefetch endpoints.

This module exposes three POST routes plus one GET diagnostic:

* ``POST /v1/agents/chat/completions`` -- a thin wrapper around the
  existing ``OpenAIServingChat`` pipeline that:
  1. Tokenizes the incoming request to obtain ``prompt_token_ids``.
  2. Delegates the real chat completion to ``OpenAIServingChat``.
  3. Records the new prompt's chunk-aligned prefix in the per-agent
     registry so a *future* ``/v1/agents/prefetch`` call can warm it.
  This endpoint **does not fire phantom prefetches**.

* ``POST /v1/agents/prefetch`` -- the new explicit cache-warming
  endpoint. Body contains ``{agent_id, prefetch_top_k?, wait?}``. Fans
  out phantom prefetches for the agent's most recent prefixes and, by
  default, blocks until all phantoms finish so APC is guaranteed warm
  on return.

* ``POST /v1/agents/reset_prefix_cache`` -- reset APC/registry/
  connector. Useful for benchmarking.

* ``GET /v1/agents/registry_stats`` -- inspect per-agent registry
  sizes for diagnostics.

The registry + submitter are lazily attached to ``app.state`` on the
first request that needs them; no startup-hook surgery is required.
"""

from __future__ import annotations

import asyncio
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.agent_chat.protocol import (
    AgentChatCompletionRequest,
    AgentPrefetchRequest,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.logger import init_logger
from vllm.v1.agent_prefetch import (
    DEFAULT_CHUNK_SIZE,
    AgentPrefixRegistry,
    PhantomPrefetchSubmitter,
    PrefixDescriptor,
    chunk_align,
    compute_prefix_hash,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.serving import (
        OpenAIServingChat,
    )

logger = init_logger(__name__)

router = APIRouter()


# Application-state attribute names used to attach the singletons.
_REGISTRY_ATTR = "agent_prefetch_registry"
_SUBMITTER_ATTR = "agent_prefetch_submitter"


def _get_or_init_state(
    raw_request: Request,
    chat_handler: "OpenAIServingChat",
) -> tuple[AgentPrefixRegistry, PhantomPrefetchSubmitter]:
    """Lazy-init the registry + submitter on the app state."""
    state = raw_request.app.state
    registry: AgentPrefixRegistry | None = getattr(state, _REGISTRY_ATTR, None)
    submitter: PhantomPrefetchSubmitter | None = getattr(
        state, _SUBMITTER_ATTR, None
    )
    if registry is None:
        registry = AgentPrefixRegistry(
            default_top_k=20,
            max_agents=10_000,
            max_per_agent=None,  # no per-agent cap
        )
        setattr(state, _REGISTRY_ATTR, registry)
        logger.info(
            "agent_prefetch: initialized registry "
            "(default_top_k=20, max_agents=10000, max_per_agent=unlimited)"
        )
    if submitter is None:
        submitter = PhantomPrefetchSubmitter(
            engine_client=chat_handler.engine_client,
            max_inflight_per_agent=64,
        )
        setattr(state, _SUBMITTER_ATTR, submitter)
        logger.info("agent_prefetch: initialized phantom submitter")
    return registry, submitter


def _chat_handler(raw_request: Request) -> "OpenAIServingChat | None":
    return raw_request.app.state.openai_serving_chat


async def _tokenize_prompt(
    chat_handler: "OpenAIServingChat",
    request: AgentChatCompletionRequest,
) -> list[int] | ErrorResponse:
    """Apply the chat template + tokenize, returning the prompt token
    ids that the engine would consume for this request."""
    inner = request.to_chat_completion_request()
    rendered = await chat_handler.render_chat_request(inner)
    if isinstance(rendered, ErrorResponse):
        return rendered
    _conversation, engine_inputs = rendered
    if not engine_inputs:
        return ErrorResponse(
            type="BadRequest",
            message="agent chat: render produced no engine inputs",
            code=HTTPStatus.BAD_REQUEST.value,
        )
    components = chat_handler._extract_prompt_components(engine_inputs[0])
    return list(components.token_ids or [])


async def _fan_out_prefetches(
    *,
    registry: AgentPrefixRegistry,
    submitter: PhantomPrefetchSubmitter,
    agent_id: str,
    k: int | None,
    wait: bool,
) -> tuple[int, int]:
    """Submit phantom prefetches for ``agent_id``.

    When ``k`` is None, every prefix the registry has stored for the
    agent is warmed (no top-K truncation). When ``k`` is a positive
    int, only the ``k`` most-recently-used prefixes are warmed; ``k``
    <= 0 is a no-op.

    Returns ``(submitted, completed)`` -- ``submitted`` is the number
    of phantoms actually handed to the engine after dedup; ``completed``
    is how many ran to completion (equal to ``submitted`` when
    ``wait=True``, else 0 because we did not block).
    """
    if k is None:
        descriptors = registry.get_all(agent_id)
    elif k <= 0:
        return 0, 0
    else:
        descriptors = registry.top_k(agent_id, k=k)

    tasks: list[asyncio.Task] = []
    for desc in descriptors:
        task = await submitter.submit(
            agent_id=agent_id,
            token_ids=desc.token_ids,
            prefix_hash=desc.prefix_hash,
            cache_salt=desc.cache_salt,
        )
        if task is not None:
            tasks.append(task)

    submitted = len(tasks)
    if not wait or not tasks:
        return submitted, 0

    # `_run_one` absorbs all exceptions except CancelledError, so
    # gather will normally complete cleanly. `return_exceptions=True`
    # is belt-and-braces in case a future refactor changes that.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    completed = sum(1 for r in results if not isinstance(r, BaseException))
    return submitted, completed


def _record_in_registry(
    *,
    registry: AgentPrefixRegistry,
    agent_id: str,
    cache_salt: str,
    model_name: str,
    prompt_token_ids: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    aligned = chunk_align(prompt_token_ids, chunk_size=chunk_size)
    if not aligned:
        return False
    prefix_hash = compute_prefix_hash(model_name, cache_salt, aligned)
    desc = PrefixDescriptor(
        token_ids=tuple(aligned),
        prefix_hash=prefix_hash,
        cache_salt=cache_salt,
        last_used_ns=time.monotonic_ns(),
    )
    registry.record(agent_id, desc)
    return True


def _resolve_chat_cache_salt(req: AgentChatCompletionRequest) -> str:
    return req.agent_cache_salt or f"agent::{req.agent_id}"


def _resolve_prefetch_cache_salt(req: AgentPrefetchRequest) -> str:
    return req.agent_cache_salt or f"agent::{req.agent_id}"


@router.post(
    "/v1/agents/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_agent_chat_completion(
    request: AgentChatCompletionRequest,
    raw_request: Request,
):
    """Agent-scoped chat completion.

    Delegates to the existing chat completion handler and, on success,
    records the prompt's chunk-aligned prefix in the per-agent
    registry so a subsequent ``POST /v1/agents/prefetch`` can warm it.

    **Does not fire phantom prefetches.** Call
    ``POST /v1/agents/prefetch`` explicitly if you want APC warmed
    before this call.
    """
    chat_handler = _chat_handler(raw_request)
    if chat_handler is None:
        return JSONResponse(
            content=ErrorResponse(
                type="NotImplemented",
                message="The model does not support Chat Completions API",
                code=HTTPStatus.NOT_IMPLEMENTED.value,
            ).model_dump(),
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        )

    registry, _submitter = _get_or_init_state(raw_request, chat_handler)

    cache_salt = _resolve_chat_cache_salt(request)

    # 1) Tokenize the incoming request so we can record its prefix
    # after delegation. Required even if record_in_registry is False
    # would be wasteful -- so skip tokenization in that case.
    prompt_token_ids: list[int] = []
    if request.record_in_registry:
        tokens_or_err = await _tokenize_prompt(chat_handler, request)
        if isinstance(tokens_or_err, ErrorResponse):
            return JSONResponse(
                content=tokens_or_err.model_dump(),
                status_code=tokens_or_err.error.code,
            )
        prompt_token_ids = tokens_or_err

    # 2) Delegate the real call to the existing chat completion handler.
    inner = request.to_chat_completion_request()
    generator = await chat_handler.create_chat_completion(inner, raw_request)

    # 3) Record this prompt's chunk-aligned prefix for future prefetches.
    # Done before returning so it's available to the next call regardless
    # of whether the response was fully streamed.
    if request.record_in_registry and prompt_token_ids:
        try:
            recorded = _record_in_registry(
                registry=registry,
                agent_id=request.agent_id,
                cache_salt=cache_salt,
                model_name=chat_handler.model_config.model,
                prompt_token_ids=prompt_token_ids,
            )
            if recorded:
                logger.debug(
                    "agent_prefetch: recorded prefix for agent %s "
                    "(%d tokens after chunk alignment)",
                    request.agent_id,
                    (len(prompt_token_ids) // DEFAULT_CHUNK_SIZE)
                    * DEFAULT_CHUNK_SIZE,
                )
        except Exception:
            # Recording is best-effort. Never break the real response.
            logger.exception(
                "agent_prefetch: failed to record prefix for agent %s",
                request.agent_id,
            )

    # 4) Return whatever the chat handler produced.
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    if isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())
    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/agents/prefetch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": None},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
async def prefetch_agent_cache(
    request: AgentPrefetchRequest,
    raw_request: Request,
):
    """Warm APC for ``agent_id`` by submitting phantom prefetches.

    Pulls the agent's most recent ``prefetch_top_k`` prefixes from the
    registry and submits a phantom request per prefix. Each phantom
    drives the LMCache -> GPU load that registers the prefix in APC.

    When ``wait`` is True (default) the response is held until every
    phantom finishes, so the caller can immediately follow up with a
    chat completion and expect APC hits. When ``wait`` is False the
    endpoint returns as soon as phantoms are submitted.

    Response body::

      {
        "agent_id": "...",
        "requested_top_k": 20,
        "available_prefixes": 3,
        "submitted": 3,
        "completed": 3,
        "waited": true,
        "duration_ms": 142.5
      }
    """
    started_ns = time.monotonic_ns()

    chat_handler = _chat_handler(raw_request)
    if chat_handler is None:
        return JSONResponse(
            content=ErrorResponse(
                type="NotImplemented",
                message="The model does not support Chat Completions API; "
                "agent prefetch is unavailable.",
                code=HTTPStatus.NOT_IMPLEMENTED.value,
            ).model_dump(),
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        )

    registry, submitter = _get_or_init_state(raw_request, chat_handler)

    # None ≡ "warm every prefix the registry has for this agent"; the
    # request validator enforces ge=0 so non-None values are truncating
    # caps (mostly useful for benchmarking).
    top_k = request.prefetch_top_k
    available = registry.agent_size(request.agent_id)
    top_k_repr = "all" if top_k is None else str(top_k)

    logger.info(
        "agent_prefetch: prefetch request for agent=%s top_k=%s "
        "available=%d wait=%s",
        request.agent_id,
        top_k_repr,
        available,
        request.wait,
    )

    submitted, completed = await _fan_out_prefetches(
        registry=registry,
        submitter=submitter,
        agent_id=request.agent_id,
        k=top_k,
        wait=request.wait,
    )

    elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
    body: dict[str, Any] = {
        "agent_id": request.agent_id,
        # "all" when no cap was provided; an int otherwise.
        "requested_top_k": top_k_repr,
        "available_prefixes": available,
        "submitted": submitted,
        "completed": completed,
        "waited": request.wait,
        "duration_ms": round(elapsed_ms, 2),
    }
    logger.info(
        "agent_prefetch: prefetch done agent=%s submitted=%d completed=%d "
        "elapsed=%.2fms",
        request.agent_id,
        submitted,
        completed,
        elapsed_ms,
    )
    return JSONResponse(content=body)


@router.post("/v1/agents/reset_prefix_cache")
async def reset_agent_prefix_cache(
    raw_request: Request,
    reset_apc: bool = Query(
        default=True,
        description="Reset vLLM's GPU paged-KV prefix cache.",
    ),
    reset_registry: bool = Query(
        default=False,
        description="Drop the in-process agent prefix registry. "
        "Forces phantom prefetches to wait until the registry is "
        "re-populated by subsequent calls.",
    ),
    reset_connector: bool = Query(
        default=False,
        description="Also reset the KV connector cache (LMCache CPU L1 "
        "via the connector). Use with care -- this throws away data "
        "the server holds in CPU memory.",
    ),
):
    """Reset prefix-cache layers used by the agent prefetch path.

    Useful for benchmarking: clear GPU APC so the next call goes cold,
    but keep the agent registry and LMCache CPU pool so the prefetch
    endpoint can demonstrate CPU->GPU warming.
    """
    started_ns = time.monotonic_ns()
    actions: list[str] = []
    result: dict[str, Any] = {
        "requested": {
            "reset_apc": reset_apc,
            "reset_registry": reset_registry,
            "reset_connector": reset_connector,
        },
        "actions": actions,
        "apc": None,
        "registry": None,
        "duration_ms": None,
    }

    chat_handler = _chat_handler(raw_request)

    # --- APC reset ---
    if reset_apc:
        if chat_handler is None:
            logger.warning(
                "agent_prefetch: reset_prefix_cache called but no chat "
                "handler is available; cannot reset APC"
            )
            result["apc"] = {
                "ok": False,
                "reason": "engine_client_unavailable",
            }
            return JSONResponse(content=result, status_code=503)

        logger.info(
            "agent_prefetch: resetting vLLM APC "
            "(reset_connector=%s)",
            reset_connector,
        )
        apc_returned = await chat_handler.engine_client.reset_prefix_cache(
            reset_running_requests=False,
            reset_connector=reset_connector,
        )
        actions.append("apc_reset")
        result["apc"] = {
            "ok": bool(apc_returned),
            "engine_returned": apc_returned,
            "connector_reset": reset_connector,
        }
        logger.info(
            "agent_prefetch: APC reset complete (engine returned %s)",
            apc_returned,
        )
        if reset_connector:
            actions.append("connector_reset")

    # --- Registry reset ---
    registry: AgentPrefixRegistry | None = getattr(
        raw_request.app.state, _REGISTRY_ATTR, None
    )
    registry_before = registry.stats() if registry is not None else None

    if reset_registry:
        if registry is None:
            logger.info(
                "agent_prefetch: reset_registry requested but registry "
                "has not been initialized yet -- no-op"
            )
        else:
            registry.clear()
            actions.append("registry_cleared")
            logger.info(
                "agent_prefetch: registry cleared (was: %s)",
                registry_before,
            )

    result["registry"] = {
        "before": registry_before,
        "after": registry.stats() if registry is not None else None,
    }

    elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
    result["duration_ms"] = round(elapsed_ms, 2)

    logger.info(
        "agent_prefetch: reset done in %.2fms actions=%s",
        elapsed_ms,
        actions,
    )
    return JSONResponse(content=result)


@router.get("/v1/agents/registry_stats")
async def get_agent_registry_stats(raw_request: Request):
    """Inspect the in-process registry. Returns counts per agent so you
    can verify phantoms have history to prefetch from before kicking
    off a benchmark."""
    registry = getattr(raw_request.app.state, _REGISTRY_ATTR, None)
    if registry is None:
        return JSONResponse(
            content={"initialized": False, "agents": {}}
        )
    overall = registry.stats()
    # Per-agent sizes via the public size accessor.
    # We intentionally don't expose token_ids -- those can be large.
    agent_sizes = {}
    for agent_id in list(registry._by_agent.keys()):  # type: ignore[attr-defined]
        agent_sizes[agent_id] = registry.agent_size(agent_id)
    return JSONResponse(
        content={
            "initialized": True,
            "overall": overall,
            "per_agent_sizes": agent_sizes,
        }
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
