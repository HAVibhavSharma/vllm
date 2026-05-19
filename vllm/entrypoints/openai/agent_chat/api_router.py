# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP route for the agent-scoped chat completion endpoint.

Wraps the existing ``OpenAIServingChat`` pipeline with a thin layer
that:

1. Tokenizes the incoming request to obtain ``prompt_token_ids``.
2. Looks up the per-agent prefix registry and fires phantom
   prefetches for the agent's most recent N prefixes (best-effort,
   non-blocking).
3. Delegates the real chat completion to ``OpenAIServingChat``.
4. Records the new prompt's chunk-aligned prefix in the registry so
   future calls from the same agent can pre-warm it.

The registry + submitter are lazily attached to ``app.state`` on the
first request; no startup-hook surgery is required.
"""

from __future__ import annotations

import time
from http import HTTPStatus
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.agent_chat.protocol import (
    AgentChatCompletionRequest,
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
            max_per_agent=64,
        )
        setattr(state, _REGISTRY_ATTR, registry)
        logger.info(
            "agent_prefetch: initialized registry "
            "(default_top_k=20, max_agents=10000, max_per_agent=64)"
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
    k: int,
) -> int:
    """Submit up to ``k`` phantom prefetches for ``agent_id``. Returns
    the number actually submitted (after dedup)."""
    if k <= 0:
        return 0
    descriptors = registry.top_k(agent_id, k=k)
    submitted = 0
    for desc in descriptors:
        ok = await submitter.submit(
            agent_id=agent_id,
            token_ids=desc.token_ids,
            prefix_hash=desc.prefix_hash,
            cache_salt=desc.cache_salt,
        )
        if ok:
            submitted += 1
    return submitted


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


def _resolve_cache_salt(req: AgentChatCompletionRequest) -> str:
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

    registry, submitter = _get_or_init_state(raw_request, chat_handler)

    cache_salt = _resolve_cache_salt(request)
    top_k = (
        request.prefetch_top_k
        if request.prefetch_top_k is not None
        else registry.default_top_k
    )

    # 1) Tokenize the incoming request. We need prompt_token_ids both
    # for the post-call registry record and (potentially) for future
    # phantom prefetches.
    tokens_or_err = await _tokenize_prompt(chat_handler, request)
    if isinstance(tokens_or_err, ErrorResponse):
        return JSONResponse(
            content=tokens_or_err.model_dump(),
            status_code=tokens_or_err.error.code,
        )
    prompt_token_ids: list[int] = tokens_or_err

    # 2) Fire phantom prefetches for this agent's recent prefixes.
    # Fire-and-forget: do not await completion.
    submitted = await _fan_out_prefetches(
        registry=registry,
        submitter=submitter,
        agent_id=request.agent_id,
        k=top_k,
    )
    if submitted:
        logger.debug(
            "agent_prefetch: submitted %d phantom prefetches for agent %s",
            submitted,
            request.agent_id,
        )

    # 3) Delegate the real call to the existing chat completion handler.
    inner = request.to_chat_completion_request()
    generator = await chat_handler.create_chat_completion(inner, raw_request)

    # 4) Record this prompt's chunk-aligned prefix for future prefetches.
    # Done *before* returning so it's already available for the next
    # agent call regardless of streaming completion.
    if request.record_in_registry:
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

    # 5) Return whatever the chat handler produced.
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    if isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())
    return StreamingResponse(content=generator, media_type="text/event-stream")


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
