# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register prefixes from ordinary `/v1/chat/completions` traffic.

Without this, prefetch origination is structurally unable to fire for a
workload that does not use the agent endpoints, and the failure is silent in
the worst way: engine core produces wants, the drainer drains them, and every
single one resolves to zero phantoms because the registry has never been
written. The first real run showed exactly that — 96 wants, 96 drained, 96
`no registered prefixes yet`, `speculative blocks=0`.

The registry was only ever written by `POST /v1/agents/chat/completions` and
`POST /v1/agents/prefetch` (`api_router.py:359` and `:475`). But the identity
the policy keys on — `job_id` / `langgraph_node` / `call_type` — already rides
in `vllm_xargs` on plain chat requests, because that is how engine core learns
it. So the plain path already carries everything needed to record a prefix; it
simply never did.

**Why the recorded prefix is hit-compatible with the real request.** The
phantom submits the same token ids and puts `cache_salt` in
`kv_transfer_params`, which is an LMCache key input. vLLM's own APC block
hashing keys off `request.cache_salt` (`kv_cache_utils.py:645`), which a
phantom does not set — so the phantom's blocks hash identically to those of a
real request carrying the same tokens, and a later real request hits them.
Recording under the *request's* salt rather than a synthetic `agent::` one
keeps that true on the LMCache side too.

Best-effort throughout: this runs on the serving hot path, and a prefetch
bookkeeping failure must never turn into a failed completion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.v1.agent_prefetch.hashing import (
    DEFAULT_CHUNK_SIZE,
    chunk_align,
    compute_prefix_hash,
)
from vllm.v1.agent_prefetch.registry import PrefixDescriptor

logger = init_logger(__name__)

_REGISTRY_ATTR = "agent_prefetch_registry"

# Resolved once. `NodeEvictionConfig.from_env()` re-reads a JSON file from
# disk, which is not something to do per request.
_namespace: str | None = None


def _agent_namespace() -> str:
    global _namespace
    if _namespace is None:
        from vllm.v1.core.node_eviction.config import NodeEvictionConfig

        try:
            _namespace = NodeEvictionConfig.from_env().prefetch_agent_namespace
        except Exception:
            # A malformed config file must not break chat. The engine-side
            # want will simply carry the default namespace and match.
            logger.exception(
                "agent_prefetch: could not read the agent namespace from "
                "config; falling back to 'langgraph'"
            )
            _namespace = "langgraph"
    return _namespace


def agent_id_for_extra_args(extra_args: dict[str, Any] | None) -> str | None:
    """`{namespace}:{langgraph_node}`, or None when the request carries no
    node identity.

    Must produce the same string as `PrefetchWant.agent_id`, which engine
    core builds as `f"{cfg.prefetch_agent_namespace}:{key.node}"`. A mismatch
    here is invisible — the drainer would look up an agent that exists under
    a different name and log "no registered prefixes yet" forever.
    """
    if not extra_args:
        return None
    node = extra_args.get("langgraph_node")
    if node is None:
        return None
    node = str(node)
    if not node:
        return None
    return f"{_agent_namespace()}:{node}"


def maybe_record_chat_prefix(
    app_state,
    *,
    extra_args: dict[str, Any] | None,
    model_name: str,
    prompt_token_ids: Sequence[int] | None,
    cache_salt: str | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """Record this prompt's chunk-aligned prefix under its node's agent id.

    Returns True if a descriptor was stored. Every "no" path returns False
    rather than raising:

    - prefetch origination is off, so no registry was ever created;
    - the request carries no `langgraph_node`, so it is not agent traffic;
    - the prompt is shorter than one chunk, so there is nothing LMCache
      would store anyway.
    """
    registry = getattr(app_state, _REGISTRY_ATTR, None)
    if registry is None:
        # Origination off: `maybe_start_prefetch_drainer` is what creates the
        # registry at startup, and it only runs when the drain flag is set.
        return False

    agent_id = agent_id_for_extra_args(extra_args)
    if agent_id is None:
        return False

    if not prompt_token_ids:
        return False

    try:
        aligned = chunk_align(prompt_token_ids, chunk_size=chunk_size)
        if not aligned:
            return False
        salt = cache_salt or ""
        registry.record(
            agent_id,
            PrefixDescriptor(
                token_ids=tuple(aligned),
                prefix_hash=compute_prefix_hash(model_name, salt, aligned),
                cache_salt=salt,
            ),
        )
    except Exception:
        logger.exception(
            "agent_prefetch: failed to auto-record prefix for %s", agent_id
        )
        return False

    logger.debug(
        "agent_prefetch: auto-recorded %d-token prefix for %s",
        len(aligned),
        agent_id,
    )
    return True
