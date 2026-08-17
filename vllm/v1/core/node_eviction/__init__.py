# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-aware KV cache eviction.

LRU evicts by age. In an agentic workflow the prefix that has been idle
longest is usually the one waiting on a tool call — which is exactly the one
about to be needed again. Age is not merely a weak signal here; it is
*anti-correlated* with need.

This package replaces the age ordering with a forecast-driven one. vLLM
maintains a `(job_id, node, call_type) -> blocks` index that outlives the
requests that created those blocks, scores every tracked key on a
wall-clock tick from a forecast published to Redis by the workflow, and
moves the worst-scoring blocks to the head of the free queue.

Design docs live in `plan/new-eviction/`. Start with `06-walkthrough.md` for
why, `02-controller-scope.md` §10 for the build order.

The policy is **off by default**. With `VLLM_NODE_EVICTION_POLICY` unset,
nothing in this package runs and the free queue behaves exactly as upstream —
that is the A/B baseline arm, and it is a property worth keeping.
"""

from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.controller import (
    NodeEvictionController,
    node_key_for_request,
)
from vllm.v1.core.node_eviction.index import BlockOwnershipIndex, NodeEntry
from vllm.v1.core.node_eviction.metrics import EvictionCounters, EvictionObserver
from vllm.v1.core.node_eviction.snapshot import (
    JSONFileSnapshotSource,
    SnapshotSource,
    StaticSnapshotSource,
)
from vllm.v1.core.node_eviction.types import (
    ImportanceRow,
    ImportanceSnapshot,
    NodeKey,
    ScoreBreakdown,
)
from vllm.v1.core.node_eviction.wantlist import PrefetchWant, PrefetchWantList

__all__ = [
    "BlockOwnershipIndex",
    "EvictionCounters",
    "EvictionObserver",
    "ImportanceRow",
    "ImportanceSnapshot",
    "JSONFileSnapshotSource",
    "NodeEntry",
    "NodeEvictionConfig",
    "NodeEvictionController",
    "NodeKey",
    "PrefetchWant",
    "PrefetchWantList",
    "ScoreBreakdown",
    "SnapshotSource",
    "StaticSnapshotSource",
    "maybe_build_controller",
    "node_key_for_request",
]


def maybe_build_controller(
    block_pool,
    num_kv_cache_groups: int,
    config: NodeEvictionConfig | None = None,
    block_size_bytes: int = 0,
    block_size_tokens: int = 0,
) -> NodeEvictionController | None:
    """Build the controller if the policy is switched on, else None.

    Returning None rather than a disabled object is deliberate: the call
    sites are then a single `is not None` check, and flag-off does no work at
    all.
    """
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if config is None:
        config = NodeEvictionConfig.from_env()
    if not config.enabled:
        return None

    try:
        config.validate()
    except ValueError:
        logger.exception(
            "Invalid node eviction config; falling back to LRU",
        )
        return None

    from vllm.v1.core.node_eviction.redis_source import build_snapshot_source

    source = build_snapshot_source(config)
    controller = NodeEvictionController(
        block_pool=block_pool,
        config=config,
        snapshot_source=source,
        num_kv_cache_groups=num_kv_cache_groups,
        block_size_bytes=block_size_bytes,
        block_size_tokens=block_size_tokens,
    )
    if not controller.enabled:
        return None

    controller.start()
    if config.observe_only:
        logger.info(
            "Node-aware KV eviction in OBSERVE-ONLY mode: the free queue is "
            "never reordered and eviction stays LRU. Bookkeeping and the "
            "kv_hbm line (variant=baseline) run so this arm can be diffed "
            "against a policy run. (tick=%.0fms, tau=%.0fms, redis=%s)",
            config.tick_period_ms,
            config.tau_ms,
            config.redis_url or "none",
        )
    else:
        logger.info(
            "Node-aware KV eviction enabled (tick=%.0fms, K=%d, tau=%.0fms, "
            "redis=%s)",
            config.tick_period_ms,
            config.splice_max_blocks,
            config.tau_ms,
            config.redis_url or "none",
        )
    return controller
