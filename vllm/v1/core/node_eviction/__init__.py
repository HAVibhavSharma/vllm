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

Prefetch *origination* is deliberately not here: phantoms are submitted by
whoever calls `POST /v1/agents/prefetch`. This package only records that a
block arrived speculatively and reports whether anything ever used it.

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
    "ScoreBreakdown",
    "SnapshotSource",
    "StaticSnapshotSource",
    "maybe_build_controller",
    "per_engine_log_path",
    "node_key_for_request",
]


def per_engine_log_path(path: str | None, data_parallel_index: int) -> str | None:
    """`path` for the engine at `data_parallel_index`: unchanged for index 0,
    `<stem>-dp<index><suffix>` beside it for the others.

    Under data parallelism every engine core inherits the one environment, so
    each would open the same decision and access logs. Two processes
    appending lines longer than a write buffer interleave them, and the logs
    are only readable per engine anyway: each engine has its own GPU cache,
    and a block hash evicted on one says nothing about the other.
    """
    if not path or data_parallel_index <= 0:
        return path
    import os.path

    stem, suffix = os.path.splitext(path)
    return f"{stem}-dp{data_parallel_index}{suffix}"


def maybe_build_controller(
    block_pool,
    num_kv_cache_groups: int,
    config: NodeEvictionConfig | None = None,
    block_size_bytes: int = 0,
    data_parallel_index: int = 0,
) -> NodeEvictionController | None:
    """Build the controller if the policy is switched on, else None.

    Returning None rather than a disabled object is deliberate: the call
    sites are then a single `is not None` check, and flag-off does no work at
    all. `data_parallel_index` is the engine's DP rank, which names its logs
    (`per_engine_log_path`).
    """
    import dataclasses

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if config is None:
        config = NodeEvictionConfig.from_env()
    if not config.enabled:
        return None
    if data_parallel_index > 0:
        config = dataclasses.replace(
            config,
            decision_log_path=per_engine_log_path(
                config.decision_log_path, data_parallel_index
            ),
            access_log_path=per_engine_log_path(
                config.access_log_path, data_parallel_index
            ),
        )

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
    )
    if not controller.enabled:
        return None

    controller.start()
    if config.basic:
        logger.info(
            "Node-aware KV eviction in BASIC mode: no scored reordering; "
            "blocks a phantom prefetch warmed are evicted only after every "
            "ordinary free block, until used or for %.0fs (variant=basic).",
            config.warm_protect_s,
        )
    elif config.observe_only:
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
