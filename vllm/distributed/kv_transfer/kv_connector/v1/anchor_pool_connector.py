# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AnchorPoolConnector — proper v1 scheduler-side prefill skip for the
chunked_chat anchor pool.

How it fits with the existing components:

    chunked_chat serving       stamps vllm_xargs["anchor_pool_spans"]
        │
        ▼
    Request enters scheduler
        │
        ▼
    AnchorPoolConnector (SCHEDULER role)
        ├─ get_num_new_matched_tokens(req):
        │       reads anchor_pool_spans from request.sampling_params.
        │       Returns the contiguous leading-prefix token count that
        │       is in self._populated_hashes  →  scheduler treats those
        │       as already-computed and only schedules prefill on the tail.
        │
        ├─ build_connector_meta(scheduler_output):
        │       builds AnchorPoolConnectorMetadata with the inject ops
        │       for newly-scheduled requests.
        │
        └─ request_finished(req):
                adds all hashes that this request produced to
                _populated_hashes (optimistic — capture happened in
                AnchorPoolWorkerExtension.process_pending).
        ▼
    AnchorPoolConnector (WORKER role).start_load_kv(...)
        For each inject op: gather pool.base() → re-rotate to t_start →
        scatter into the request's allocated blocks.
        Model forward then runs only over the un-cached tail.

The actual capture-after-prefill pipeline stays in
`AnchorPoolWorkerExtension` (the existing worker_ext + process_pending
hook). This connector reads from the same `AnchorPoolManager` via the
module-level singleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


logger = init_logger(__name__)

# xargs keys this connector reads. Mirrors the keys the worker_ext
# already understands so the chunked_chat serving layer only has to
# stamp them once.
_XARG_ANCHOR_SPANS = "anchor_pool_spans"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class _InjectOp:
    """One inject op the scheduler wants the worker to perform this step."""

    req_id: str
    chunk_hash: str
    t_start: int
    num_tokens: int
    block_ids: list[int]


@dataclass
class AnchorPoolConnectorMetadata(KVConnectorMetadata):
    """Per-step metadata: list of inject ops to perform pre-forward."""

    injects: list[_InjectOp] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class AnchorPoolConnector(KVConnectorBase_V1):
    """Scheduler-aware prefill skip for static anchor chunks.

    Configured via `--kv-transfer-config '{"kv_connector":
    "AnchorPoolConnector", "kv_role": "kv_both"}'`.

    Stateless on the worker side beyond a module-level reference to
    the running `AnchorPoolRunnerState` (set by the worker extension).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        # Scheduler-side state.
        self._populated_hashes: set[str] = set()
        # req_id -> (claimed_tokens, list of inject ops to put in next metadata)
        self._pending: dict[str, list[_InjectOp]] = {}
        # Worker-side: lazy lookup; resolved on first call.
        self._block_size: int | None = None
        self._kv_caches: dict[str, torch.Tensor] | None = None
        logger.info("AnchorPoolConnector initialized role=%s", role.name)

    # ==================================================================
    # Scheduler-side
    # ==================================================================
    def _read_spans(self, request: "Request") -> list[dict[str, Any]]:
        sp = getattr(request, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) if sp is not None else None
        if not extra:
            return []
        spans = extra.get(_XARG_ANCHOR_SPANS)
        if not spans:
            return []
        # Defensive copy so we never mutate the request object.
        return [dict(s) for s in spans]

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Only claim cached tokens at first-scheduling, before any local
        # computation; otherwise our injects could collide with already-
        # computed KV.
        if num_computed_tokens > 0:
            return 0, False

        spans = self._read_spans(request)
        if not spans:
            return 0, False

        # Sort spans by t_start; we'll walk a contiguous leading prefix.
        spans = sorted(spans, key=lambda s: int(s["t_start"]))

        claimed = 0
        ops: list[_InjectOp] = []
        cursor = 0
        for span in spans:
            t_start = int(span["t_start"])
            num_tokens = int(span["num_tokens"])
            chunk_hash = str(span["chunk_hash"])

            # We only skip a CONTIGUOUS leading prefix. Stop if this
            # span doesn't sit right after the running cursor.
            if t_start != cursor:
                break
            if chunk_hash not in self._populated_hashes:
                break
            ops.append(
                _InjectOp(
                    req_id=request.request_id,
                    chunk_hash=chunk_hash,
                    t_start=t_start,
                    num_tokens=num_tokens,
                    block_ids=[],  # filled in build_connector_meta
                )
            )
            claimed += num_tokens
            cursor = t_start + num_tokens

        if claimed == 0:
            return 0, False

        self._pending[request.request_id] = ops
        logger.info(
            "AnchorPoolConnector: claim req=%s tokens=%d ops=%d",
            request.request_id,
            claimed,
            len(ops),
        )
        # Synchronous load (we do it pre-forward, not asynchronously).
        return claimed, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        # block_ids resolution happens in build_connector_meta from
        # scheduler_output, which is more reliable. Nothing to do here.
        return

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        meta = AnchorPoolConnectorMetadata()
        if not self._pending:
            return meta

        # Resolve block_ids for each newly-scheduled request from the
        # scheduler output. block_ids is a tuple-of-lists (one per KV
        # group), we use group 0 (single group is the common case).
        new_req_blocks: dict[str, list[int]] = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            bids = new_req.block_ids
            if isinstance(bids, tuple):
                bids = bids[0] if bids else []
            new_req_blocks[new_req.req_id] = [int(b) for b in bids]

        ready: list[str] = []
        for req_id, ops in self._pending.items():
            blocks = new_req_blocks.get(req_id)
            if blocks is None:
                # Not yet scheduled (or finished before scheduling).
                # Leave in pending; will retry next step.
                continue
            for op in ops:
                op.block_ids = blocks
                meta.injects.append(op)
            ready.append(req_id)

        for req_id in ready:
            self._pending.pop(req_id, None)

        if meta.injects:
            logger.info(
                "AnchorPoolConnector: dispatching %d inject ops to worker",
                len(meta.injects),
            )
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # Optimistic admission: assume the worker extension successfully
        # captured the bases for any anchor spans this request had.
        spans = self._read_spans(request)
        added: list[str] = []
        for span in spans:
            h = str(span["chunk_hash"])
            if h not in self._populated_hashes:
                self._populated_hashes.add(h)
                added.append(h)
        if added:
            logger.info(
                "AnchorPoolConnector: mark populated %d hashes "
                "(total=%d)",
                len(added),
                len(self._populated_hashes),
            )
        # Clean up any leftover pending state.
        self._pending.pop(request.request_id, None)
        return False, None

    # ==================================================================
    # Worker-side
    # ==================================================================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Capture the per-layer kv-cache tensor map for inject scatter.
        # Map preserves insertion order, which mirrors layer order.
        self._kv_caches = kv_caches
        if kv_caches:
            sample = next(iter(kv_caches.values()))
            # FA: (2, num_blocks, block_size, kv_heads, head_size) → block dim 1
            # MLA: (num_blocks, block_size, kv_heads, head_size)    → block dim 0
            self._kv_has_kv_split = sample.shape[0] == 2

    def _get_anchor_state(self):
        # Lazy import to avoid pulling in the worker_ext module at
        # scheduler-process import time.
        from vllm.entrypoints.openai.cached_chat.anchor_pool_worker_ext import (
            get_active_anchor_pool_state,
        )

        return get_active_anchor_pool_state()

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, AnchorPoolConnectorMetadata):
            return
        if not meta.injects:
            return

        state = self._get_anchor_state()
        if state is None:
            logger.warning(
                "AnchorPoolConnector.start_load_kv: anchor pool state not "
                "yet installed on this worker; skipping %d injects",
                len(meta.injects),
            )
            return

        for op in meta.injects:
            try:
                self._inject_one(state, op)
            except Exception as e:
                logger.warning(
                    "AnchorPoolConnector.start_load_kv: inject failed "
                    "req=%s chunk=%s: %s",
                    op.req_id,
                    op.chunk_hash[:12],
                    e,
                )

    def _inject_one(self, state, op: _InjectOp) -> None:
        if not state.manager.has_pool(op.chunk_hash):
            logger.warning(
                "AnchorPoolConnector: no pool for chunk=%s; skip inject",
                op.chunk_hash[:12],
            )
            return
        pool = state.manager.pool(op.chunk_hash)
        base = pool.base()
        if base is None:
            logger.warning(
                "AnchorPoolConnector: pool for chunk=%s has no base",
                op.chunk_hash[:12],
            )
            return
        K_base, V_base = base
        if pool.num_tokens != op.num_tokens:
            logger.warning(
                "AnchorPoolConnector: pool.num_tokens=%d != op.num_tokens=%d",
                pool.num_tokens,
                op.num_tokens,
            )
            return

        # Re-rotate K back to the request's position offset.
        K = state.rerotate_from_position_zero(K_base, op.t_start)
        V = V_base

        # Scatter into the request's blocks. We use the same gather/scatter
        # geometry the worker extension does — by abs token position.
        block_size = state.block_size
        positions = torch.arange(
            op.t_start,
            op.t_start + op.num_tokens,
            dtype=torch.long,
            device=state.device,
        )
        block_pos = positions // block_size
        slot_pos = positions % block_size
        block_ids_t = torch.tensor(
            op.block_ids, dtype=torch.long, device=state.device
        )
        physical_blocks = block_ids_t[block_pos]

        kv_caches = state._kv_cache_tensors()
        for layer_idx, layer_kv in enumerate(kv_caches):
            if state._kv_has_kv_split:
                layer_kv[0, physical_blocks, slot_pos] = K[layer_idx]
                layer_kv[1, physical_blocks, slot_pos] = V[layer_idx]
            else:
                layer_kv[physical_blocks, slot_pos] = K[layer_idx]
        logger.info(
            "AnchorPoolConnector: injected req=%s chunk=%s t_start=%d "
            "tokens=%d",
            op.req_id,
            op.chunk_hash[:12],
            op.t_start,
            op.num_tokens,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Synchronous inject in start_load_kv; nothing to wait on.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # Capture is driven by AnchorPoolWorkerExtension.process_pending,
        # not by this connector. Nothing to do per-layer.
        return

    def wait_for_save(self) -> None:
        return
