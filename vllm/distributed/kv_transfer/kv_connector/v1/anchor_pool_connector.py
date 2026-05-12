# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AnchorPoolConnector — entropy-gated v1 prefill skip for the
chunked_chat anchor pool.

Decision flow (single gate, pre-forward):

    chunked_chat serving       stamps vllm_xargs["anchor_pool_spans"]
        │
        ▼
    AnchorPoolConnector (SCHEDULER role)
        ├─ get_num_new_matched_tokens(req):
        │       reads anchor_pool_spans and, for each span, queries the
        │       worker's cached admit verdict
        │       (AnchorPoolWorkerExtension.get_admit_verdict).
        │
        │       verdict == False  → pool is saturated, chunk is well-
        │                           explained by existing anchors. The
        │                           span is claimed for skip; the inject
        │                           path will scatter a blended KV.
        │
        │       verdict == True   → chunk is novel; the span is NOT
        │       (or unset)         claimed, dense prefill runs, and the
        │                          worker captures a new anchor in
        │                          _handle_span post-prefill.
        │
        ├─ build_connector_meta:
        │       emits one _InjectOp per claimed run, carrying the
        │       caller's blend_temperature.
        ▼
    AnchorPoolConnector (WORKER role).start_load_kv(...)
        For each inject op: run blend_anchors(pool, real_ph=base, ...)
        → rerotate K to t_start → scatter blended (K, V) into the
        request's allocated blocks. Model forward then sees these as
        already-computed tokens.

Capture-after-prefill stays in `AnchorPoolWorkerExtension` (only the
admit decision and the blend moved here). After every successful
capture the worker refreshes its admit verdict for the chunk_hash, so
the next request's scheduling decision picks up the new pool state.
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
_XARG_BLEND_TEMPERATURE = "anchor_pool_blend_temperature"


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
    blend_temperature: float = 1.0


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
        # req_id -> list of (start, length, chunk_hash) — every populated
        # span for this request (leading + non-leading). The trailing run
        # is trimmed by 1 token if it would otherwise cover the last
        # prompt position (so the dense forward always sees at least one
        # token at the end, producing logits for the first decode step).
        self._req_runs: dict[str, list[tuple[int, int, str]]] = {}
        # req_id -> set of (start, length) for runs whose inject op has
        # already been emitted to the worker. Prevents double-injecting
        # across the multiple steps of a chunked prefill.
        self._req_emitted: dict[str, set[tuple[int, int]]] = {}
        # req_id -> blend temperature, captured at first-scheduling.
        # Threaded into each _InjectOp so the worker-side blend uses the
        # caller-configured temperature.
        self._req_blend_temperatures: dict[str, float] = {}
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

    def _populated_runs_for(
        self,
        request: "Request",
    ) -> list[tuple[int, int, str]]:
        """Return injectable runs for `request`, sorted by start, with
        the trailing run trimmed by 1 if it covers the last prompt
        position. Cached per request via `self._req_runs`.

        Admit gate: a run is injectable only if the worker's admit
        verdict for its `chunk_hash` is False (i.e. the pool has been
        saturated enough that the chunk is well-explained by existing
        anchors — safe to skip dense prefill and use a blended
        approximation). Runs whose verdict is True or unset (no pool
        observation yet) are dropped here so dense prefill runs for
        them; the worker's `_handle_span` will then capture a new
        anchor and refresh the verdict.
        """
        cached = self._req_runs.get(request.request_id)
        if cached is not None:
            return cached

        spans = self._read_spans(request)
        if not spans:
            self._req_runs[request.request_id] = []
            return []

        # Stash blend temperature for this request — picked up by
        # build_connector_meta when emitting inject ops.
        sp = getattr(request, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) if sp is not None else None
        if extra:
            try:
                self._req_blend_temperatures[request.request_id] = float(
                    extra.get(_XARG_BLEND_TEMPERATURE, 1.0)
                )
            except (TypeError, ValueError):
                self._req_blend_temperatures[request.request_id] = 1.0

        state = self._get_anchor_state()

        spans = sorted(spans, key=lambda s: int(s["t_start"]))
        runs: list[tuple[int, int, str]] = []
        for span in spans:
            t_start = int(span["t_start"])
            num_tokens = int(span["num_tokens"])
            chunk_hash = str(span["chunk_hash"])
            if num_tokens <= 0:
                continue
            verdict: bool | None
            if state is None:
                # Worker state not yet attached (very first request before
                # anchor_pool_install RPC has resolved). Treat as no
                # verdict: run prefill, let capture populate the pool.
                verdict = None
            else:
                verdict = state.get_admit_verdict(chunk_hash)
            if verdict is not False:
                # verdict True (novel) or None (no observation yet) -> let
                # dense prefill run for this span so the worker can
                # capture base / new anchor.
                continue
            runs.append((t_start, num_tokens, chunk_hash))

        # Trailing trim: leave at least one dense position at the end so
        # the forward produces logits for sampling. Only applies if a
        # populated run ends exactly at the last prompt position.
        num_prompt = request.num_prompt_tokens
        if runs:
            t_start, num_tokens, h = runs[-1]
            if t_start + num_tokens >= num_prompt:
                new_len = max(0, (num_prompt - 1) - t_start)
                if new_len <= 0:
                    runs.pop()
                else:
                    runs[-1] = (t_start, new_len, h)

        self._req_runs[request.request_id] = runs
        return runs

    def get_external_runs(
        self,
        request: "Request",
    ) -> list[tuple[int, int]]:
        return [(s, l) for s, l, _ in self._populated_runs_for(request)]

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

        runs = self._populated_runs_for(request)
        if not runs:
            return 0, False

        # Length of the contiguous leading prefix run starting at 0 — what
        # the scheduler's `num_external_computed_tokens` mechanism handles.
        # The remaining non-leading runs are surfaced via `get_external_runs`
        # and handled by the scheduler's per-step run-boundary advance.
        leading = 0
        cursor = 0
        for t_start, num_tokens, _ in runs:
            if t_start != cursor:
                break
            leading += num_tokens
            cursor = t_start + num_tokens

        total_runs_tokens = sum(l for _, l, _ in runs)
        non_leading = total_runs_tokens - leading
        logger.info(
            "AnchorPoolConnector: claim req=%s leading=%d non_leading=%d "
            "total_runs=%d",
            request.request_id,
            leading,
            non_leading,
            len(runs),
        )
        if leading > 0:
            logger.info(
                "TOKENS SKIPPED DURING PREFILL (leading): %d (REQ=%s)",
                leading,
                request.request_id,
            )
        if non_leading > 0:
            logger.info(
                "TOKENS SKIPPED DURING PREFILL (non-leading runs): %d "
                "across %d runs (REQ=%s)",
                non_leading,
                len(runs) - (1 if leading > 0 else 0),
                request.request_id,
            )

        # Synchronous load — we inject pre-forward, not asynchronously.
        return leading, False

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
        """Emit inject ops for every populated run whose entire span has
        been advanced past by the scheduler (run.end <= num_computed) and
        which we haven't already emitted.

        Block resolution is deferred to the worker side: leaving
        `op.block_ids` empty signals `_inject_one` to look them up via
        the running anchor-pool state, which always has the live block
        table for the request. This avoids having to track new_block_ids
        across chunked-prefill steps on the scheduler side."""
        meta = AnchorPoolConnectorMetadata()
        if not self._req_runs:
            return meta

        # Walk both newly-scheduled and continuing requests to learn each
        # request's current num_computed_tokens.
        req_num_computed: dict[str, int] = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            req_num_computed[new_req.req_id] = int(new_req.num_computed_tokens)
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            req_num_computed[rid] = int(cached.num_computed_tokens[i])

        for req_id, runs in self._req_runs.items():
            num_computed = req_num_computed.get(req_id)
            if num_computed is None:
                continue
            emitted = self._req_emitted.setdefault(req_id, set())
            blend_temp = self._req_blend_temperatures.get(req_id, 1.0)
            for t_start, num_tokens, chunk_hash in runs:
                if t_start + num_tokens > num_computed:
                    # Scheduler hasn't advanced past this run yet — wait
                    # for a later step.
                    continue
                key = (t_start, num_tokens)
                if key in emitted:
                    continue
                meta.injects.append(
                    _InjectOp(
                        req_id=req_id,
                        chunk_hash=chunk_hash,
                        t_start=t_start,
                        num_tokens=num_tokens,
                        block_ids=[],  # worker resolves at inject time
                        blend_temperature=blend_temp,
                    )
                )
                emitted.add(key)

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
        # Clean up per-request state.
        self._req_runs.pop(request.request_id, None)
        self._req_emitted.pop(request.request_id, None)
        self._req_blend_temperatures.pop(request.request_id, None)
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

        # Inject-time blend: use base K/V as the "real_ph" proxy (we
        # don't have a real prefill output here — that's the whole point
        # of skipping prefill). The blend weights anchor deltas by
        # similarity-to-base; for a single anchor the result reduces to
        # that anchor's stored K/V, for N anchors it's a softmax-weighted
        # combination favoring anchors whose stored K/V are closest to
        # the base.
        #
        # blend_anchors operates in canonical position-0 frame; K_base
        # and V_base are stored derotated, so they go in as-is. The
        # corrected K is then rerotated to op.t_start before scatter.
        zero_pf_K = torch.zeros_like(K_base)
        zero_pf_V = torch.zeros_like(V_base)
        try:
            (
                K_pos0,
                V,
                _pf_K,
                _pf_V,
            ) = state.blend_anchors(
                pool,
                real_ph_K=K_base,
                real_ph_V=V_base,
                base_pf_K=zero_pf_K,
                base_pf_V=zero_pf_V,
                temperature=float(op.blend_temperature),
                skip_layer_zero=True,
            )
        except Exception as e:
            logger.warning(
                "AnchorPoolConnector: blend failed chunk=%s; falling "
                "back to base-only inject: %s",
                op.chunk_hash[:12],
                e,
            )
            K_pos0 = K_base
            V = V_base
        K = state.rerotate_from_position_zero(K_pos0, op.t_start)

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
        # Resolve the request's block list. Connector-scheduled inject ops
        # (non-leading runs especially) carry empty op.block_ids — the
        # runner-side state always has the up-to-date allocation, so use
        # that as the source of truth.
        if op.block_ids:
            block_id_list = list(op.block_ids)
        else:
            try:
                block_id_list = state._block_ids_for(op.req_id)
            except Exception as e:
                logger.warning(
                    "AnchorPoolConnector: block-id lookup failed req=%s "
                    "chunk=%s: %s",
                    op.req_id,
                    op.chunk_hash[:12],
                    e,
                )
                return
        block_ids_t = torch.tensor(
            block_id_list, dtype=torch.long, device=state.device
        )
        physical_blocks = block_ids_t[block_pos]

        kv_caches = state._kv_cache_tensors()
        for layer_idx, layer_kv in enumerate(kv_caches):
            if state._kv_has_kv_split:
                layer_kv[0, physical_blocks, slot_pos] = K[layer_idx]
                layer_kv[1, physical_blocks, slot_pos] = V[layer_idx]
            else:
                layer_kv[physical_blocks, slot_pos] = K[layer_idx]
        # Tell the worker extension this span's dense prefill was skipped,
        # so its post-prefill hook knows to apply the blend correction
        # (instead of leaving freshly-prefilled K/V alone).
        try:
            state.mark_injected(op.req_id, op.chunk_hash)
        except Exception as e:
            logger.warning(
                "AnchorPoolConnector: mark_injected failed req=%s chunk=%s: %s",
                op.req_id,
                op.chunk_hash[:12],
                e,
            )
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
