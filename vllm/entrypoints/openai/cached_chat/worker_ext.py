# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side half of the manual KV-cache pipeline.

Loaded into the worker process via:

    --worker-extension-cls \\
        vllm.entrypoints.openai.cached_chat.worker_ext.ManualKVWorkerExtension

Two control fields, set by the API serving via ``vllm_xargs`` and surfaced
on the request as ``sampling_params.extra_args[...]``:

    manual_kv_handle              : str   — the handle to capture/inject on
    manual_kv_prefilled_tokens    : int   — if > 0, INJECT this many tokens
    manual_kv_capture_after_prefill : bool — if True, CAPTURE after prefill

The capture and inject steps run inside the worker via a small hook in
``GPUModelRunner._update_states`` that delegates to
:meth:`_ManualKVRunnerState.process_pending`. The extension also exposes
``manual_kv_drop`` and ``manual_kv_block_size`` over ``collective_rpc`` so
the API process can manage eviction and discover the block size.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


EXTRA_KEY_HANDLE = "manual_kv_handle"
EXTRA_KEY_PREFILLED_TOKENS = "manual_kv_prefilled_tokens"
EXTRA_KEY_CAPTURE = "manual_kv_capture_after_prefill"


class _ManualKVRunnerState:
    """Per-runner manual-KV state.

    Holds:
        _blobs[handle]          : list[CPU pinned tensor], one per layer.
                                  First dim is "block index within the
                                  saved prefix"; rest mirrors the runner's
                                  own kv_caches[i] layout.
        _blob_lengths[handle]   : how many prompt tokens are saved.
        _pending_captures[req_id]: (handle, num_prompt_tokens). Populated
                                  when a new request comes in with the
                                  capture flag; consumed once
                                  num_computed_tokens >= num_prompt_tokens.
    """

    def __init__(self, model_runner: Any) -> None:
        self._runner = model_runner
        self._lock = threading.Lock()
        self._blobs: dict[str, list[torch.Tensor]] = {}
        self._blob_lengths: dict[str, int] = {}
        self._pending_captures: dict[str, tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # block_size discovery
    # ------------------------------------------------------------------
    @property
    def block_size(self) -> int:
        # Prefer the runner's input_batch if it has a populated MultiGroupBlockTable.
        ib = getattr(self._runner, "input_batch", None)
        bt = getattr(ib, "block_table", None) if ib is not None else None
        block_tables = getattr(bt, "block_tables", None) if bt is not None else None
        if block_tables:
            bs = getattr(block_tables[0], "block_size", None)
            if bs:
                return int(bs)
        cache_config = getattr(self._runner, "cache_config", None)
        if cache_config is not None and getattr(cache_config, "block_size", None):
            return int(cache_config.block_size)
        raise RuntimeError("Could not determine block_size from model runner")

    # ------------------------------------------------------------------
    # Single entry point called from _update_states.
    # ------------------------------------------------------------------
    def process_pending(self, scheduler_output: "SchedulerOutput") -> None:
        """Inject blobs for new requests; capture blobs for finished prefills.

        Order matters: inject must run on the very step the new request
        enters the batch (because that's when the freshly-allocated blocks
        need to be filled before forward); capture runs after prefill is
        observed via num_computed_tokens.
        """
        # Drop pending-capture state for any request the scheduler just
        # finished — we missed the window for those.
        for finished_id in scheduler_output.finished_req_ids:
            self._pending_captures.pop(finished_id, None)

        block_size = self.block_size

        # ---- 1. Inject + record-capture-pending for new requests ------
        for new_req_data in scheduler_output.scheduled_new_reqs:
            sp = getattr(new_req_data, "sampling_params", None)
            extra = getattr(sp, "extra_args", None) if sp is not None else None
            if not extra:
                continue
            handle = extra.get(EXTRA_KEY_HANDLE)
            if not handle:
                continue

            prefilled = int(extra.get(EXTRA_KEY_PREFILLED_TOKENS) or 0)
            if prefilled > 0:
                self._do_inject(new_req_data, handle, prefilled, block_size)

            if extra.get(EXTRA_KEY_CAPTURE):
                num_prompt_tokens = self._num_prompt_tokens_for(new_req_data.req_id)
                if num_prompt_tokens is None:
                    continue
                self._pending_captures[new_req_data.req_id] = (
                    handle,
                    num_prompt_tokens,
                )

        # ---- 2. Capture blobs whose prefill just finished -------------
        if self._pending_captures:
            ready: list[str] = []
            for req_id, (handle, num_prompt_tokens) in self._pending_captures.items():
                req_state = self._runner.requests.get(req_id)
                if req_state is None:
                    # Request was removed without us seeing prefill complete
                    # (e.g. aborted). Drop.
                    ready.append(req_id)
                    continue
                if req_state.num_computed_tokens < num_prompt_tokens:
                    continue
                try:
                    self._capture_request(req_id, num_prompt_tokens, handle, block_size)
                except Exception as e:
                    logger.warning(
                        "[manual-kv] capture failed for req_id=%s handle=%s: %s",
                        req_id,
                        str(handle)[:12],
                        e,
                    )
                ready.append(req_id)
            for req_id in ready:
                self._pending_captures.pop(req_id, None)

    # ------------------------------------------------------------------
    # Inject implementation
    # ------------------------------------------------------------------
    def _do_inject(
        self,
        new_req_data: Any,
        handle: str,
        prefilled: int,
        block_size: int,
    ) -> None:
        with self._lock:
            blob = self._blobs.get(handle)
            stored_tokens = self._blob_lengths.get(handle, 0)
        if blob is None:
            logger.warning(
                "[manual-kv] inject MISS handle=%s req_id=%s — blob not on "
                "this worker; falling through to fresh prefill",
                str(handle)[:12],
                new_req_data.req_id,
            )
            return

        num_blocks = prefilled // block_size
        if num_blocks == 0:
            return
        if num_blocks * block_size > stored_tokens:
            logger.warning(
                "[manual-kv] inject SHORT handle=%s req_id=%s — request "
                "asks for %d tokens but blob only has %d; truncating",
                str(handle)[:12],
                new_req_data.req_id,
                prefilled,
                stored_tokens,
            )
            num_blocks = stored_tokens // block_size
            if num_blocks == 0:
                return

        req_block_ids_groups = new_req_data.block_ids
        req_block_ids = (
            req_block_ids_groups[0]
            if isinstance(req_block_ids_groups, tuple)
            else req_block_ids_groups
        )
        target_block_ids = list(req_block_ids[:num_blocks])
        if len(target_block_ids) < num_blocks:
            logger.warning(
                "[manual-kv] inject SKIP req_id=%s — only %d allocated "
                "blocks for %d requested",
                new_req_data.req_id,
                len(target_block_ids),
                num_blocks,
            )
            return

        kv_caches = self._kv_cache_tensors()
        if len(kv_caches) != len(blob):
            logger.warning(
                "[manual-kv] inject SKIP handle=%s req_id=%s — layer count "
                "mismatch: cache has %d, blob has %d",
                str(handle)[:12],
                new_req_data.req_id,
                len(kv_caches),
                len(blob),
            )
            return

        sample = kv_caches[0]
        block_dim = 1 if sample.shape[0] == 2 else 0
        num_cache_blocks = sample.shape[block_dim]
        max_id = max(target_block_ids)
        if max_id >= num_cache_blocks:
            raise RuntimeError(
                f"manual_kv inject: block id {max_id} >= num_blocks "
                f"{num_cache_blocks} (kv shape={tuple(sample.shape)})"
            )

        for layer_idx, layer_kv in enumerate(kv_caches):
            host_buf = blob[layer_idx]
            gpu_index = torch.tensor(
                target_block_ids, dtype=torch.long, device=layer_kv.device
            )
            src = host_buf.to(layer_kv.device, non_blocking=False)
            layer_kv.index_copy_(block_dim, gpu_index, src)
            del src

        logger.info(
            "[manual-kv] inject OK handle=%s req_id=%s blocks=%d tokens=%d",
            str(handle)[:12],
            new_req_data.req_id,
            num_blocks,
            num_blocks * block_size,
        )

    # ------------------------------------------------------------------
    # Capture implementation
    # ------------------------------------------------------------------
    def _capture_request(
        self,
        req_id: str,
        num_prompt_tokens: int,
        handle: str,
        block_size: int,
    ) -> None:
        block_ids = self._block_ids_for(req_id)
        num_blocks = num_prompt_tokens // block_size
        if num_blocks == 0:
            return
        if num_blocks > len(block_ids):
            raise RuntimeError(
                f"capture: requested {num_blocks} blocks but request "
                f"{req_id} has only {len(block_ids)} allocated."
            )
        prefix_block_ids = block_ids[:num_blocks]

        kv_caches = self._kv_cache_tensors()
        if not kv_caches:
            return

        sample = kv_caches[0]
        block_dim = 1 if sample.shape[0] == 2 else 0
        num_cache_blocks = sample.shape[block_dim]
        max_id = max(prefix_block_ids)
        if max_id >= num_cache_blocks:
            raise RuntimeError(
                f"manual_kv capture: block id {max_id} >= num_blocks "
                f"{num_cache_blocks} (kv shape={tuple(sample.shape)})"
            )

        per_layer: list[torch.Tensor] = []
        index_t = torch.tensor(
            prefix_block_ids, dtype=torch.long, device=sample.device
        )
        for layer_kv in kv_caches:
            gpu_view = layer_kv.index_select(block_dim, index_t)
            host_buf = torch.empty(
                gpu_view.shape, dtype=gpu_view.dtype, pin_memory=True
            )
            host_buf.copy_(gpu_view, non_blocking=False)
            per_layer.append(host_buf)
            del gpu_view

        with self._lock:
            self._blobs[handle] = per_layer
            self._blob_lengths[handle] = num_blocks * block_size

        logger.info(
            "[manual-kv] capture OK handle=%s req_id=%s blocks=%d tokens=%d "
            "layers=%d",
            handle[:12],
            req_id,
            num_blocks,
            num_blocks * block_size,
            len(per_layer),
        )

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------
    def drop(self, handle: str) -> bool:
        with self._lock:
            had = handle in self._blobs
            self._blobs.pop(handle, None)
            self._blob_lengths.pop(handle, None)
        if had:
            logger.info("[manual-kv] drop handle=%s", handle[:12])
        return had

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _num_prompt_tokens_for(self, req_id: str) -> int | None:
        req_state = self._runner.requests.get(req_id)
        if req_state is None:
            return None
        if req_state.prompt_token_ids is not None:
            return len(req_state.prompt_token_ids)
        # Fallback for embed-only prompts.
        return getattr(req_state, "num_prompt_tokens", None)

    def _block_ids_for(self, req_id: str) -> list[int]:
        req_state = self._runner.requests.get(req_id)
        if req_state is not None and getattr(req_state, "block_ids", None):
            groups = req_state.block_ids
            group0 = groups[0] if isinstance(groups, tuple) else groups
            return list(group0)
        # Fallback: read from input_batch's row buffer.
        ib = self._runner.input_batch
        idx = ib.req_id_to_index.get(req_id)
        if idx is None:
            raise KeyError(
                f"manual_kv: request id {req_id!r} is not present on this worker."
            )
        bt = ib.block_table
        group0 = bt.block_tables[0] if hasattr(bt, "block_tables") else bt
        row = group0.block_table.np[idx]
        nonzero = (row != 0).nonzero()[0]
        if len(nonzero) == 0:
            return []
        return [int(x) for x in row[: int(nonzero[-1]) + 1]]

    def _kv_cache_tensors(self) -> list[torch.Tensor]:
        kv = getattr(self._runner, "kv_caches", None)
        if not kv:
            raise RuntimeError(
                "manual_kv: model runner has no kv_caches (worker not "
                "initialised?)"
            )
        out: list[torch.Tensor] = []
        for layer in kv:
            if isinstance(layer, torch.Tensor):
                out.append(layer)
            else:
                raise RuntimeError(
                    "manual_kv: encountered non-tensor entry in kv_caches "
                    "(probably mamba). The manual pipeline targets standard "
                    "attention models only."
                )
        return out


def _attach_state(worker: Any) -> _ManualKVRunnerState:
    runner = worker.model_runner
    state = getattr(runner, "_manual_kv_state", None)
    if state is None:
        state = _ManualKVRunnerState(runner)
        runner._manual_kv_state = state
    return state


class ManualKVWorkerExtension:
    """Mixed into the worker class via ``--worker-extension-cls``.

    Provides the small set of methods callable from the API process via
    ``engine_client.collective_rpc``. Capture and inject themselves run
    inside the engine loop (see ``_update_states`` hook); this extension
    exists to (a) install the runner state on first touch and (b) expose
    drop / block_size queries.
    """

    def manual_kv_drop(self, handle: str) -> bool:
        return _attach_state(self).drop(handle)

    def manual_kv_block_size(self) -> int:
        return _attach_state(self).block_size

    def manual_kv_install(self) -> int:
        """Force-install the runner state and return the block size."""
        return _attach_state(self).block_size
