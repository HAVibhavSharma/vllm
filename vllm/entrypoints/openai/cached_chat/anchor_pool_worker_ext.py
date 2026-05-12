# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker extension exposing the anchor pool over `collective_rpc`.

Loaded via:

    --worker-extension-cls \\
        vllm.entrypoints.openai.cached_chat.anchor_pool_worker_ext.\\
        AnchorPoolWorkerExtension

This is a sibling of `ManualKVWorkerExtension`; both can be loaded
together (vLLM allows multiple worker-extension classes to be mixed in
provided their attribute names don't collide).

Responsibilities:

    1. Own the per-worker `AnchorPoolManager`.
    2. Expose RPCs the API server / engine can call.
    3. Abstract away paged-attention KV layout: callers describe a chunk
       by `(req_id, t_start, num_tokens)`; this layer handles the
       block-id / slot arithmetic internally.

RoPE de/re-rotation is implemented for standard llama-style rotary
embeddings (covers Qwen2/2.5, Llama 2/3, Mistral, and similar). Models
using exotic rotary variants (DeepSeek scaling, mRoPE) will land on the
correct module via `_get_rotary_emb` (any subclass of
`RotaryEmbeddingBase`), but the inverse-rotation math has only been
verified against the plain RoPE path so far.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from vllm.entrypoints.openai.cached_chat.anchor_pool import (
    AnchorEntry,
    AnchorPool,
    AnchorPoolManager,
    PrefixSegment,
    PrefixSegmentStore,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

logger = init_logger(__name__)

# Keys consumed from sampling_params.extra_args / vllm_xargs.
EXTRA_KEY_ANCHOR_SPANS = "anchor_pool_spans"
EXTRA_KEY_ENTROPY_THRESHOLD = "anchor_pool_entropy_threshold"
EXTRA_KEY_TOP_P = "anchor_pool_top_p"
EXTRA_KEY_BLEND_TEMPERATURE = "anchor_pool_blend_temperature"


# ---------------------------------------------------------------------------
# Worker-side runner state (lazy, mirrors the pattern in worker_ext.py).
# ---------------------------------------------------------------------------


class _AnchorPoolRunnerState:
    """Per-worker handle. Holds the manager and caches model dims."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.manager = AnchorPoolManager()
        self.prefix_store = PrefixSegmentStore()
        # req_id -> (spans, num_prompt_tokens, threshold, top_p,
        #            blend_temperature)
        self._pending: dict[
            str,
            tuple[list[dict[str, Any]], int, float, float, float],
        ] = {}
        # Set of chunk_hashes whose KV the connector injected for a given
        # req_id (i.e. the spans whose dense prefill was skipped). Used to
        # gate the post-prefill capture decision: inject-skipped spans
        # already have anchor-derived KV in the cache, so they must NOT
        # be re-captured as anchors. Freshly-prefilled spans (admit=True
        # decided pre-forward by the connector) are eligible for capture.
        self._injected_spans: dict[str, set[str]] = {}
        # Admit verdict cache, keyed by chunk_hash. Computed by running
        # `predict_as_anchor` with the pool's stored base V as the
        # candidate (a deterministic per-chunk_hash signal). Updated after
        # every successful capture. The connector reads this from the
        # scheduler side to decide whether to claim a span for prefill
        # skip (verdict=False => pool explains the chunk => inject blend
        # and skip prefill).
        self._admit_verdicts: dict[str, bool] = {}
        # Per-request blend temperature, stashed when a request is queued.
        # Read by the connector's inject path so the blend at inject time
        # uses the caller-configured temperature.
        self._req_blend_temperature: dict[str, float] = {}

        cache_config = runner.vllm_config.cache_config
        model_config = runner.vllm_config.model_config

        self.block_size: int = int(cache_config.block_size)
        self.num_layers: int = int(model_config.get_num_layers(runner.vllm_config.parallel_config))
        self.num_kv_heads: int = int(
            model_config.get_num_kv_heads(runner.vllm_config.parallel_config)
        )
        self.head_size: int = int(model_config.get_head_size())

        # Probe the actual KV cache to derive the on-device dtype/device,
        # which can disagree with the configured model dtype (e.g. FP8 KV
        # over a half-precision model).
        kv_caches = self._kv_cache_tensors_unsafe()
        if not kv_caches:
            raise RuntimeError(
                "AnchorPoolWorkerExtension: KV cache is not yet allocated; "
                "install must run after the engine has finished warmup."
            )
        sample = kv_caches[0]
        self.kv_dtype: torch.dtype = sample.dtype
        self.device: torch.device = sample.device
        # FA layout is `(2, num_blocks, block_size, num_kv_heads, head_size)`;
        # MLA omits the leading 2. Detect once.
        self._kv_block_dim: int = 1 if sample.shape[0] == 2 else 0
        self._kv_has_kv_split: bool = sample.shape[0] == 2

        # Rotary module — discovered lazily, since the model isn't required
        # for paged-block gather/scatter and we want to keep that path
        # available even on models that don't use RoPE (e.g. some MLA setups).
        self._rotary_emb: RotaryEmbeddingBase | None = None

        logger.info(
            "[anchor-pool] runner-state installed layers=%d kv_heads=%d "
            "head_size=%d block_size=%d dtype=%s device=%s kv_split=%s",
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.block_size,
            self.kv_dtype,
            self.device,
            self._kv_has_kv_split,
        )

    # ------------------------------------------------------------------
    # KV-cache access
    # ------------------------------------------------------------------
    def _kv_cache_tensors_unsafe(self) -> list[torch.Tensor]:
        """Return the per-layer KV-cache tensors, or [] if not allocated."""
        # vLLM v1 stores them on the model_runner; layout varies slightly
        # across backends but `kv_caches` (list[Tensor], one per layer) is
        # the v1 contract.
        kv = getattr(self.runner, "kv_caches", None)
        if kv is None:
            return []
        return list(kv)

    def _kv_cache_tensors(self) -> list[torch.Tensor]:
        kv = self._kv_cache_tensors_unsafe()
        if not kv:
            raise RuntimeError(
                "AnchorPoolWorkerExtension: KV cache not available on runner"
            )
        return kv

    def _block_ids_for(self, req_id: str) -> list[int]:
        """Return GPU block IDs allocated to `req_id`, ordered by token
        position. Mirrors the helper in worker_ext.py."""
        req_state = self.runner.requests.get(req_id)
        if req_state is not None and getattr(req_state, "block_ids", None):
            groups = req_state.block_ids
            group0 = groups[0] if isinstance(groups, tuple) else groups
            return list(group0)

        ib = self.runner.input_batch
        idx = ib.req_id_to_index.get(req_id)
        if idx is None:
            raise KeyError(
                f"anchor_pool: request id {req_id!r} not present on this worker"
            )
        bt = ib.block_table
        group0 = bt.block_tables[0] if hasattr(bt, "block_tables") else bt
        row = group0.block_table.np[idx]
        return [int(b) for b in row if int(b) >= 0]

    # ------------------------------------------------------------------
    # Gather / scatter — abstract over paged attention
    # ------------------------------------------------------------------
    def gather_chunk_kv(
        self, req_id: str, t_start: int, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a contiguous token range out of paged blocks.

        Returns (K, V) of shape `(num_layers, num_tokens, num_kv_heads,
        head_size)` and the pool's expected dtype/device.

        Handles partial leading and trailing blocks transparently — for a
        30-token chunk with `block_size=16`, this reads 16 + 14 across
        two physical blocks and concatenates them.
        """
        if num_tokens <= 0:
            raise ValueError(f"gather_chunk_kv: num_tokens must be > 0, got {num_tokens}")
        if t_start < 0:
            raise ValueError(f"gather_chunk_kv: t_start must be >= 0, got {t_start}")

        block_size = self.block_size
        block_ids = self._block_ids_for(req_id)
        t_end = t_start + num_tokens

        if t_end > len(block_ids) * block_size:
            raise RuntimeError(
                f"gather_chunk_kv: req {req_id} has {len(block_ids)} blocks "
                f"({len(block_ids) * block_size} tokens) but caller asked "
                f"for [{t_start}, {t_end})"
            )

        positions = torch.arange(t_start, t_end, dtype=torch.long, device=self.device)
        block_pos = positions // block_size  # index into block_ids
        slot_pos = positions % block_size  # slot within that block

        block_ids_t = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        physical_blocks = block_ids_t[block_pos]  # (num_tokens,)

        kv_caches = self._kv_cache_tensors()
        K_out = torch.empty(
            (self.num_layers, num_tokens, self.num_kv_heads, self.head_size),
            dtype=self.kv_dtype,
            device=self.device,
        )
        V_out = torch.empty_like(K_out)

        for layer_idx, layer_kv in enumerate(kv_caches):
            # FA layout: (2, num_blocks, block_size, num_kv_heads, head_size)
            # MLA layout: (num_blocks, block_size, num_kv_heads, head_size)
            #             — single tensor, no K/V split (caller may not need V).
            if self._kv_has_kv_split:
                K_out[layer_idx] = layer_kv[0, physical_blocks, slot_pos]
                V_out[layer_idx] = layer_kv[1, physical_blocks, slot_pos]
            else:
                K_out[layer_idx] = layer_kv[physical_blocks, slot_pos]
                V_out[layer_idx] = layer_kv[physical_blocks, slot_pos]

        return K_out, V_out

    def scatter_chunk_kv(
        self,
        req_id: str,
        t_start: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> None:
        """Write `(K, V)` of shape `(num_layers, num_tokens, num_kv_heads,
        head_size)` into the request's paged blocks starting at token
        position `t_start`.

        Inverse of `gather_chunk_kv`. The caller is responsible for any
        position rotation already being applied to `K`.
        """
        num_tokens = int(K.shape[1])
        if V.shape[1] != num_tokens:
            raise ValueError(
                f"scatter_chunk_kv: K has {num_tokens} tokens but V has "
                f"{V.shape[1]}"
            )
        block_size = self.block_size
        block_ids = self._block_ids_for(req_id)
        t_end = t_start + num_tokens

        if t_end > len(block_ids) * block_size:
            raise RuntimeError(
                f"scatter_chunk_kv: req {req_id} has {len(block_ids)} blocks "
                f"but caller asked to write [{t_start}, {t_end})"
            )

        positions = torch.arange(t_start, t_end, dtype=torch.long, device=self.device)
        block_pos = positions // block_size
        slot_pos = positions % block_size
        block_ids_t = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        physical_blocks = block_ids_t[block_pos]

        kv_caches = self._kv_cache_tensors()
        for layer_idx, layer_kv in enumerate(kv_caches):
            if self._kv_has_kv_split:
                layer_kv[0, physical_blocks, slot_pos] = K[layer_idx]
                layer_kv[1, physical_blocks, slot_pos] = V[layer_idx]
            else:
                layer_kv[physical_blocks, slot_pos] = K[layer_idx]
                # MLA: V is unused / shares storage with K. No-op.

    # ------------------------------------------------------------------
    # RoPE de/re-rotation — Qwen2.5 / llama-style
    # ------------------------------------------------------------------
    def _get_rotary_emb(self) -> RotaryEmbeddingBase:
        """Locate the model's rotary embedding module. Cached.

        For Qwen2/llama-style models, every attention layer typically
        references the same `RotaryEmbedding` instance (often via a
        shared registered buffer), so finding the first one is enough.
        """
        if self._rotary_emb is not None:
            return self._rotary_emb
        model = getattr(self.runner, "model", None)
        if model is None:
            raise RuntimeError(
                "anchor_pool: runner has no `model` attribute; "
                "weights may not be loaded yet"
            )
        for _name, mod in model.named_modules():
            if isinstance(mod, RotaryEmbeddingBase):
                self._rotary_emb = mod
                logger.info(
                    "[anchor-pool] using rotary module %s "
                    "(rotary_dim=%d, neox_style=%s, max_pos=%d)",
                    type(mod).__name__,
                    mod.rotary_dim,
                    mod.is_neox_style,
                    mod.max_position_embeddings,
                )
                return mod
        raise RuntimeError(
            "anchor_pool: could not locate a RotaryEmbeddingBase module on "
            "the loaded model; this model may not use standard RoPE"
        )

    def _apply_rope(
        self,
        key: torch.Tensor,
        t_start: int,
        *,
        invert: bool,
    ) -> torch.Tensor:
        """Apply (or invert) RoPE on `key` of shape
        `(num_layers, num_tokens, num_kv_heads, head_size)` for absolute
        positions `[t_start, t_start + num_tokens)`.

        `invert=False` re-applies RoPE for the given positions.
        `invert=True` strips RoPE (negates `sin` to use the transpose of
        the rotation matrix — RoPE is orthogonal so transpose = inverse).
        """
        rotary = self._get_rotary_emb()
        rotary_dim = int(rotary.rotary_dim)
        is_neox = bool(rotary.is_neox_style)

        if key.ndim != 4:
            raise ValueError(
                f"_apply_rope: expected 4-D key, got shape {tuple(key.shape)}"
            )
        num_layers, num_tokens, num_kv_heads, head_size = key.shape
        if rotary_dim > head_size:
            raise RuntimeError(
                f"_apply_rope: rotary_dim {rotary_dim} > head_size {head_size}"
            )
        max_pos = int(rotary.max_position_embeddings)
        if t_start + num_tokens > max_pos:
            raise RuntimeError(
                f"_apply_rope: position range [{t_start}, {t_start + num_tokens}) "
                f"exceeds max_position_embeddings ({max_pos})"
            )

        positions = torch.arange(
            t_start,
            t_start + num_tokens,
            dtype=torch.long,
            device=key.device,
        )
        cos_sin_cache = rotary.cos_sin_cache.to(
            device=key.device, dtype=key.dtype
        )
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)  # each (num_tokens, rotary_dim // 2)

        if invert:
            sin = -sin

        # Layer dim is RoPE-independent, so flatten (layer, token) into one
        # seq dim and repeat cos/sin along it.
        cos_rep = cos.repeat(num_layers, 1)  # (num_layers * num_tokens, rotary_dim // 2)
        sin_rep = sin.repeat(num_layers, 1)

        K_flat = key.reshape(
            num_layers * num_tokens, num_kv_heads, head_size
        )
        if rotary_dim < head_size:
            K_rot_in = K_flat[..., :rotary_dim].contiguous()
            K_pass = K_flat[..., rotary_dim:]
        else:
            K_rot_in = K_flat
            K_pass = None

        K_rot_out = ApplyRotaryEmb.forward_static(
            K_rot_in,
            cos_rep,
            sin_rep,
            is_neox,
        )

        if K_pass is not None:
            K_full = torch.cat((K_rot_out, K_pass), dim=-1)
        else:
            K_full = K_rot_out

        return K_full.reshape(num_layers, num_tokens, num_kv_heads, head_size)

    def derotate_to_position_zero(
        self, key: torch.Tensor, t_start: int
    ) -> torch.Tensor:
        """Strip RoPE from `key` so its content matches what the model
        would have produced if these tokens had been at positions
        `[0, num_tokens)`.

        Inverse of `rerotate_from_position_zero(key, t_start)`.
        """
        return self._apply_rope(key, t_start, invert=True)

    def rerotate_from_position_zero(
        self, key: torch.Tensor, t_start: int
    ) -> torch.Tensor:
        """Apply RoPE to `key` for positions `[t_start, t_start +
        num_tokens)`. Use after reading a base K from the pool to make
        it usable in the live cache at a new position offset."""
        return self._apply_rope(key, t_start, invert=False)

    # ------------------------------------------------------------------
    # Engine-loop orchestration: capture-after-prefill driven by xargs.
    # ------------------------------------------------------------------
    def _num_prompt_tokens_for(self, req_id: str) -> int | None:
        req_state = self.runner.requests.get(req_id)
        if req_state is None:
            return None
        ids = getattr(req_state, "prompt_token_ids", None)
        if ids is not None:
            return len(ids)
        return getattr(req_state, "num_prompt_tokens", None)

    def mark_injected(self, req_id: str, chunk_hash: str) -> None:
        """Record that the connector skipped dense prefill for this span
        (KV was loaded from the anchor pool). Only inject-skipped spans
        are eligible for the blend write-back in `_handle_span`."""
        self._injected_spans.setdefault(req_id, set()).add(chunk_hash)

    def was_injected(self, req_id: str, chunk_hash: str) -> bool:
        return chunk_hash in self._injected_spans.get(req_id, ())

    def process_pending(self, scheduler_output: "SchedulerOutput") -> None:
        """Single hook driven from `GPUModelRunner._update_states`.

        For each newly-scheduled request that carries `anchor_pool_spans`
        in its `sampling_params.extra_args`, queue post-prefill capture
        work. Each subsequent step, check if prefill has completed; if
        so, walk the spans and:

            - if the pool for `chunk_hash` has no base yet:
                  capture_placeholder_base   (then update admit verdict)
            - else if was_injected(req, chunk_hash):
                  do nothing — connector skipped dense prefill, the live
                  KV is the anchor-derived approximation, capturing it
                  would be re-injecting the same content.
            - else:
                  capture_anchor              (then update admit verdict)

        The entropy admit decision itself has moved to the connector's
        scheduler-side path (see `AnchorPoolConnector._populated_runs_for`);
        post-prefill capture is unconditional on the "prefill actually
        ran" case.

        All errors are caught and logged — a failure in capture must
        never bring down the engine step.
        """
        # Drop pending entries for finished requests.
        for finished_id in scheduler_output.finished_req_ids:
            self._pending.pop(finished_id, None)
            self._injected_spans.pop(finished_id, None)
            self._req_blend_temperature.pop(finished_id, None)

        # 1. Queue new requests that carry anchor_pool_spans.
        for new_req in scheduler_output.scheduled_new_reqs:
            sp = getattr(new_req, "sampling_params", None)
            extra = getattr(sp, "extra_args", None) if sp is not None else None
            if not extra:
                continue
            spans = extra.get(EXTRA_KEY_ANCHOR_SPANS)
            if not spans:
                continue
            num_prompt_tokens = self._num_prompt_tokens_for(new_req.req_id)
            if num_prompt_tokens is None:
                continue
            threshold = float(extra.get(EXTRA_KEY_ENTROPY_THRESHOLD, 0.3))
            top_p = float(extra.get(EXTRA_KEY_TOP_P, 0.9))
            blend_temperature = float(
                extra.get(EXTRA_KEY_BLEND_TEMPERATURE, 1.0)
            )
            self._pending[new_req.req_id] = (
                list(spans),
                int(num_prompt_tokens),
                threshold,
                top_p,
                blend_temperature,
            )
            # Make blend_temperature visible to the connector at inject
            # time (start_load_kv runs before this request hits the
            # capture path).
            self._req_blend_temperature[new_req.req_id] = blend_temperature
            logger.info(
                "[anchor-pool] queued req=%s spans=%d prompt_tokens=%d",
                new_req.req_id,
                len(spans),
                num_prompt_tokens,
            )

        # 2. For each pending request, check if prefill has completed.
        if not self._pending:
            return
        ready: list[str] = []
        for req_id, (
            spans,
            num_prompt_tokens,
            threshold,
            top_p,
            blend_temperature,
        ) in self._pending.items():
            req_state = self.runner.requests.get(req_id)
            if req_state is None:
                ready.append(req_id)
                continue
            if int(req_state.num_computed_tokens) < num_prompt_tokens:
                continue
            # Prefill done — run captures for each span.
            for span in spans:
                try:
                    self._handle_span(
                        req_id,
                        span,
                        threshold,
                        top_p,
                    )
                except Exception as e:
                    logger.warning(
                        "[anchor-pool] span handler failed req=%s span=%s: %s",
                        req_id,
                        span,
                        e,
                    )
            ready.append(req_id)
        for req_id in ready:
            self._pending.pop(req_id, None)

    def _handle_span(
        self,
        req_id: str,
        span: dict[str, Any],
        threshold: float,
        top_p: float,
    ) -> None:
        """Capture-only post-prefill handler.

        The entropy admit decision has already been made pre-forward in
        the connector. Here we only need to:

          * First observation of a chunk_hash -> capture base.
          * If the connector skipped dense prefill (was_injected=True) ->
            do nothing; the live KV is the anchor-derived approximation.
          * Otherwise (prefill ran, i.e. admit=True or no verdict yet) ->
            capture a new anchor.

        After any successful capture, refresh the cached admit verdict
        for this chunk_hash so the connector picks it up on subsequent
        scheduling decisions.
        """
        chunk_hash = str(span["chunk_hash"])
        t_start = int(span["t_start"])
        num_tokens = int(span["num_tokens"])

        if num_tokens <= 0:
            return

        # First observation: record the base and exit.
        if not self.manager.has_pool(chunk_hash) or not self.manager.pool(
            chunk_hash
        ).has_base:
            try:
                self.capture_placeholder_base(
                    req_id=req_id,
                    chunk_hash=chunk_hash,
                    t_start=t_start,
                    num_tokens=num_tokens,
                    derotate=True,
                )
                logger.info(
                    "[anchor-pool] base captured chunk=%s tokens=%d",
                    chunk_hash[:12],
                    num_tokens,
                )
                self._recompute_admit_verdict(chunk_hash, threshold, top_p)
            except Exception as e:
                logger.warning(
                    "[anchor-pool] base capture failed chunk=%s: %s",
                    chunk_hash[:12],
                    e,
                )
            return

        # Connector skipped dense prefill — KV at this span is the
        # injected blend, not real model output. Don't capture it as a
        # new anchor; that would re-anchor the approximation.
        if self.was_injected(req_id, chunk_hash):
            logger.info(
                "[anchor-pool] skip capture chunk=%s t_start=%d tokens=%d "
                "(prefill skipped by connector inject)",
                chunk_hash[:12],
                t_start,
                num_tokens,
            )
            return

        # Prefill ran (admit=True decided pre-forward, or no verdict yet)
        # -> capture a new anchor and refresh the verdict.
        anchor_id = f"anc-{req_id[:10]}-{chunk_hash[:8]}"
        try:
            self.capture_anchor(
                req_id=req_id,
                chunk_hash=chunk_hash,
                anchor_id=anchor_id,
                ph_t_start=t_start,
                prefix_id=None,
                derotate=True,
            )
            logger.info(
                "[anchor-pool] anchor captured chunk=%s anchor=%s",
                chunk_hash[:12],
                anchor_id,
            )
            self._recompute_admit_verdict(chunk_hash, threshold, top_p)
        except Exception as e:
            logger.warning(
                "[anchor-pool] anchor capture failed chunk=%s: %s",
                chunk_hash[:12],
                e,
            )

    # ------------------------------------------------------------------
    # Admit-verdict cache (read by the connector scheduler-side)
    # ------------------------------------------------------------------
    def _recompute_admit_verdict(
        self,
        chunk_hash: str,
        threshold: float,
        top_p: float,
    ) -> None:
        """Refresh `_admit_verdicts[chunk_hash]` by running
        `predict_as_anchor` with the pool's stored base V as the
        candidate. The candidate is deterministic per chunk_hash, so the
        verdict is a property of the pool state itself, not of any
        specific request.

        verdict=False means the pool is "saturated" enough that the
        chunk is well-explained by existing anchors — the connector
        can skip dense prefill for this chunk_hash and inject a
        blended approximation instead. verdict=True means the pool
        cannot yet reconstruct the chunk reliably, so dense prefill
        must run (and the resulting K/V will be captured as a new
        anchor by `_handle_span`).
        """
        if not self.manager.has_pool(chunk_hash):
            self._admit_verdicts.pop(chunk_hash, None)
            return
        pool = self.manager.pool(chunk_hash)
        base = pool.base()
        if base is None:
            self._admit_verdicts.pop(chunk_hash, None)
            return
        _K_base, V_base = base
        # `bump=False`: this is an introspective recompute, not a real
        # admission event; don't bump anchor activations.
        result = self.predict_as_anchor(
            pool,
            V_base,
            threshold=float(threshold),
            top_p=float(top_p),
            bump=False,
        )
        n_anchors = int(result.get("n_anchors", 0))
        # n_anchors == 1 trap: a 1-element softmax has zero entropy, so
        # the entropy math always returns admit=False at n=1 regardless
        # of how anchor 1 relates to base. That would freeze the pool
        # at a single anchor forever — no later request could ever
        # capture a second one. Force admit=True here so the pool can
        # grow to at least 2 anchors; from n=2 onward the entropy math
        # has a non-trivial distribution to work with and decides on
        # its own.
        if n_anchors == 1:
            verdict = True
        else:
            verdict = bool(result.get("admit"))
        self._admit_verdicts[chunk_hash] = verdict
        logger.info(
            "[anchor-pool] admit verdict chunk=%s admit=%s n_anchors=%d "
            "entropy=%.3f/%.3f",
            chunk_hash[:12],
            verdict,
            n_anchors,
            result.get("entropy", 0.0),
            result.get("max_entropy", 0.0),
        )

    def get_admit_verdict(self, chunk_hash: str) -> bool | None:
        """Return the cached admit verdict for `chunk_hash`, or None if
        no verdict has been computed yet (e.g. the chunk has never been
        observed). Called by the connector scheduler-side."""
        return self._admit_verdicts.get(chunk_hash)

    def get_blend_temperature(self, req_id: str) -> float:
        """Return the blend temperature stashed for `req_id` at queue
        time, or 1.0 if unknown. Called by the connector inject path."""
        return self._req_blend_temperature.get(req_id, 1.0)

    # ------------------------------------------------------------------
    # Capture — three explicit primitives
    # ------------------------------------------------------------------
    def capture_placeholder_base(
        self,
        req_id: str,
        chunk_hash: str,
        t_start: int,
        num_tokens: int,
        derotate: bool = True,
    ) -> dict[str, Any]:
        """Register the pool for `chunk_hash` (idempotent) and store the
        canonical position-0 placeholder base. Errors if a base is
        already set."""
        pool = self.manager.register_pool(
            chunk_hash,
            num_tokens=num_tokens,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_dtype,
            device=self.device,
        )
        K, V = self.gather_chunk_kv(req_id, t_start, num_tokens)
        if derotate:
            K = self.derotate_to_position_zero(K, t_start)
        pool.set_base(K, V)
        return {"chunk_hash": chunk_hash, "num_tokens": num_tokens}

    def capture_prefix_base(
        self,
        req_id: str,
        prefix_id: str,
        t_start: int,
        num_tokens: int,
        derotate: bool = True,
    ) -> dict[str, Any]:
        """Gather a prefix segment from the live cache and store it (de-
        rotated to position 0) in the prefix store under `prefix_id`."""
        K, V = self.gather_chunk_kv(req_id, t_start, num_tokens)
        if derotate:
            K = self.derotate_to_position_zero(K, t_start)
        self.prefix_store.set(prefix_id, K, V)
        return {"prefix_id": prefix_id, "num_tokens": num_tokens}

    def capture_anchor(
        self,
        req_id: str,
        chunk_hash: str,
        anchor_id: str,
        ph_t_start: int,
        prefix_id: str | None = None,
        pf_t_start: int | None = None,
        pf_num_tokens: int | None = None,
        derotate: bool = True,
    ) -> dict[str, Any]:
        """Add a new anchor (six tensors) to the pool for `chunk_hash`.

        Reads the request's live KV for the placeholder span. If
        `prefix_id` is supplied, also reads the matching prefix segment
        from the request and computes prefix deltas against the stored
        prefix base. Otherwise, leaves prefix deltas zero-initialized
        (useful when the template has no static prefix between
        placeholders).

        Records:
            placeholder_*_embedding = real placeholder K/V
            placeholder_*_delta     = real - placeholder_base
            prefix_*_delta          = real - prefix_base   (or 0)
        """
        if not self.manager.has_pool(chunk_hash):
            raise KeyError(
                f"capture_anchor: pool for chunk_hash={chunk_hash[:12]!r} "
                "is not registered; call capture_placeholder_base first"
            )
        pool = self.manager.pool(chunk_hash)
        base = pool.base()
        if base is None:
            raise RuntimeError(
                f"capture_anchor: pool for chunk_hash={chunk_hash[:12]!r} "
                "has no base; capture_placeholder_base first"
            )
        K_base, V_base = base

        # Optional prefix.
        prefix_seg: PrefixSegment | None = None
        if prefix_id is not None:
            prefix_seg = self.prefix_store.get(prefix_id)
            if prefix_seg is None:
                raise KeyError(
                    f"capture_anchor: no prefix segment for "
                    f"prefix_id={prefix_id!r}; call capture_prefix_base first"
                )
            if pf_num_tokens is None or pf_t_start is None:
                raise ValueError(
                    "capture_anchor: prefix_id supplied without "
                    "pf_t_start / pf_num_tokens"
                )
            if prefix_seg.num_tokens != int(pf_num_tokens):
                raise ValueError(
                    f"capture_anchor: prefix segment {prefix_id!r} has "
                    f"{prefix_seg.num_tokens} tokens but caller passed "
                    f"pf_num_tokens={pf_num_tokens}"
                )

        # Real placeholder KV from the current request.
        ph_K, ph_V = self.gather_chunk_kv(req_id, ph_t_start, pool.num_tokens)
        if derotate:
            ph_K = self.derotate_to_position_zero(ph_K, ph_t_start)

        entry = pool.add_entry(anchor_id)
        entry.placeholder_key_embedding.copy_(ph_K)
        entry.placeholder_value_embedding.copy_(ph_V)
        entry.placeholder_key_delta.copy_(ph_K - K_base)
        entry.placeholder_value_delta.copy_(ph_V - V_base)

        if prefix_seg is not None:
            pf_K, pf_V = self.gather_chunk_kv(
                req_id, int(pf_t_start), int(pf_num_tokens)  # type: ignore[arg-type]
            )
            if derotate:
                pf_K = self.derotate_to_position_zero(pf_K, int(pf_t_start))  # type: ignore[arg-type]
            entry.prefix_key_delta.copy_(pf_K - prefix_seg.K_base)
            entry.prefix_value_delta.copy_(pf_V - prefix_seg.V_base)
        # else: prefix_*_delta stays zero (allocated as zeros).

        return {
            "chunk_hash": chunk_hash,
            "anchor_id": anchor_id,
            "ph_num_tokens": pool.num_tokens,
            "pf_num_tokens": int(pf_num_tokens) if pf_num_tokens else 0,
            "has_prefix": prefix_seg is not None,
        }

    # ------------------------------------------------------------------
    # Anchor blending
    # ------------------------------------------------------------------
    def blend_anchors(
        self,
        pool: AnchorPool,
        real_ph_K: torch.Tensor,
        real_ph_V: torch.Tensor,
        base_pf_K: torch.Tensor,
        base_pf_V: torch.Tensor,
        *,
        temperature: float = 1.0,
        skip_layer_zero: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Weighted-sum anchor blending. Returns four corrected tensors:
        `(corrected_ph_K, corrected_ph_V, corrected_pf_K, corrected_pf_V)`.

        All inputs and outputs share shape
        `(num_layers, num_tokens, num_kv_heads, head_size)` matching the
        owning pool. `real_ph_*` is the placeholder KV gathered from the
        current request after dense prefill. `base_pf_*` is the prefix
        segment's stored base, already re-rotated for the target
        position (caller's responsibility).

        Math, per anchor `i`, layer `L`, head `H`:

            sim_K[i,L,H] = ‖real_ph_K[L,:,H,:] − anchor_i.ph_K_emb[L,:,H,:]‖₂
            sim_V[i,L,H] = ‖real_ph_V[L,:,H,:] − anchor_i.ph_V_emb[L,:,H,:]‖₂
            w_K[i,L,H]   = softmax_i(−sim_K / temperature)
            w_V[i,L,H]   = softmax_i(−sim_V / temperature)

        Corrections:
            corrected_ph_K = real_ph_K + Σᵢ w_K[i] · anchor_i.ph_K_delta
            corrected_ph_V = real_ph_V + Σᵢ w_V[i] · anchor_i.ph_V_delta
            corrected_pf_K = base_pf_K + Σᵢ w_K[i] · anchor_i.pf_K_delta
            corrected_pf_V = base_pf_V + Σᵢ w_V[i] · anchor_i.pf_V_delta

        If `skip_layer_zero=True`, layer 0 of the four outputs is left as
        the corresponding input (treats the embedding layer's KV as
        ground truth, matching the reference flow).

        If the pool has no anchors, the inputs are returned unchanged.
        """
        anchor_ids = pool.ids()
        if not anchor_ids:
            return real_ph_K, real_ph_V, base_pf_K, base_pf_V

        anchors: list[AnchorEntry] = []
        for aid in anchor_ids:
            entry = pool.get_entry(aid)
            if entry is not None:
                anchors.append(entry)
        if not anchors:
            return real_ph_K, real_ph_V, base_pf_K, base_pf_V

        # Stack into (N, num_layers, num_tokens, num_kv_heads, head_size).
        # Anchor tensors live on the same device as real_*; safe to stack.
        ph_K_emb_stack = torch.stack([a.placeholder_key_embedding for a in anchors])
        ph_V_emb_stack = torch.stack([a.placeholder_value_embedding for a in anchors])
        ph_K_delta_stack = torch.stack([a.placeholder_key_delta for a in anchors])
        ph_V_delta_stack = torch.stack([a.placeholder_value_delta for a in anchors])
        pf_K_delta_stack = torch.stack([a.prefix_key_delta for a in anchors])
        pf_V_delta_stack = torch.stack([a.prefix_value_delta for a in anchors])

        # L2 over (token, head_size) per (anchor, layer, head).
        # diff: (N, L, T, H, D); pow + sum over dims (T, D); sqrt
        diff_K = real_ph_K.unsqueeze(0) - ph_K_emb_stack
        diff_V = real_ph_V.unsqueeze(0) - ph_V_emb_stack
        sim_K = diff_K.pow(2).sum(dim=(2, 4)).clamp_min(0).sqrt()  # (N, L, H)
        sim_V = diff_V.pow(2).sum(dim=(2, 4)).clamp_min(0).sqrt()  # (N, L, H)

        w_K = torch.softmax(-sim_K / float(temperature), dim=0)  # (N, L, H)
        w_V = torch.softmax(-sim_V / float(temperature), dim=0)

        # Broadcast (N, L, H) → (N, L, 1, H, 1) for elementwise multiply
        # against (N, L, T, H, D).
        w_K_bcast = w_K.unsqueeze(2).unsqueeze(-1)
        w_V_bcast = w_V.unsqueeze(2).unsqueeze(-1)

        ph_K_corr = real_ph_K + (w_K_bcast * ph_K_delta_stack).sum(dim=0)
        ph_V_corr = real_ph_V + (w_V_bcast * ph_V_delta_stack).sum(dim=0)
        pf_K_corr = base_pf_K + (w_K_bcast * pf_K_delta_stack).sum(dim=0)
        pf_V_corr = base_pf_V + (w_V_bcast * pf_V_delta_stack).sum(dim=0)

        if skip_layer_zero:
            ph_K_corr[0] = real_ph_K[0]
            ph_V_corr[0] = real_ph_V[0]
            pf_K_corr[0] = base_pf_K[0]
            pf_V_corr[0] = base_pf_V[0]

        return ph_K_corr, ph_V_corr, pf_K_corr, pf_V_corr

    # ------------------------------------------------------------------
    # Entropy-gated admission (`predict_as_anchor`)
    # ------------------------------------------------------------------
    def predict_as_anchor(
        self,
        pool: AnchorPool,
        candidate_V: torch.Tensor,
        *,
        threshold: float = 0.3,
        top_p: float = 0.9,
        bump: bool = True,
    ) -> dict[str, Any]:
        """Decide whether `candidate_V` should be promoted to a new anchor.

        `candidate_V` is the placeholder *value* tensor for the current
        request's chunk, shape `(num_layers, num_tokens, num_kv_heads,
        head_size)`. V is position-independent under standard RoPE so it
        can be compared directly against stored
        `placeholder_value_embedding`s without de-rotation.

        Math:
            d_i      = ‖candidate_V − anchor_i.placeholder_value_embedding‖₂
                       (Frobenius norm over all four dims)
            sim_i    = softmax_i(−d_i)
            entropy  = −Σ sim_i · log2(sim_i + 1e-40)
            max_ent  = log2(N)
            admit    = entropy > threshold * max_ent

        Returns:
            {
              "admit": bool,
              "n_anchors": int,
              "entropy": float,
              "max_entropy": float,
              "activated_anchor_ids": [...],   # empty if admit=True
            }

        Side effect: when `bump=True` and `admit=False`, increments
        `activation_count` on each id in `activated_anchor_ids` (the
        top-p cumulative subset of the similarity distribution).
        """
        anchor_ids = pool.ids()
        if not anchor_ids:
            # No anchors yet: must admit the first one.
            return {
                "admit": True,
                "n_anchors": 0,
                "entropy": 0.0,
                "max_entropy": 0.0,
                "activated_anchor_ids": [],
            }

        anchors: list[tuple[str, AnchorEntry]] = []
        for aid in anchor_ids:
            entry = pool.get_entry(aid)
            if entry is not None:
                anchors.append((aid, entry))
        n = len(anchors)
        if n == 0:
            return {
                "admit": True,
                "n_anchors": 0,
                "entropy": 0.0,
                "max_entropy": 0.0,
                "activated_anchor_ids": [],
            }

        v_emb_stack = torch.stack(
            [e.placeholder_value_embedding for _aid, e in anchors]
        )
        # Frobenius norm over the four tensor dims (layer, token, head, head_size).
        diff = candidate_V.unsqueeze(0) - v_emb_stack  # (N, L, T, H, D)
        d = diff.reshape(n, -1).norm(p=2, dim=1)  # (N,)

        sim = torch.softmax(-d, dim=0)  # (N,)
        entropy_t = -(sim * torch.log2(sim.clamp_min(1e-40))).sum()
        entropy = float(entropy_t.item())

        # Special case: a single anchor's max_entropy is 0, so any
        # threshold > 0 of 0 is unreachable; we never admit but we do
        # activate that one anchor.
        max_entropy = math.log2(n) if n > 1 else 0.0
        admit = bool(entropy > float(threshold) * max_entropy)

        activated_ids: list[str] = []
        if not admit:
            # Top-p cumulative subset over the sim distribution.
            # Sort sim descending, take ids whose cumulative sum < top_p,
            # plus the next one (so the subset's mass is >= top_p).
            order = torch.argsort(sim, descending=True)
            cumsum = torch.cumsum(sim[order], dim=0)
            # Number of items needed for cumulative >= top_p.
            mask = cumsum < float(top_p)
            k = int(mask.sum().item()) + 1
            k = min(k, n)
            for idx in order[:k].tolist():
                activated_ids.append(anchors[idx][0])
            if bump and activated_ids:
                for aid in activated_ids:
                    pool.bump_activation(aid)

        return {
            "admit": admit,
            "n_anchors": n,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "activated_anchor_ids": activated_ids,
        }

    def predict_as_anchor_for_request(
        self,
        req_id: str,
        chunk_hash: str,
        ph_t_start: int,
        *,
        threshold: float = 0.3,
        top_p: float = 0.9,
        bump: bool = True,
    ) -> dict[str, Any]:
        """RPC-friendly wrapper: gather the candidate V from the live
        cache and run `predict_as_anchor` on it."""
        if not self.manager.has_pool(chunk_hash):
            return {
                "admit": True,
                "reason": "no pool registered (first observation)",
                "n_anchors": 0,
                "entropy": 0.0,
                "max_entropy": 0.0,
                "activated_anchor_ids": [],
            }
        pool = self.manager.pool(chunk_hash)
        # V is position-independent; gather without de-rotation.
        _K, V = self.gather_chunk_kv(req_id, ph_t_start, pool.num_tokens)
        return self.predict_as_anchor(
            pool,
            V,
            threshold=float(threshold),
            top_p=float(top_p),
            bump=bool(bump),
        )

    # ------------------------------------------------------------------
    # Naive reuse path (no blending) — kept for parity tests.
    # ------------------------------------------------------------------
    def reuse(
        self,
        req_id: str,
        chunk_hash: str,
        anchor_id: str,
        t_start: int,
        rerotate: bool = True,
    ) -> dict[str, Any]:
        """Naive reconstruction: `K_base + ph_delta + pf_delta` re-rotated
        and scattered. This path does NOT blend over multiple anchors —
        use `blend_anchors` for that. Kept as a sanity-check primitive."""
        if not self.manager.has_pool(chunk_hash):
            return {"kind": "miss", "reason": "no pool for chunk_hash"}
        pool = self.manager.pool(chunk_hash)
        base = pool.base()
        if base is None:
            return {"kind": "miss", "reason": "pool has no base yet"}
        entry = pool.get_entry(anchor_id)
        if entry is None:
            return {"kind": "miss", "reason": "no entry for anchor_id"}

        K_base, V_base = base
        K = K_base + entry.placeholder_key_delta + entry.prefix_key_delta
        V = V_base + entry.placeholder_value_delta + entry.prefix_value_delta

        if rerotate:
            K = self.rerotate_from_position_zero(K, t_start)

        self.scatter_chunk_kv(req_id, t_start, K, V)
        return {
            "kind": "ok",
            "chunk_hash": chunk_hash,
            "anchor_id": anchor_id,
            "num_tokens": pool.num_tokens,
        }


# ---------------------------------------------------------------------------
# Singleton attachment (mirrors worker_ext._attach_state)
# ---------------------------------------------------------------------------

_RUNNER_STATE_ATTR = "_anchor_pool_state"

# Module-level reference for cross-component access (e.g. from the
# AnchorPoolConnector that lives in `vllm.distributed.kv_transfer`).
# Both the worker extension and the connector live in the same worker
# process, so a module-level singleton is the simplest bridge.
_ACTIVE_STATE: _AnchorPoolRunnerState | None = None


def _attach_state(worker_self: Any) -> _AnchorPoolRunnerState:
    global _ACTIVE_STATE
    runner = worker_self.model_runner
    state = getattr(runner, _RUNNER_STATE_ATTR, None)
    if state is None:
        state = _AnchorPoolRunnerState(runner)
        setattr(runner, _RUNNER_STATE_ATTR, state)
        _ACTIVE_STATE = state
    return state


def get_active_anchor_pool_state() -> _AnchorPoolRunnerState | None:
    """Public accessor used by the AnchorPoolConnector worker side."""
    return _ACTIVE_STATE


# ---------------------------------------------------------------------------
# Public worker extension class
# ---------------------------------------------------------------------------


class AnchorPoolWorkerExtension:
    """Mixed into the worker via `--worker-extension-cls`.

    Provides RPCs callable from the API process / engine via
    `engine_client.collective_rpc`. All RPCs return JSON-friendly
    dictionaries so they survive serialization.
    """

    # -- Lifecycle ------------------------------------------------------
    def anchor_pool_install(self) -> dict[str, Any]:
        """Force-install runner state and return model/cache dims so the
        caller can construct compatible tensors if needed."""
        state = _attach_state(self)
        return {
            "block_size": state.block_size,
            "num_layers": state.num_layers,
            "num_kv_heads": state.num_kv_heads,
            "head_size": state.head_size,
            "kv_dtype": str(state.kv_dtype),
            "device": str(state.device),
        }

    # -- Pool registration ---------------------------------------------
    def anchor_pool_register(
        self, chunk_hash: str, num_tokens: int, max_entries: int = 20
    ) -> dict[str, Any]:
        state = _attach_state(self)
        pool = state.manager.register_pool(
            chunk_hash,
            num_tokens=num_tokens,
            num_layers=state.num_layers,
            num_kv_heads=state.num_kv_heads,
            head_size=state.head_size,
            dtype=state.kv_dtype,
            device=state.device,
            max_entries=max_entries,
        )
        return {
            "chunk_hash": chunk_hash,
            "num_tokens": pool.num_tokens,
            "has_base": pool.has_base,
            "num_entries": len(pool),
        }

    def anchor_pool_drop_pool(self, chunk_hash: str) -> dict[str, Any]:
        state = _attach_state(self)
        return {"dropped": state.manager.drop_pool(chunk_hash)}

    def anchor_pool_drop_entry(
        self, chunk_hash: str, anchor_id: str
    ) -> dict[str, Any]:
        state = _attach_state(self)
        if not state.manager.has_pool(chunk_hash):
            return {"dropped": False, "reason": "no pool"}
        return {"dropped": state.manager.pool(chunk_hash).remove_entry(anchor_id)}

    # -- Capture (three primitives) ------------------------------------
    def anchor_pool_capture_placeholder_base(
        self,
        req_id: str,
        chunk_hash: str,
        t_start: int,
        num_tokens: int,
        derotate: bool = True,
    ) -> dict[str, Any]:
        state = _attach_state(self)
        return state.capture_placeholder_base(
            req_id=req_id,
            chunk_hash=chunk_hash,
            t_start=int(t_start),
            num_tokens=int(num_tokens),
            derotate=bool(derotate),
        )

    def anchor_pool_capture_prefix_base(
        self,
        req_id: str,
        prefix_id: str,
        t_start: int,
        num_tokens: int,
        derotate: bool = True,
    ) -> dict[str, Any]:
        state = _attach_state(self)
        return state.capture_prefix_base(
            req_id=req_id,
            prefix_id=prefix_id,
            t_start=int(t_start),
            num_tokens=int(num_tokens),
            derotate=bool(derotate),
        )

    def anchor_pool_capture_anchor(
        self,
        req_id: str,
        chunk_hash: str,
        anchor_id: str,
        ph_t_start: int,
        prefix_id: str,
        pf_t_start: int,
        pf_num_tokens: int,
        derotate: bool = True,
    ) -> dict[str, Any]:
        state = _attach_state(self)
        return state.capture_anchor(
            req_id=req_id,
            chunk_hash=chunk_hash,
            anchor_id=anchor_id,
            ph_t_start=int(ph_t_start),
            prefix_id=prefix_id,
            pf_t_start=int(pf_t_start),
            pf_num_tokens=int(pf_num_tokens),
            derotate=bool(derotate),
        )

    # -- Naive reuse (kept for parity tests; does NOT blend) -----------
    def anchor_pool_reuse(
        self,
        req_id: str,
        chunk_hash: str,
        anchor_id: str,
        t_start: int,
        rerotate: bool = True,
    ) -> dict[str, Any]:
        state = _attach_state(self)
        return state.reuse(
            req_id=req_id,
            chunk_hash=chunk_hash,
            anchor_id=anchor_id,
            t_start=int(t_start),
            rerotate=bool(rerotate),
        )

    # -- Prefix segment maintenance ------------------------------------
    def anchor_pool_drop_prefix(self, prefix_id: str) -> dict[str, Any]:
        state = _attach_state(self)
        return {"dropped": state.prefix_store.remove(prefix_id)}

    # -- Entropy-gated admission --------------------------------------
    def anchor_pool_predict_as_anchor(
        self,
        req_id: str,
        chunk_hash: str,
        ph_t_start: int,
        threshold: float = 0.3,
        top_p: float = 0.9,
        bump: bool = True,
    ) -> dict[str, Any]:
        state = _attach_state(self)
        return state.predict_as_anchor_for_request(
            req_id=req_id,
            chunk_hash=chunk_hash,
            ph_t_start=int(ph_t_start),
            threshold=float(threshold),
            top_p=float(top_p),
            bump=bool(bump),
        )

    def anchor_pool_bump_activation(
        self, chunk_hash: str, anchor_id: str, by: int = 1
    ) -> dict[str, Any]:
        state = _attach_state(self)
        if not state.manager.has_pool(chunk_hash):
            return {"updated": None, "reason": "no pool"}
        new_count = state.manager.pool(chunk_hash).bump_activation(
            anchor_id, by=int(by)
        )
        return {"updated": new_count}

    # -- Introspection -------------------------------------------------
    def anchor_pool_status(self) -> dict[str, Any]:
        state = _attach_state(self)
        pools = {}
        for h in state.manager.chunk_hashes():
            p = state.manager.pool(h)
            pools[h] = {
                "num_tokens": p.num_tokens,
                "has_base": p.has_base,
                "num_entries": len(p),
                "max_entries": p.max_entries,
                "anchor_ids": p.ids(),
                "activation_counts": p.activation_counts(),
            }
        return {
            "num_pools": len(pools),
            "pools": pools,
            "prefix_segments": {
                pid: {"num_tokens": state.prefix_store.get(pid).num_tokens}  # type: ignore[union-attr]
                for pid in state.prefix_store.ids()
            },
            "block_size": state.block_size,
        }
