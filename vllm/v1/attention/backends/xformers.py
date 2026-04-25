# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with xFormers.

This is a v1-compatible port of the xFormers backend that supports the
CacheBlend / KVLink hooks used by the `epic` research stack
(`cache_fuse_metadata`, `status`, `old_kv`).

Standard prefill and decode paths are routed through the Triton kernels
already used by `TritonAttentionBackend`, so this backend works end-to-end
with the v1 scheduler / KV cache manager. The xFormers-specific path is
only taken during prefill when `layer.cache_fuse_metadata` is active; in
that case we fall back to `xformers.ops.memory_efficient_attention_forward`
with a custom `BlockDiagonalCausalMask` / partial-bias so we can faithfully
reproduce CacheBlend and KVLink.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class XFormersAttentionMetadata:
    """Metadata required by the xFormers v1 backend.

    Mirrors the fields needed by triton-style paged attention kernels
    so the normal prefill/decode path just works.
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    query_start_loc_cpu: torch.Tensor | None
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class XFormersMetadataBuilder(AttentionMetadataBuilder[XFormersAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XFormersAttentionMetadata:
        return XFormersAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
        )


class XFormersBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "XFORMERS"

    @staticmethod
    def get_impl_cls() -> type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def get_builder_cls() -> type["XFormersMetadataBuilder"]:
        return XFormersMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # xformers memory_efficient_attention requires head_size % 8 == 0.
        return head_size % 8 == 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class XFormersImpl(AttentionImpl):
    """Attention implementation that ports the epic CacheBlend / KVLink hooks.

    Normal prefill and decode paths are handled by the same Triton kernels
    used by :class:`TritonAttentionImpl`. The ``cache_fuse_metadata`` /
    ``status`` / ``old_kv`` fields (optionally attached to the attention
    *layer* before calling forward) steer the CacheBlend / KVLink code
    paths implemented here.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **extra: object,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = (
            torch.tensor(alibi_slopes, dtype=torch.float32)
            if alibi_slopes is not None
            else None
        )
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.fp8_dtype = current_platform.fp8_dtype()

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "XFormers v1 backend does not support quantized KV cache."
            )

        if logits_soft_cap:
            logger.warning_once(
                "XFormers backend does not apply a logits soft cap. "
                "Outputs may differ slightly from other backends."
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with standard paged attention + CacheBlend / KVLink hooks.

        ``query``/``key``/``value``: shape [num_tokens, num_{q|kv}_heads, head_size].
        ``output``: preallocated tensor with the same shape as query.
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported by XFormersImpl."
            )

        if attn_metadata is None:
            # Dummy/profiling run.
            return output.fill_(0)

        cache_fuse_metadata = getattr(layer, "cache_fuse_metadata", None)
        status = getattr(layer, "status", None)
        old_kv = getattr(layer, "old_kv", None)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # CacheBlend collect pass: remember the raw (pre-RoPE) K/V on the
        # layer so the driver can fish them out between generate() calls.
        if (
            cache_fuse_metadata is not None
            and cache_fuse_metadata.get("collect", False)
            and key is not None
            and value is not None
        ):
            # Matches `hack_kv = [k.clone(), v.clone()]` from the epic Qwen2.
            layer.hack_kv = [
                key[:num_actual_tokens].clone(),
                value[:num_actual_tokens].clone(),
            ]

        # Fast path: no CacheBlend/KVLink state → standard paged attention.
        kvlink_active = (
            cache_fuse_metadata is not None
            and cache_fuse_metadata.get("kvlink") is not None
            and len(cache_fuse_metadata.get("kvlink") or []) > 0
        )
        if status not in (1, 2) and not kvlink_active:
            self._write_kv_cache(
                layer, key, value, kv_cache, attn_metadata.slot_mapping
            )
            self._run_standard_attention(
                layer,
                query[:num_actual_tokens],
                key[:num_actual_tokens] if key is not None else None,
                value[:num_actual_tokens] if value is not None else None,
                output[:num_actual_tokens],
                kv_cache,
                attn_metadata,
            )
            return output

        # ---- CacheBlend / KVLink path (prefill-only) -----------------------
        # We only support single-sequence prefill here, matching the epic
        # implementation which is marked `FIXME: only focus on batch size = 1`.
        assert attn_metadata.max_query_len == num_actual_tokens, (
            "CacheBlend / KVLink hooks require a single-sequence prefill; "
            "decode or multi-sequence batches must not be mixed with the "
            "CacheBlend modes."
        )
        assert old_kv is not None, (
            "CacheBlend / KVLink require `layer.old_kv` to be set with "
            "cached [key, value] tensors before the forward pass."
        )

        q = query[:num_actual_tokens]
        k = key[:num_actual_tokens] if key is not None else None
        v = value[:num_actual_tokens] if value is not None else None

        key_old = old_kv[0].view(-1, self.num_kv_heads, self.head_size)
        value_old = old_kv[1].view(-1, self.num_kv_heads, self.head_size)

        if kvlink_active:
            imp_indices = cache_fuse_metadata["kvlink"]
            cache_fuse_metadata["imp_indices"] = imp_indices
            # Splice the freshly-computed K/V for the selected indices into
            # the cached "old" K/V.
            key_old[imp_indices] = k
            value_old[imp_indices] = v
            k_full, v_full = key_old, value_old
            # kvlink only fires once per generate() call.
            cache_fuse_metadata["kvlink"] = []
            attn_bias = _make_partial_bias(
                cache_fuse_metadata,
                q.device,
                self.num_kv_heads,
                self.num_queries_per_kv,
            )
        elif status == 1:
            # Pick the top-k tokens whose value vectors diverge the most from
            # the cached values, plus the final query token.
            recomp_ratio = cache_fuse_metadata["recomp_ratio"]
            topk_num = int((v.shape[0] - 1) * recomp_ratio)
            temp_diff = torch.sum(
                (v[:-1, :, :] - value_old[:-1, :, :]) ** 2, dim=[1, 2]
            )
            top_indices = torch.topk(temp_diff, k=topk_num).indices
            top_indices = torch.cat(
                [
                    top_indices,
                    torch.tensor(
                        [q.shape[0] - 1],
                        device=top_indices.device,
                        dtype=top_indices.dtype,
                    ),
                ]
            )
            q = q[top_indices]
            cache_fuse_metadata["imp_indices"] = top_indices
            k_full, v_full = key_old, value_old
            attn_bias = _make_partial_bias(
                cache_fuse_metadata,
                q.device,
                self.num_kv_heads,
                self.num_queries_per_kv,
            )
        else:
            # status == 2: reuse the indices computed on the check layer.
            imp_indices = cache_fuse_metadata["imp_indices"]
            key_old[imp_indices] = k
            value_old[imp_indices] = v
            k_full, v_full = key_old, value_old
            attn_bias = _make_partial_bias(
                cache_fuse_metadata,
                q.device,
                self.num_kv_heads,
                self.num_queries_per_kv,
            )

        cache_fuse_metadata["kv_cache_dtype"] = v.dtype

        # Still write the *new* (freshly computed) K/V for the selected
        # indices to the paged cache, using the slot_mapping that was
        # prepared for those tokens.
        if k is not None and v is not None:
            self._write_kv_cache(
                layer, k, v, kv_cache, attn_metadata.slot_mapping[: k.shape[0]]
            )

        out = _xformers_custom_bias_attention(
            q,
            k_full,
            v_full,
            attn_bias,
            self.scale,
            self.num_heads,
            self.num_kv_heads,
            self.num_queries_per_kv,
        )
        # Output may be smaller than the pre-allocated buffer (status=1 path
        # selects a subset of queries); the caller (model decoder layer)
        # slices residuals/hidden_states the same way.
        output[: out.shape[0]].copy_(out.view(out.shape[0], self.num_heads, -1))
        return output

    # ---- helpers -----------------------------------------------------------

    def _write_kv_cache(
        self,
        layer: AttentionLayer,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if key is None or value is None or kv_cache.numel() == 0:
            return
        key_cache, value_cache = kv_cache.unbind(0)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _run_standard_attention(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: XFormersAttentionMetadata,
    ) -> None:
        if kv_cache.numel() == 0:
            # No cache yet: this is a pure prefill (no prefix caching).
            # Fall through to xformers for a plain causal attention.
            if key is None or value is None:
                return
            attn_bias = _make_block_diagonal_causal_mask(attn_metadata)
            if attn_bias is None:
                return
            out = _xformers_custom_bias_attention(
                query,
                key,
                value,
                attn_bias,
                self.scale,
                self.num_heads,
                self.num_kv_heads,
                self.num_queries_per_kv,
            )
            output.copy_(out.view_as(output))
            return

        key_cache, value_cache = kv_cache.unbind(0)

        chunked_prefill_paged_decode(
            query=query,
            key=key,
            value=value,
            output=output,
            kv_cache_dtype=self.kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            max_query_len=attn_metadata.max_query_len,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            sm_scale=self.scale,
            causal=attn_metadata.causal,
        )


def _make_block_diagonal_causal_mask(attn_metadata: XFormersAttentionMetadata):
    """Build a BlockDiagonalCausalMask from the v1 attention metadata."""
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

    seq_lens_cpu = attn_metadata.seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = attn_metadata.seq_lens.to("cpu")
    seq_lens = seq_lens_cpu.tolist()
    if not seq_lens:
        return None
    return BlockDiagonalCausalMask.from_seqlens(seq_lens)


def _xformers_custom_bias_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
) -> torch.Tensor:
    """Run xformers.memory_efficient_attention_forward with the given bias.

    Query may have fewer tokens than key/value (CacheBlend status=1 path).
    For GQA we expand K/V to the query's head count.
    """
    from xformers import ops as xops

    original_query = query
    if num_kv_heads != num_heads:
        query = query.view(
            query.shape[0], num_kv_heads, num_queries_per_kv, query.shape[-1]
        )
        key = key[:, :, None, :].expand(
            key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
        )
        value = value[:, :, None, :].expand(
            value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
        )

    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    out = xops.memory_efficient_attention_forward(
        query, key, value, attn_bias=attn_bias, p=0.0, scale=scale
    )
    return out.view_as(original_query)


def _make_partial_bias(
    cache_fuse_metadata: dict,
    device: torch.device,
    num_kv_heads: int,
    num_queries_per_kv: int,
) -> torch.Tensor:
    """Attention bias for the CacheBlend partial-recompute path.

    Mirrors _make_partial_bias_mha / _make_partial_bias_gqa from the old
    xformers backend but merged into a single function.
    """
    if num_queries_per_kv == 1:
        return _make_partial_bias_mha(cache_fuse_metadata, device, num_kv_heads)
    return _make_partial_bias_gqa(
        cache_fuse_metadata, device, num_kv_heads, num_queries_per_kv
    )


def _make_partial_bias_mha(
    cache_fuse_metadata: dict, device: torch.device, num_kv_heads: int
) -> torch.Tensor:
    """Partial causal bias for multi-head attention (batch size == 1)."""
    seq_len = cache_fuse_metadata["org_seq_len"]
    padded_len = (seq_len + 7) // 8 * 8
    dtype = cache_fuse_metadata["kv_cache_dtype"]
    imp_indices = cache_fuse_metadata["imp_indices"]
    attn_mask = torch.triu(
        torch.ones(padded_len, padded_len, dtype=dtype, device=device),
        diagonal=1,
    )
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(
        1, 1, padded_len, padded_len
    )
    attn_mask = attn_mask[:, :, imp_indices]
    attn_mask = attn_mask.expand(1, num_kv_heads, -1, -1)
    return attn_mask[:, :, :, :seq_len]


def _make_partial_bias_gqa(
    cache_fuse_metadata: dict,
    device: torch.device,
    num_kv_heads: int,
    num_queries_per_kv: int,
) -> torch.Tensor:
    """Partial causal bias for grouped-query attention (batch size == 1)."""
    seq_len = cache_fuse_metadata["org_seq_len"]
    padded_len = (seq_len + 7) // 8 * 8
    dtype = cache_fuse_metadata["kv_cache_dtype"]
    imp_indices = cache_fuse_metadata["imp_indices"]
    attn_mask = torch.triu(
        torch.ones(padded_len, padded_len, dtype=dtype, device=device),
        diagonal=1,
    )
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(
        1, 1, 1, padded_len, padded_len
    )
    attn_mask = attn_mask[:, :, :, imp_indices]
    attn_mask = attn_mask.expand(1, num_kv_heads, num_queries_per_kv, -1, -1)
    return attn_mask[:, :, :, :, :seq_len]
