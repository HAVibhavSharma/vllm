# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Anchor KV pool — worker-side GPU memory keyed by static-chunk hash.

A *static chunk* is a piece of prompt content that recurs across many
requests. Each static chunk gets its own `AnchorPool`, identified by a
content-stable hash (typically `sha256(chunk_text + tokenizer + model)`).

Per-pool storage layout:

    base  ──  one (K_base, V_base) pair, shape
              (num_layers, num_tokens, num_kv_heads, head_size).
              Captured once, the first time the chunk is seen at canonical
              position 0 (i.e. with RoPE de-rotated). Shared by every
              entry below — the bulk of memory savings comes from this.

    entries ─ dict of `AnchorEntry`, capacity-capped (default 20).
              Each entry is six tensors (two embeddings + four deltas)
              describing how the chunk's KV diverged from `base` for one
              specific surrounding context. Eviction policy is LFU on
              `activation_count` (least-frequently-activated first; ties
              broken by oldest insertion). The worker extension's
              entropy-gated `predict_as_anchor` is responsible for
              calling `bump_activation` on anchors that explained a
              candidate.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class AnchorEntry:
    """Six tensors describing one past observation of the placeholder
    under a specific surrounding context.

    All tensors have shape `(num_layers, num_tokens, num_kv_heads,
    head_size)`, matching the owning pool's dims exactly.

    Roles:
        placeholder_key_embedding   — real K observed for this placeholder
                                       in this context. Used as the
                                       similarity key during blending.
        placeholder_value_embedding — real V observed (similarity key).
        placeholder_key_delta       — placeholder real_K − base_K.
        placeholder_value_delta     — placeholder real_V − base_V.
        prefix_key_delta            — prefix real_K − prefix_base_K.
        prefix_value_delta          — prefix real_V − prefix_base_V.

    During reuse: anchors are weighted by L2-distance softmax between the
    current request's `real_ph_kv` and each anchor's
    `placeholder_*_embedding`. The weighted sum of deltas is then added
    to the current real placeholder K/V (and to the prefix base K/V) to
    approximate the dense-prefill answer.
    """

    placeholder_key_embedding: torch.Tensor
    placeholder_value_embedding: torch.Tensor
    placeholder_key_delta: torch.Tensor
    placeholder_value_delta: torch.Tensor
    prefix_key_delta: torch.Tensor
    prefix_value_delta: torch.Tensor

    # Bumped by `bump_activation` when this anchor was among the top-p
    # nearest neighbours for a non-admitted candidate. Used for LFU
    # eviction so frequently-useful anchors stay in the pool.
    activation_count: int = field(default=0)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.placeholder_key_embedding,
            self.placeholder_value_embedding,
            self.placeholder_key_delta,
            self.placeholder_value_delta,
            self.prefix_key_delta,
            self.prefix_value_delta,
        )

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


class AnchorPool:
    """One pool per static chunk.

    Args:
        chunk_hash:   content-addressable identity of the static chunk.
        num_tokens:   tokens in the chunk; LOCKED for the life of the pool.
        num_layers:   model layer count.
        num_kv_heads: KV-head count.
        head_size:    per-head dim.
        dtype:        model KV dtype.
        device:       worker GPU device.
        max_entries:  LRU cap on `AnchorEntry` count. Default 20.
    """

    def __init__(
        self,
        chunk_hash: str,
        num_tokens: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        max_entries: int = 20,
    ) -> None:
        if num_tokens <= 0:
            raise ValueError(f"AnchorPool: num_tokens must be > 0, got {num_tokens}")
        if num_layers <= 0 or num_kv_heads <= 0 or head_size <= 0:
            raise ValueError(
                f"AnchorPool: invalid dims num_layers={num_layers} "
                f"num_kv_heads={num_kv_heads} head_size={head_size}"
            )
        if max_entries <= 0:
            raise ValueError(f"AnchorPool: max_entries must be > 0, got {max_entries}")

        self.chunk_hash = chunk_hash
        self.num_tokens = int(num_tokens)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_size = int(head_size)
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_entries = int(max_entries)

        self._base_k: torch.Tensor | None = None
        self._base_v: torch.Tensor | None = None
        self._entries: OrderedDict[str, AnchorEntry] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Base (one shared (K, V) per pool)
    # ------------------------------------------------------------------
    @property
    def has_base(self) -> bool:
        with self._lock:
            return self._base_k is not None

    def base(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        with self._lock:
            if self._base_k is None or self._base_v is None:
                return None
            return self._base_k, self._base_v

    def set_base(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Set the canonical-position-0 base K/V for this chunk.

        Tensors must match the pool's expected shape and dtype/device
        exactly. Set-once: a second call raises `RuntimeError`.
        """
        expected = (
            self.num_layers,
            self.num_tokens,
            self.num_kv_heads,
            self.head_size,
        )
        for name, t in (("key", key), ("value", value)):
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"AnchorPool.set_base: {name} shape {tuple(t.shape)} != "
                    f"expected {expected}"
                )
            if t.dtype != self.dtype:
                raise ValueError(
                    f"AnchorPool.set_base: {name} dtype {t.dtype} != "
                    f"pool dtype {self.dtype}"
                )
            if t.device != self.device:
                raise ValueError(
                    f"AnchorPool.set_base: {name} on {t.device}, "
                    f"pool on {self.device}"
                )

        with self._lock:
            if self._base_k is not None:
                raise RuntimeError(
                    f"AnchorPool[{self.chunk_hash[:12]}]: base already set"
                )
            self._base_k = key
            self._base_v = value

        logger.info(
            "[anchor-pool] set_base chunk=%s tokens=%d bytes=%d",
            self.chunk_hash[:12],
            self.num_tokens,
            (key.numel() * key.element_size()) * 2,
        )

    # ------------------------------------------------------------------
    # Entry CRUD with LRU
    # ------------------------------------------------------------------
    def add_entry(self, anchor_id: str) -> AnchorEntry:
        """Allocate a new entry. If at cap, evict the least-activated
        entry first (LFU; ties broken by oldest insertion).

        Raises `KeyError` if `anchor_id` already exists.
        """
        evicted_id: str | None = None
        with self._lock:
            if anchor_id in self._entries:
                raise KeyError(
                    f"AnchorPool[{self.chunk_hash[:12]}].add_entry: "
                    f"id {anchor_id!r} already exists"
                )
            if len(self._entries) >= self.max_entries:
                evicted_id = self._pick_lfu_locked()
                _evicted = self._entries.pop(evicted_id)
                del _evicted

            shape = (
                self.num_layers,
                self.num_tokens,
                self.num_kv_heads,
                self.head_size,
            )

            def _alloc() -> torch.Tensor:
                return torch.zeros(shape, dtype=self.dtype, device=self.device)

            entry = AnchorEntry(
                placeholder_key_embedding=_alloc(),
                placeholder_value_embedding=_alloc(),
                placeholder_key_delta=_alloc(),
                placeholder_value_delta=_alloc(),
                prefix_key_delta=_alloc(),
                prefix_value_delta=_alloc(),
            )
            self._entries[anchor_id] = entry

        if evicted_id is not None:
            logger.info(
                "[anchor-pool] LRU evict chunk=%s anchor=%s for new=%s",
                self.chunk_hash[:12],
                evicted_id,
                anchor_id,
            )
        logger.info(
            "[anchor-pool] add_entry chunk=%s anchor=%s (%d/%d) bytes=%d",
            self.chunk_hash[:12],
            anchor_id,
            len(self._entries),
            self.max_entries,
            entry.nbytes(),
        )
        return entry

    def get_entry(self, anchor_id: str) -> AnchorEntry | None:
        # No move-to-end: eviction is LFU on activation_count, not LRU.
        with self._lock:
            return self._entries.get(anchor_id)

    def contains(self, anchor_id: str) -> bool:
        with self._lock:
            return anchor_id in self._entries

    def bump_activation(self, anchor_id: str, by: int = 1) -> int | None:
        """Increment an anchor's activation count. Called by the worker
        extension when an anchor was selected by top-p over similarity.

        Returns the new count, or None if `anchor_id` is not present.
        """
        with self._lock:
            entry = self._entries.get(anchor_id)
            if entry is None:
                return None
            entry.activation_count += int(by)
            return entry.activation_count

    def _pick_lfu_locked(self) -> str:
        """Return the id of the least-activated entry. Ties are broken
        by insertion order (oldest first). Caller must hold the lock."""
        # OrderedDict preserves insertion order; min() with a stable key
        # picks the first id at the minimum count, which is the oldest.
        return min(
            self._entries.items(),
            key=lambda kv: kv[1].activation_count,
        )[0]

    def activation_counts(self) -> dict[str, int]:
        with self._lock:
            return {aid: e.activation_count for aid, e in self._entries.items()}

    def remove_entry(self, anchor_id: str) -> bool:
        with self._lock:
            entry = self._entries.pop(anchor_id, None)
        if entry is None:
            return False
        del entry
        logger.info(
            "[anchor-pool] remove_entry chunk=%s anchor=%s (%d/%d)",
            self.chunk_hash[:12],
            anchor_id,
            len(self._entries),
            self.max_entries,
        )
        return True

    def clear(self) -> None:
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            self._base_k = None
            self._base_v = None
        if n:
            logger.info(
                "[anchor-pool] cleared chunk=%s (%d entries + base)",
                self.chunk_hash[:12],
                n,
            )

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class AnchorPoolManager:
    """Engine-side façade owning multiple `AnchorPool`s — one per static
    chunk seen by this worker.

    Pools are created lazily via `register_pool` and looked up by
    `chunk_hash`. Reading `pool(chunk_hash)` for an unregistered chunk
    raises `KeyError` rather than silently allocating, since `num_tokens`
    is locked at pool creation time and we need the caller to commit to
    a specific chunk-token-count up front.
    """

    def __init__(self) -> None:
        self._pools: dict[str, AnchorPool] = {}
        self._lock = threading.Lock()

    def register_pool(
        self,
        chunk_hash: str,
        *,
        num_tokens: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        max_entries: int = 20,
    ) -> AnchorPool:
        """Idempotent. If a pool already exists for `chunk_hash`, returns
        it (after validating `num_tokens` matches)."""
        with self._lock:
            existing = self._pools.get(chunk_hash)
            if existing is not None:
                if existing.num_tokens != num_tokens:
                    raise ValueError(
                        f"AnchorPoolManager.register_pool: chunk={chunk_hash[:12]} "
                        f"already registered with num_tokens={existing.num_tokens}, "
                        f"caller asked for {num_tokens}"
                    )
                return existing
            pool = AnchorPool(
                chunk_hash=chunk_hash,
                num_tokens=num_tokens,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                device=device,
                max_entries=max_entries,
            )
            self._pools[chunk_hash] = pool
        logger.info(
            "[anchor-pool] register_pool chunk=%s tokens=%d cap=%d",
            chunk_hash[:12],
            num_tokens,
            max_entries,
        )
        return pool

    def pool(self, chunk_hash: str) -> AnchorPool:
        with self._lock:
            pool = self._pools.get(chunk_hash)
        if pool is None:
            raise KeyError(
                f"AnchorPoolManager.pool: chunk={chunk_hash[:12]!r} not "
                f"registered; call register_pool() first"
            )
        return pool

    def has_pool(self, chunk_hash: str) -> bool:
        with self._lock:
            return chunk_hash in self._pools

    def drop_pool(self, chunk_hash: str) -> bool:
        with self._lock:
            pool = self._pools.pop(chunk_hash, None)
        if pool is None:
            return False
        pool.clear()
        logger.info("[anchor-pool] drop_pool chunk=%s", chunk_hash[:12])
        return True

    def chunk_hashes(self) -> list[str]:
        with self._lock:
            return list(self._pools.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._pools)


# ---------------------------------------------------------------------------
# Prefix segment store
# ---------------------------------------------------------------------------


@dataclass
class PrefixSegment:
    """A static prefix segment held at canonical position 0 (de-rotated).

    Shape: `(num_layers, num_tokens, num_kv_heads, head_size)` — matches
    the anchor-pool tensor layout so the same gather/scatter helpers
    work for both.
    """

    K_base: torch.Tensor
    V_base: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return int(self.K_base.shape[1])

    def nbytes(self) -> int:
        return (
            self.K_base.numel() * self.K_base.element_size()
            + self.V_base.numel() * self.V_base.element_size()
        )


class PrefixSegmentStore:
    """Per-worker store of static prefix segments.

    Indexed by `prefix_id` — typically a string derived from the
    template position (e.g. `"tpl-greeting:seg-0"`). Set-once per id;
    re-storing the same id raises unless the caller explicitly drops
    first. Unlike the anchor pool there is no per-id cap and no LRU —
    prefix segments are part of the template's static surface.
    """

    def __init__(self) -> None:
        self._segments: dict[str, PrefixSegment] = {}
        self._lock = threading.Lock()

    def set(
        self, prefix_id: str, K_base: torch.Tensor, V_base: torch.Tensor
    ) -> PrefixSegment:
        if K_base.shape != V_base.shape:
            raise ValueError(
                f"PrefixSegmentStore.set: K shape {tuple(K_base.shape)} != "
                f"V shape {tuple(V_base.shape)}"
            )
        if K_base.ndim != 4:
            raise ValueError(
                f"PrefixSegmentStore.set: expected 4-D tensor "
                f"(num_layers, num_tokens, num_kv_heads, head_size); "
                f"got {tuple(K_base.shape)}"
            )
        seg = PrefixSegment(K_base=K_base, V_base=V_base)
        with self._lock:
            if prefix_id in self._segments:
                raise KeyError(
                    f"PrefixSegmentStore.set: prefix_id {prefix_id!r} already "
                    "exists; remove it first"
                )
            self._segments[prefix_id] = seg
        logger.info(
            "[anchor-pool] prefix-segment set id=%s tokens=%d bytes=%d",
            prefix_id,
            seg.num_tokens,
            seg.nbytes(),
        )
        return seg

    def get(self, prefix_id: str) -> PrefixSegment | None:
        with self._lock:
            return self._segments.get(prefix_id)

    def has(self, prefix_id: str) -> bool:
        with self._lock:
            return prefix_id in self._segments

    def remove(self, prefix_id: str) -> bool:
        with self._lock:
            seg = self._segments.pop(prefix_id, None)
        if seg is None:
            return False
        del seg
        logger.info("[anchor-pool] prefix-segment remove id=%s", prefix_id)
        return True

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._segments.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._segments)
