# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable prefix-hash identity for the agent prefetch registry.

The registry needs a deterministic identifier per (model, cache_salt,
token_ids) tuple to use as a dedup / lookup key. This hash is purely an
internal registry concern -- it does **not** have to match LMCache's
chunk-key derivation. When we submit a phantom prefetch we pass the raw
token ids; LMCache hashes them on its own side.

Using SHA-256 keeps the hash:

* Cross-process stable (no PYTHONHASHSEED concerns).
* Cheap (~microseconds for typical prefix lengths).
* Collision-resistant enough that we can use a truncated form for log
  lines / request-id derivation.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

DEFAULT_CHUNK_SIZE = 16
"""Default LMCache chunk size in tokens.

Mirrors the ``--chunk-size 16`` passed to ``lmcache server`` in the
reference deployment. The registry rounds recorded prefixes down to a
multiple of this size; partial chunks at the tail are dropped because
LMCache only stores complete chunks (see
``LMCacheMPRequestMetadata.GetStoreMetadata``).
"""


def chunk_align(
    token_ids: Sequence[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[int]:
    """Floor a token-id sequence to a multiple of ``chunk_size``.

    Returns an empty list when ``len(token_ids) < chunk_size`` -- there
    is no full chunk to record in that case.

    Args:
        token_ids: Prompt token ids.
        chunk_size: LMCache chunk size in tokens. Must be positive.

    Returns:
        The longest chunk-aligned prefix of ``token_ids``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    aligned_len = (len(token_ids) // chunk_size) * chunk_size
    if aligned_len == 0:
        return []
    return list(token_ids[:aligned_len])


def compute_prefix_hash(
    model_name: str,
    cache_salt: str,
    token_ids: Sequence[int],
) -> bytes:
    """Stable SHA-256 over (model_name, cache_salt, token_ids).

    The hash uses a length-prefixed framing per field so that distinct
    inputs cannot collide via string concatenation
    (``("ab", "c") != ("a", "bc")``).

    Args:
        model_name: The serving model id, e.g. ``"Qwen/Qwen3-8B"``.
            Distinct models must produce distinct hashes even for
            identical token ids -- token spaces are not portable.
        cache_salt: Per-agent / per-tenant isolation salt.
        token_ids: The (chunk-aligned) token sequence.

    Returns:
        32-byte SHA-256 digest.
    """
    h = hashlib.sha256()
    name_bytes = model_name.encode("utf-8")
    salt_bytes = cache_salt.encode("utf-8")
    h.update(struct.pack(">I", len(name_bytes)))
    h.update(name_bytes)
    h.update(struct.pack(">I", len(salt_bytes)))
    h.update(salt_bytes)
    h.update(struct.pack(">I", len(token_ids)))
    # Pack token ids as big-endian uint32 so the encoding is
    # platform-independent. vLLM vocabularies fit in 32 bits.
    h.update(struct.pack(f">{len(token_ids)}I", *token_ids))
    return h.digest()
