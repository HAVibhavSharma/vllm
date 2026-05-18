# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.agent_prefetch.hashing import (
    DEFAULT_CHUNK_SIZE,
    chunk_align,
    compute_prefix_hash,
)


# ---------------------------------------------------------------------------
# chunk_align
# ---------------------------------------------------------------------------


def test_chunk_align_drops_partial_tail():
    tokens = list(range(35))
    aligned = chunk_align(tokens, chunk_size=16)
    assert aligned == list(range(32))


def test_chunk_align_returns_empty_when_below_chunk():
    assert chunk_align([1, 2, 3], chunk_size=16) == []


def test_chunk_align_passes_through_exact_multiple():
    tokens = list(range(48))
    assert chunk_align(tokens, chunk_size=16) == tokens


def test_chunk_align_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError):
        chunk_align([1, 2], chunk_size=0)
    with pytest.raises(ValueError):
        chunk_align([1, 2], chunk_size=-1)


def test_chunk_align_default_chunk_size_is_16():
    assert DEFAULT_CHUNK_SIZE == 16


# ---------------------------------------------------------------------------
# compute_prefix_hash
# ---------------------------------------------------------------------------


def test_compute_prefix_hash_is_deterministic():
    h1 = compute_prefix_hash("Qwen/Qwen3-8B", "salt", [1, 2, 3, 4])
    h2 = compute_prefix_hash("Qwen/Qwen3-8B", "salt", [1, 2, 3, 4])
    assert h1 == h2
    assert len(h1) == 32  # sha256


def test_compute_prefix_hash_model_isolation():
    a = compute_prefix_hash("modelA", "s", [1, 2, 3])
    b = compute_prefix_hash("modelB", "s", [1, 2, 3])
    assert a != b


def test_compute_prefix_hash_salt_isolation():
    a = compute_prefix_hash("m", "saltA", [1, 2, 3])
    b = compute_prefix_hash("m", "saltB", [1, 2, 3])
    assert a != b


def test_compute_prefix_hash_token_sensitivity():
    a = compute_prefix_hash("m", "s", [1, 2, 3])
    b = compute_prefix_hash("m", "s", [1, 2, 4])
    assert a != b


def test_compute_prefix_hash_length_framing_prevents_collision():
    # Without length-prefix framing, ("ab", "c") and ("a", "bc") could
    # collide. The framing in compute_prefix_hash prevents that across
    # the (model_name, cache_salt) boundary.
    a = compute_prefix_hash("ab", "c", [1])
    b = compute_prefix_hash("a", "bc", [1])
    assert a != b


def test_compute_prefix_hash_empty_tokens_allowed():
    # The function does not assert non-empty; chunk_align is the
    # gatekeeper at the layer above. Make sure we still produce a
    # deterministic digest for the boundary case.
    h1 = compute_prefix_hash("m", "s", [])
    h2 = compute_prefix_hash("m", "s", [])
    assert h1 == h2
    assert len(h1) == 32
