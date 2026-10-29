# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-store coalescing in LMCacheMPConnector (LMCACHE_MP_DECODE_STORE_CHUNKS).

`GetStoreMetadata` on a stub tracker: a prompt prefilled in one step, then one
generated token per step, the way the scheduler-side connector sees them.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    lmcache_mp_connector as mp,
)

BLOCK = 16


def _tracker(prompt: int, stored: int = 0, vllm_hits: int = 0, lmcache_hits: int = 0):
    t = SimpleNamespace(
        request_id="r",
        cache_salt="",
        num_scheduled_tokens=0,
        num_stored_blocks=stored,
        num_vllm_hit_blocks=vllm_hits,
        num_lmcache_hit_blocks=lmcache_hits,
        num_prompt_tokens=prompt,
        resumed=False,
        all_token_ids=list(range(prompt)),
        block_hashes=[],
        allocated_block_ids=[],
    )
    t.increase_num_stored_blocks = lambda k: setattr(
        t, "num_stored_blocks", t.num_stored_blocks + k
    )
    return t


def _run(prompt: int, output: int, chunks: int) -> tuple[list[tuple], SimpleNamespace]:
    """Stores as (first chunk, chunks, flush): prefill, one token a step, finish."""
    t = _tracker(prompt)
    stores: list[tuple] = []

    def record(meta):
        if meta is not None:
            span = (meta.op.end - meta.op.start) // BLOCK
            stores.append((meta.op.start // BLOCK, span, meta.flush))

    def step(new_tokens: int):
        t.num_scheduled_tokens += new_tokens
        t.block_hashes = list(range(len(t.all_token_ids) // BLOCK))
        t.allocated_block_ids = list(range(-(-t.num_scheduled_tokens // BLOCK)))
        record(
            mp.LMCacheMPRequestMetadata.GetStoreMetadata(
                t, 1, BLOCK, min_decode_chunks=chunks
            )
        )

    step(prompt)
    for i in range(output):
        t.all_token_ids.append(prompt + i)
        step(1)
    record(mp.LMCacheMPRequestMetadata.GetStoreMetadata(t, 1, BLOCK, flush=True))
    return stores, t


def test_one_chunk_is_the_old_behaviour():
    stores, t = _run(100, 300, 1)
    assert stores[0] == (0, 6, False)
    assert all(span == 1 and not flush for _, span, flush in stores[1:])
    assert t.num_stored_blocks == 400 // BLOCK


def test_generated_chunks_wait_for_n_and_the_rest_is_flushed():
    stores, t = _run(100, 300, 4)
    # The prompt's full chunks at once, then the chunk holding its last
    # tokens on its own, then generated chunks four at a time.
    assert stores[:2] == [(0, 6, False), (6, 1, False)]
    assert all(span == 4 for _, span, flush in stores[2:-1])
    first, span, flush = stores[-1]
    assert flush and 1 <= span < 4
    # Exactly the chunks the per-chunk path stores, each once, in order.
    covered = [c for first, span, _ in stores for c in range(first, first + span)]
    assert covered == list(range(400 // BLOCK))
    assert t.num_stored_blocks == 400 // BLOCK


def test_nothing_held_means_no_flush():
    stores, _ = _run(100, 4, 4)
    assert not any(flush for *_, flush in stores)


def test_a_pending_prompt_chunk_is_never_flushed():
    # A phantom finalized after its load: never scheduled, so APC-hit blocks
    # beyond what LMCache holds are pending -- and the per-chunk path never
    # stored them either.
    t = _tracker(96, stored=2, vllm_hits=5, lmcache_hits=2)
    t.block_hashes = list(range(6))
    t.allocated_block_ids = list(range(6))
    assert mp.LMCacheMPRequestMetadata.GetStoreMetadata(t, 1, BLOCK, flush=True) is None


def test_a_resumed_request_stores_nothing_more():
    # Preemption freed the blocks its tracker still points at.
    t = _tracker(96)
    t.num_scheduled_tokens = 200
    t.all_token_ids = list(range(200))
    t.block_hashes = list(range(12))
    t.allocated_block_ids = list(range(12))
    t.resumed = True
    assert (
        mp.LMCacheMPRequestMetadata.GetStoreMetadata(t, 1, BLOCK, min_decode_chunks=4)
        is None
    )
    assert mp.LMCacheMPRequestMetadata.GetStoreMetadata(t, 1, BLOCK, flush=True) is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("", 1), ("16", 16), ("0", 1), ("-3", 1), ("x", 1)]
)
def test_setting(monkeypatch, raw, expected):
    monkeypatch.setenv(mp.DECODE_STORE_CHUNKS_ENV_VAR, raw)
    assert mp._decode_store_chunks() == expected
