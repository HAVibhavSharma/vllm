# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-question KV reuse attribution.

The claim these pin down is the one the experiment rests on: when job 12's
prompt hits the cache, the tokens are credited to whichever job actually
computed those blocks, and never to job 12 itself unless job 12 wrote them.
Get first-writer-wins backwards and every run reports self-reuse where there
was cross-question reuse, which is the opposite of the finding.
"""

import json
from types import SimpleNamespace

import pytest

from vllm.v1.core.kv_provenance import UNTRACKED, KVReuseProvenance

BLOCK = 16


def make_request(job_id="7", node="research", num_tokens=None, hashes=()):
    extra_args = {}
    if job_id is not None:
        extra_args["job_id"] = job_id
    if node is not None:
        extra_args["langgraph_node"] = node
    block_hashes = list(hashes)
    return SimpleNamespace(
        request_id=f"req-{job_id}",
        sampling_params=SimpleNamespace(extra_args=extra_args),
        block_hashes=block_hashes,
        num_tokens=(
            num_tokens if num_tokens is not None else len(block_hashes) * BLOCK
        ),
    )


def make_tracker(tmp_path=None, max_hashes=1000):
    tracker = KVReuseProvenance(
        max_hashes=max_hashes,
        record_path=str(tmp_path / "reuse.jsonl") if tmp_path else None,
        top_pairs=4,
    )
    tracker.configure(BLOCK)
    return tracker


def test_first_writer_keeps_the_claim():
    tracker = make_tracker()
    producer = make_request(job_id="7", hashes=[b"a", b"b"])
    tracker.claim(producer, [b"a", b"b"], BLOCK)
    # Job 12 re-caches the same content after an eviction. The compute was
    # still job 7's.
    tracker.claim(make_request(job_id="12"), [b"a", b"b"], BLOCK)

    consumer = make_request(job_id="12", hashes=[b"a", b"b", b"c"])
    tracker.record_prefill(
        consumer, num_local_cached_tokens=2 * BLOCK, num_external_cached_tokens=0
    )

    assert tracker.cross_tokens == 2 * BLOCK
    assert tracker.self_tokens == 0
    assert tracker.pair_tokens == {("7", "12"): 2 * BLOCK}


def test_self_reuse_is_not_cross_reuse():
    tracker = make_tracker()
    tracker.claim(make_request(job_id="7"), [b"a", b"b"], BLOCK)
    consumer = make_request(job_id="7", hashes=[b"a", b"b"])
    tracker.record_prefill(
        consumer, num_local_cached_tokens=2 * BLOCK, num_external_cached_tokens=0
    )

    assert tracker.self_tokens == 2 * BLOCK
    assert tracker.cross_tokens == 0
    assert tracker.pair_tokens == {}


def test_unclaimed_hits_are_unknown_not_self():
    """A hit on content this process never saw claimed.

    Happens with a connector store that outlived the server. Folding it into
    `self` would invent reuse; folding it into `cross` would invent a source.
    """
    tracker = make_tracker()
    consumer = make_request(job_id="7", hashes=[b"a", b"b"])
    tracker.record_prefill(
        consumer, num_local_cached_tokens=2 * BLOCK, num_external_cached_tokens=0
    )

    assert tracker.unknown_tokens == 2 * BLOCK
    assert tracker.self_tokens == 0
    assert tracker.cross_tokens == 0


def test_external_hits_are_attributed_after_the_local_prefix():
    tracker = make_tracker()
    tracker.claim(make_request(job_id="3"), [b"a", b"b"], BLOCK)
    tracker.claim(make_request(job_id="5"), [b"c", b"d"], BLOCK)

    consumer = make_request(job_id="9", hashes=[b"a", b"b", b"c", b"d"])
    tracker.record_prefill(
        consumer,
        num_local_cached_tokens=2 * BLOCK,
        num_external_cached_tokens=2 * BLOCK,
    )

    assert tracker.local_hit_tokens == 2 * BLOCK
    assert tracker.external_hit_tokens == 2 * BLOCK
    assert tracker.pair_tokens == {
        ("3", "9"): 2 * BLOCK,
        ("5", "9"): 2 * BLOCK,
    }


def test_external_runs_win_over_the_leading_count():
    """Non-contiguous skip: the connector matched a later range too."""
    tracker = make_tracker()
    tracker.claim(make_request(job_id="3"), [b"a"], BLOCK)
    tracker.claim(make_request(job_id="5"), [b"d"], BLOCK)

    consumer = make_request(job_id="9", hashes=[b"a", b"b", b"c", b"d"])
    tracker.record_prefill(
        consumer,
        num_local_cached_tokens=0,
        num_external_cached_tokens=BLOCK,
        # (start_token, num_tokens): block 0 and block 3, block 1-2 missing.
        external_runs=[(0, BLOCK), (3 * BLOCK, BLOCK)],
    )

    assert tracker.external_hit_tokens == 2 * BLOCK
    assert tracker.pair_tokens == {("3", "9"): BLOCK, ("5", "9"): BLOCK}


def test_external_run_overlapping_local_is_not_double_counted():
    tracker = make_tracker()
    tracker.claim(make_request(job_id="3"), [b"a", b"b"], BLOCK)

    consumer = make_request(job_id="9", hashes=[b"a", b"b"])
    tracker.record_prefill(
        consumer,
        num_local_cached_tokens=2 * BLOCK,
        num_external_cached_tokens=2 * BLOCK,
        external_runs=[(0, 2 * BLOCK)],
    )

    assert tracker.hit_tokens == 2 * BLOCK
    assert tracker.external_hit_tokens == 0


def test_requests_without_a_job_id_are_untracked():
    tracker = make_tracker()
    tracker.claim(make_request(job_id=None), [b"a"], BLOCK)
    consumer = make_request(job_id="9", hashes=[b"a"])
    tracker.record_prefill(
        consumer, num_local_cached_tokens=BLOCK, num_external_cached_tokens=0
    )

    assert tracker.pair_tokens == {(UNTRACKED, "9"): BLOCK}


def test_mismatched_block_size_is_not_claimed():
    """A group hashing at a multiple of `hash_block_size`.

    Those hashes are a different key space than `Request.block_hashes`;
    claiming them would fill the map with entries that can never be hit.
    """
    tracker = make_tracker()
    tracker.claim(make_request(job_id="3"), [b"ab"], 2 * BLOCK)
    consumer = make_request(job_id="9", hashes=[b"ab"])
    tracker.record_prefill(
        consumer, num_local_cached_tokens=BLOCK, num_external_cached_tokens=0
    )

    assert tracker.unknown_tokens == BLOCK


def test_phantom_prefetch_claims_but_does_not_consume():
    """`/v1/agents/prefetch` warms the cache; it never *experienced* a hit."""
    tracker = make_tracker()
    tracker.claim(make_request(job_id="3"), [b"a"], BLOCK)
    tracker.record_prefill(
        make_request(job_id="9", hashes=[b"a"]),
        num_local_cached_tokens=BLOCK,
        num_external_cached_tokens=0,
        phantom=True,
    )

    assert tracker.requests == 0
    assert tracker.cross_tokens == 0


def test_reset_keeps_claims_and_zeroes_totals():
    """The warm phase is *supposed* to hit what the warmup left behind."""
    tracker = make_tracker()
    tracker.claim(make_request(job_id="3"), [b"a"], BLOCK)
    tracker.record_prefill(
        make_request(job_id="9", hashes=[b"a"]),
        num_local_cached_tokens=BLOCK,
        num_external_cached_tokens=0,
    )
    before = tracker.reset(1, "cold_done")

    assert before["reuse_cross_tokens"] == BLOCK
    assert tracker.cross_tokens == 0
    assert tracker.epoch == 1

    tracker.record_prefill(
        make_request(job_id="12", hashes=[b"a"]),
        num_local_cached_tokens=BLOCK,
        num_external_cached_tokens=0,
    )
    # Still job 3's block, not `unknown`.
    assert tracker.pair_tokens == {("3", "12"): BLOCK}


def test_claim_map_is_bounded():
    tracker = make_tracker(max_hashes=4)
    for i in range(10):
        tracker.claim(make_request(job_id="3"), [bytes([i])], BLOCK)

    assert len(tracker._claims) == 4
    assert tracker.claims_trimmed == 6


def test_record_is_written_per_prefill(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.claim(make_request(job_id="3"), [b"a", b"b"], BLOCK)
    tracker.record_prefill(
        make_request(job_id="9", node="supervisor", hashes=[b"a", b"b", b"c"]),
        num_local_cached_tokens=2 * BLOCK,
        num_external_cached_tokens=0,
    )

    lines = (tmp_path / "reuse.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["job_id"] == "9"
    assert record["node"] == "supervisor"
    assert record["cross_tokens"] == 2 * BLOCK
    assert record["sources"] == {"3": 2 * BLOCK}


@pytest.mark.parametrize("hit", [0, BLOCK])
def test_summary_line_shape(hit):
    tracker = make_tracker()
    if hit:
        tracker.claim(make_request(job_id="3"), [b"a"], BLOCK)
        tracker.record_prefill(
            make_request(job_id="9", hashes=[b"a"]),
            num_local_cached_tokens=hit,
            num_external_cached_tokens=0,
        )
    line = tracker.summary()
    fields = dict(part.split("=", 1) for part in line.split()[1:])

    assert line.startswith("kv_reuse ")
    assert fields["cross"] == str(hit)
    # Never a bare division by zero on an idle server.
    assert float(fields["cross_frac"]) == (1.0 if hit else 0.0)
