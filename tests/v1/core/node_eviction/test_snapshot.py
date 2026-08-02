# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build step 2: snapshot publication and payload decoding.

The payload shape is still unanswered by the workflow side (03 §7 questions
4 and 6), so the decoder accepts every shape the contract leaves open and
these tests pin that down.
"""

import json

from vllm.v1.core.node_eviction.snapshot import (
    JSONFileSnapshotSource,
    StaticSnapshotSource,
    decode_rows,
)
from vllm.v1.core.node_eviction.types import ImportanceRow, NodeKey

KEY = NodeKey("run-42", "research", "tavily:summary")


def test_publish_is_a_reference_swap():
    """The handoff to the scheduler thread is a single attribute assignment
    of an immutable object — which is what retires the previous attempt's
    RLock (01 §2)."""
    source = StaticSnapshotSource()
    first = source.get_snapshot()
    source.set_rows({KEY: ImportanceRow(prob=0.5)})
    second = source.get_snapshot()

    assert first is not second
    assert second.revision > first.revision
    # The old snapshot is untouched: a reader holding it sees a consistent
    # view even while the writer publishes.
    assert first.get(KEY) is None
    assert second.get(KEY).prob == 0.5


def test_snapshot_identity_is_the_change_check():
    source = StaticSnapshotSource({KEY: ImportanceRow(prob=0.5)})
    assert source.get_snapshot() is source.get_snapshot()


def test_decode_list_payload():
    rows = decode_rows(
        [
            {
                "job_id": "run-42",
                "node_name": "research",
                "call_type": "tavily:summary",
                "prob": 0.9,
                "time_to_next_call": 8000,
                "update_ts": 1700000000000,
            }
        ]
    )
    assert rows[KEY].prob == 0.9
    assert rows[KEY].time_to_next_call_ms == 8000.0


def test_decode_job_id_from_the_channel_name():
    rows = decode_rows(
        [{"node_name": "research", "call_type": "tavily:summary", "prob": 0.5}],
        job_id="run-42",
    )
    assert KEY in rows


def test_decode_wrapped_document():
    rows = decode_rows(
        {
            "job_id": "run-42",
            "rows": [
                {"node_name": "research", "call_type": "tavily:summary", "prob": 0.7}
            ],
        }
    )
    assert rows[KEY].prob == 0.7


def test_decode_mapping_keyed_by_node():
    rows = decode_rows(
        {"research": [{"call_type": "tavily:summary", "prob": 0.3}]},
        job_id="run-42",
    )
    assert rows[KEY].prob == 0.3


def test_decode_single_row_document():
    rows = decode_rows(
        {"node_name": "research", "call_type": "tavily:summary", "prob": 0.4},
        job_id="run-42",
    )
    assert rows[KEY].prob == 0.4


def test_prob_and_history_merge_into_one_row():
    """One row per (node, call_type) regardless of which keyspace it came
    from or the order the two arrived in."""
    rows = {}
    decode_rows(
        [
            {
                "job_id": "run-42",
                "node_name": "research",
                "call_type": "tavily:summary",
                "prob": 0.8,
                "time_to_next_call": 5000,
                "update_ts": 100,
            }
        ],
        into=rows,
    )
    decode_rows(
        [
            {
                "job_id": "run-42",
                "node_name": "research",
                "call_type": "tavily:summary",
                "p_l1": 0.9,
                "p_cold": 0.1,
                "time_taken_cold": 11400,
                "update_ts": 200,
            }
        ],
        into=rows,
    )
    row = rows[KEY]
    assert row.prob == 0.8
    assert row.time_to_next_call_ms == 5000.0
    assert row.p_cold == 0.1
    assert row.delta_cold_ms == 11400.0
    # update_ts moves forward, never backward.
    assert row.update_ts_ms == 200.0


def test_rows_without_identity_are_dropped_not_guessed():
    """A mis-joined row is silent, and silence is the failure mode this
    design is least able to detect (03 §5)."""
    rows = decode_rows([{"node_name": "research", "prob": 0.9}], job_id="run-42")
    assert rows == {}
    rows = decode_rows([{"call_type": "summary", "prob": 0.9}], job_id="run-42")
    assert rows == {}


def test_string_numbers_are_accepted():
    """Redis hashes stringify everything."""
    rows = decode_rows(
        [
            {
                "job_id": "run-42",
                "node_name": "research",
                "call_type": "tavily:summary",
                "prob": "0.6",
                "time_to_next_call": "8000",
            }
        ]
    )
    assert rows[KEY].prob == 0.6
    assert rows[KEY].time_to_next_call_ms == 8000.0


def test_garbage_payload_does_not_raise():
    assert decode_rows("not a payload") == {}
    assert decode_rows(42) == {}
    assert decode_rows([None, 7, "x"]) == {}


def test_json_file_source(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            [
                {
                    "job_id": "run-42",
                    "node_name": "research",
                    "call_type": "tavily:summary",
                    "prob": 0.55,
                }
            ]
        )
    )
    source = JSONFileSnapshotSource(str(path))
    assert source.get_snapshot().get(KEY).prob == 0.55


def test_note_job_on_a_source_without_a_thread_is_a_noop():
    source = StaticSnapshotSource()
    source.note_job("run-42")
    source.start()
    source.close()
