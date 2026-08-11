# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot sources — build step 2 (02 §10).

`KVCacheManager`, `BlockPool` and `Scheduler` all live in `EngineCoreProc`,
and the engine loop must never touch a socket: a synchronous Redis GET in the
tick puts network RTT on the critical path of every scheduler step. So the
forecast reaches the loop by **reference swap, never by a call**.

A source owns whatever machinery it needs (a thread, a file, nothing at all)
and publishes by assigning to `self._snapshot`. The reader takes one
reference per tick. That single attribute assignment is atomic under the GIL
and the snapshot is immutable, which is what retires the previous attempt's
`RLock` — it was guarding a race that does not exist once the handoff is a
reference swap (01 §2).
"""

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.core.node_eviction.types import (
    EMPTY_SNAPSHOT,
    ImportanceRow,
    ImportanceSnapshot,
    NodeKey,
)

logger = init_logger(__name__)


class SnapshotSource(ABC):
    """Publishes `ImportanceSnapshot`s to the scheduler thread."""

    def __init__(self) -> None:
        self._snapshot: ImportanceSnapshot = EMPTY_SNAPSHOT
        self._revision = 0

    def get_snapshot(self) -> ImportanceSnapshot:
        """One attribute read. No lock, no allocation, no I/O.

        Callers compare the returned object with `is` against the previous
        tick's to decide whether anything changed.
        """
        return self._snapshot

    def _publish(
        self,
        rows: dict[NodeKey, ImportanceRow],
        built_at_ms: float,
        built_at_monotonic: float,
    ) -> None:
        self._revision += 1
        self._snapshot = ImportanceSnapshot(
            rows=rows,
            built_at_ms=built_at_ms,
            built_at_monotonic=built_at_monotonic,
            revision=self._revision,
        )

    def note_job(self, job_id: str) -> None:
        """Tell the source that a job is live.

        The scheduler thread calls this when it first indexes blocks for a
        job. It must be non-blocking and must not touch a socket; the Redis
        source turns it into a `put_nowait` on a bounded queue, which drives
        seeding for jobs first seen after startup (01 §4).
        """

    def start(self) -> None:
        """Optional. Called once from engine core after construction."""

    def close(self) -> None:
        """Optional. Called on shutdown."""

    @property
    def healthy(self) -> bool:
        return True

    def stats(self) -> dict[str, float | int | bool]:
        return {"revision": self._revision}


class StaticSnapshotSource(SnapshotSource):
    """A snapshot handed over directly. No thread, no Redis.

    This is what build step 2 is tested against, and what the replay harness
    drives: it makes every downstream stage exercisable before the transport
    exists.
    """

    def __init__(
        self,
        rows: dict[NodeKey, ImportanceRow] | None = None,
        built_at_ms: float = 0.0,
        built_at_monotonic: float = 0.0,
    ) -> None:
        super().__init__()
        if rows:
            self._publish(dict(rows), built_at_ms, built_at_monotonic)

    def set_rows(
        self,
        rows: dict[NodeKey, ImportanceRow],
        built_at_ms: float = 0.0,
        built_at_monotonic: float = 0.0,
    ) -> None:
        """Replace the snapshot. Publishes a new immutable object rather than
        mutating the live one, exactly as the Redis thread does."""
        self._publish(dict(rows), built_at_ms, built_at_monotonic)


class JSONFileSnapshotSource(StaticSnapshotSource):
    """Reads a snapshot from a JSON file once at construction.

    Useful for driving the policy on a dev box with no Redis. Same payload
    shape the Redis source decodes, so a captured `PSUBSCRIBE` dump can be
    replayed through it.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.reload()

    def reload(self) -> None:
        with open(self.path) as f:
            payload = json.load(f)
        rows = decode_rows(payload)
        self._publish(rows, 0.0, 0.0)


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def decode_rows(
    payload: object,
    job_id: str | None = None,
    into: dict[NodeKey, ImportanceRow] | None = None,
) -> dict[NodeKey, ImportanceRow]:
    """Decode a `PROB`/`HISTORY` payload into forecast rows.

    Accepts the shapes 03 §4 leaves open, because which one the writers use
    is still unanswered (03 §7 questions 4 and 6):

    - a list of row objects,
    - a mapping of `node_name -> row` or `node_name -> [rows]`,
    - a single row object,
    - either of the above wrapped in `{"job_id": ..., "rows": [...]}`.

    Fields are merged by key, so `PROB` and `HISTORY` for the same
    `(node, call_type)` combine into one row regardless of arrival order.
    A row missing `node_name` or `call_type` is dropped rather than guessed
    at: a mis-joined row is silent, and silence is the failure mode this
    design is least able to detect (03 §5).
    """
    rows: dict[NodeKey, ImportanceRow] = {} if into is None else into

    if isinstance(payload, dict):
        if "rows" in payload and isinstance(payload["rows"], (list, tuple)):
            job_id = str(payload.get("job_id", job_id or ""))
            for item in payload["rows"]:
                _decode_one(item, job_id, rows)
            return rows
        if "node_name" in payload:
            _decode_one(payload, job_id, rows)
            return rows
        # node_name -> row | [row, ...]
        for node_name, value in payload.items():
            if isinstance(value, dict):
                _decode_one({"node_name": node_name, **value}, job_id, rows)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict):
                        _decode_one(
                            {"node_name": node_name, **item}, job_id, rows
                        )
        return rows

    if isinstance(payload, (list, tuple)):
        for item in payload:
            _decode_one(item, job_id, rows)
        return rows

    logger.warning(
        "Unrecognised importance payload of type %s; ignoring", type(payload)
    )
    return rows


def _decode_one(
    item: object,
    job_id: str | None,
    rows: dict[NodeKey, ImportanceRow],
) -> None:
    if not isinstance(item, dict):
        return
    node_name = item.get("node_name")
    call_type = item.get("call_type")
    row_job_id = item.get("job_id", job_id)
    if node_name is None or call_type is None or row_job_id is None:
        logger.debug(
            "Dropping importance row without job_id/node_name/call_type: %s",
            item,
        )
        return

    key = NodeKey(str(row_job_id), str(node_name), str(call_type))
    prev = rows.get(key)

    def pick(field: str, fallback: float) -> float:
        if field in item:
            return _as_float(item[field], fallback)
        return fallback

    base = prev if prev is not None else ImportanceRow()

    # Declared by the publisher; absent for a forecast that answers the
    # time-free "fires again at all", which is every publisher predating the
    # field. Absent therefore has to keep the old scoring exactly.
    if "prob_horizon" in item:
        horizon = _as_int(item["prob_horizon"])
        prob_horizon = horizon if horizon and horizon > 0 else None
    else:
        prob_horizon = base.prob_horizon

    delta_l1 = item.get("time_taken_l1", item.get("delta_l1_ms"))
    delta_cold = item.get("time_taken_cold", item.get("delta_cold_ms"))

    rows[key] = ImportanceRow(
        prob=pick("prob", base.prob),
        time_to_next_call_ms=pick(
            "time_to_next_call", base.time_to_next_call_ms
        ),
        update_ts_ms=max(pick("update_ts", base.update_ts_ms), base.update_ts_ms),
        prob_horizon=prob_horizon,
        p_l1=pick("p_l1", base.p_l1),
        p_cold=pick("p_cold", base.p_cold),
        delta_l1_ms=(
            _as_float(delta_l1) if delta_l1 is not None else base.delta_l1_ms
        ),
        delta_cold_ms=(
            _as_float(delta_cold) if delta_cold is not None else base.delta_cold_ms
        ),
        num_samples=(
            _as_int(item["n"]) if "n" in item else base.num_samples
        ),
    )


class ThreadedSnapshotSource(SnapshotSource):
    """Base for sources that run their own daemon thread.

    Failure isolation is mandatory: Redis down, malformed JSON, a dropped
    subscription — none of it may propagate into the scheduler loop. The
    thread catches everything, logs, backs off and reconnects; the snapshot
    simply goes stale in the meantime, and the staleness gate turns that into
    LRU behaviour rather than a wrong answer.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._name = name

    def start(self) -> None:
        if self._thread is not None:
            return
        # Daemon, like ZmqEventPublisher's "zmq-publisher" thread and the two
        # IO threads EngineCoreProc already starts: the engine must be able
        # to exit without waiting on the forecast.
        self._thread = threading.Thread(
            target=self._run_guarded, name=self._name, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception:
            logger.exception(
                "%s thread died; node eviction will run on a stale snapshot "
                "and degrade to LRU once it passes the staleness cutoff",
                self._name,
            )

    @abstractmethod
    def _run(self) -> None: ...


def rows_from_iterable(
    items: Iterable[dict], job_id: str | None = None
) -> dict[NodeKey, ImportanceRow]:
    """Convenience for tests and the replay harness."""
    rows: dict[NodeKey, ImportanceRow] = {}
    for item in items:
        _decode_one(item, job_id, rows)
    return rows
