# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Redis importance feed — build step 4 (02 §10).

Runs in a daemon thread inside the engine-core process and publishes
immutable snapshots by reference swap. The scheduler loop never sees a
socket.

Transport is **pub/sub with a seed read** (01 §2). Pub/sub delivers only what
is published *after* subscribe and is at-most-once, so a one-shot pipelined
`MGET` seed is mandatory at two points: on first subscribe (otherwise a
server that starts mid-job knows nothing until the next publish) and after
every reconnect (the gap is lost). A low-frequency periodic reseed backstops
a message dropped because Redis killed a slow consumer.

Several transport details are genuinely unanswered by the workflow side
(03 §7): whether the writers `PUBLISH` at all, whether the channel is the key
or one global channel, and whether a message carries the full document or
just a change notification. Each is a config knob here rather than an
assumption baked into the code, so answering them is an edit to a JSON file.
"""

import json
import time
from collections import deque
from queue import Empty, Queue

from vllm.logger import init_logger
from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.snapshot import ThreadedSnapshotSource, decode_rows
from vllm.v1.core.node_eviction.types import ImportanceRow, NodeKey

logger = init_logger(__name__)


def redis_available() -> bool:
    """Whether the optional `redis` dependency is importable.

    vLLM must boot without it; the policy just cannot be fed.
    """
    try:
        import redis  # noqa: F401
    except ImportError:
        return False
    return True


class RedisSnapshotSource(ThreadedSnapshotSource):
    """Subscribes to `PROB`/`HISTORY` and republishes them as snapshots."""

    def __init__(self, config: NodeEvictionConfig) -> None:
        super().__init__(name="node-eviction-redis")
        self.config = config
        # Jobs the scheduler thread has seen. `put_nowait` on a bounded queue
        # is a non-blocking append that touches no socket, which is the only
        # thing the loop is allowed to do here (01 §4).
        self._new_jobs: Queue[str] = Queue(maxsize=4096)
        self._known_jobs: set[str] = set()
        self._recent_jobs: deque[str] = deque(maxlen=4096)
        self._rows: dict[NodeKey, ImportanceRow] = {}
        self._connected = False
        self._decode_errors = 0
        self._reconnects = 0
        self._reseeds = 0
        self._messages = 0
        self._last_message_ts = 0.0

    # -- scheduler-thread side --------------------------------------------

    def note_job(self, job_id: str) -> None:
        if job_id in self._known_jobs:
            return
        self._known_jobs.add(job_id)
        self._recent_jobs.append(job_id)
        try:
            self._new_jobs.put_nowait(job_id)
        except Exception:
            # A full queue means seeding is behind; the periodic reseed will
            # pick the job up. Never block the scheduler for this.
            logger.debug("Node eviction job queue full; dropping seed for %s", job_id)

    @property
    def healthy(self) -> bool:
        return self._connected

    def stats(self) -> dict[str, float | int | bool]:
        base = super().stats()
        base.update(
            {
                "connected": self._connected,
                "messages": self._messages,
                "decode_errors": self._decode_errors,
                "reconnects": self._reconnects,
                "reseeds": self._reseeds,
                "tracked_jobs": len(self._known_jobs),
            }
        )
        return base

    # -- thread side -------------------------------------------------------

    def _run(self) -> None:
        try:
            import redis
        except ImportError:
            logger.warning(
                "redis is not installed; the node eviction policy will run "
                "without a forecast (blocks are indexed but never ranked). "
                "Install it with: uv pip install redis"
            )
            return

        backoff = self.config.redis_reconnect_backoff_s
        while not self._stop.is_set():
            client = None
            pubsub = None
            try:
                client = redis.Redis.from_url(
                    self.config.redis_url, decode_responses=True
                )
                client.ping()
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.psubscribe(
                    self.config.redis_prob_pattern,
                    self.config.redis_history_pattern,
                )
                self._connected = True
                self._reconnects += 1
                backoff = self.config.redis_reconnect_backoff_s
                logger.info(
                    "Node eviction connected to Redis at %s", self.config.redis_url
                )

                # Seed on subscribe, and again after every reconnect: pub/sub
                # only delivers post-subscribe messages, so the gap is lost.
                self._seed(client, self._drain_all_known_jobs())
                self._consume(client, pubsub)
            except Exception as exc:
                self._connected = False
                logger.warning(
                    "Node eviction Redis connection failed (%s); retrying in "
                    "%.1fs. The policy degrades to LRU past the staleness "
                    "cutoff.",
                    exc,
                    backoff,
                )
                self._stop.wait(backoff)
                backoff = min(
                    backoff * 2.0, self.config.redis_reconnect_backoff_max_s
                )
            finally:
                self._connected = False
                for closeable in (pubsub, client):
                    if closeable is not None:
                        try:
                            closeable.close()
                        except Exception:
                            pass

    def _drain_all_known_jobs(self) -> list[str]:
        """Every job seen so far, plus anything queued since."""
        self._drain_new_jobs()
        return list(self._recent_jobs)

    def _drain_new_jobs(self) -> list[str]:
        drained: list[str] = []
        while True:
            try:
                drained.append(self._new_jobs.get_nowait())
            except Empty:
                return drained

    def _consume(self, client, pubsub) -> None:
        """Steady state: block on `get_message`, with a timeout so shutdown
        and the job queue stay responsive."""
        last_reseed = time.monotonic()
        while not self._stop.is_set():
            message = pubsub.get_message(
                timeout=self.config.redis_message_timeout_s
            )
            if message is not None:
                self._handle_message(client, message)

            # Seed jobs that appeared after we subscribed. Keys are
            # *constructed* from the active-job set rather than SCANned, so
            # this costs one round trip per batch of new jobs.
            new_jobs = self._drain_new_jobs()
            if new_jobs:
                self._seed(client, new_jobs)

            now = time.monotonic()
            if now - last_reseed >= self.config.redis_reseed_period_s:
                # Belt and braces against an at-most-once drop.
                self._seed(client, list(self._recent_jobs))
                last_reseed = now

    def _handle_message(self, client, message: dict) -> None:
        self._messages += 1
        self._last_message_ts = time.monotonic()
        channel = message.get("channel") or ""
        data = message.get("data")

        if not self.config.redis_payload_is_document:
            # The message is only "key X changed"; fetch the document.
            key = data if isinstance(data, str) else channel
            try:
                data = client.get(key)
            except Exception:
                logger.debug("Node eviction GET failed for %s", key, exc_info=True)
                return
            channel = key

        if data is None:
            return
        self._ingest(channel, data)

    def _seed(self, client, job_ids: list[str]) -> None:
        """One pipelined MGET over the active jobs' keys.

        `MGET` on a key that does not exist yet returns None, which leaves
        the key unscored. A job whose PROB row has not been written is not an
        error case (03 §3.5).
        """
        if not job_ids:
            return
        prob_prefix = self.config.redis_prob_pattern.rstrip("*")
        history_prefix = self.config.redis_history_pattern.rstrip("*")
        keys = [f"{prob_prefix}{job_id}" for job_id in job_ids]
        keys += [f"{history_prefix}{job_id}" for job_id in job_ids]
        try:
            values = client.mget(keys)
        except Exception:
            logger.debug("Node eviction seed MGET failed", exc_info=True)
            return
        self._reseeds += 1
        changed = False
        for key, value in zip(keys, values):
            if value is not None:
                changed |= self._ingest(key, value, publish=False)
        if changed:
            self._republish()

    def _ingest(self, channel: str, data: object, publish: bool = True) -> bool:
        """Decode one payload into the row table. Returns whether it changed
        anything worth republishing."""
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError:
                self._decode_errors += 1
                return False
        if not isinstance(data, str):
            return False

        try:
            payload = json.loads(data)
        except ValueError:
            self._decode_errors += 1
            logger.debug("Node eviction dropped malformed payload on %s", channel)
            return False

        job_id = channel.split("|", 1)[1] if "|" in channel else None
        before = dict(self._rows)
        decode_rows(payload, job_id=job_id, into=self._rows)
        if self._rows == before:
            return False
        if publish:
            self._republish()
        return True

    def _republish(self) -> None:
        # Copy on publish: the snapshot handed to the scheduler thread must
        # never be mutated afterwards, which is what makes the lock-free
        # handoff sound.
        self._publish(dict(self._rows), time.time() * 1000.0, time.monotonic())


def build_snapshot_source(config: NodeEvictionConfig):
    """Pick a source from config. Returns None when there is nothing to
    connect to, in which case the policy indexes blocks but ranks nothing."""
    if not config.redis_url:
        return None
    if not redis_available():
        logger.warning(
            "VLLM_NODE_EVICTION_REDIS_URL is set but the redis package is not "
            "installed; the policy will index blocks but rank nothing. "
            "Install it with: uv pip install redis"
        )
        return None
    return RedisSnapshotSource(config)
