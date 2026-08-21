# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Who produced the prefix this request just reused.

`hbm_summary.py` answers *how much* of the cache was hit. It cannot answer
*whose*: `on_cache_query` is handed two integers, and a hit rate of 60% is
the same number whether every request is re-reading its own prefill or
question 12 is riding entirely on blocks question 7 paid for. For an agentic
workload the second case is the whole point — the questions share a system
prompt, tool schemas and often whole research turns — and it is invisible in
every aggregate the server already emits.

So this tracks one extra fact per block: **the first job that computed it**.

    claim:   job 7 prefills, its blocks become full, hash H -> job 7
    lookup:  job 12 arrives, the prefix cache hits on H,
             so 16 of job 12's "cached" tokens came from job 7

The map is keyed on the *bare* block hash at `hash_block_size` granularity —
the same key space as `Request.block_hashes` — rather than on block ids or
the group-qualified hash stored on a `KVCacheBlock`. Three reasons:

1. Block ids are recycled. A block id names a slot in HBM, not content, so
   an id-keyed owner is overwritten by whoever allocates the slot next and
   the provenance of an evicted-then-rematerialized prefix is lost.
2. External (KV connector / LMCache) hits never touch a `KVCacheBlock` at
   all. They are hits all the same — for cross-question reuse they are the
   *likely* case, since HBM churns within a job while the CPU tier survives
   across them — and the only handle on them is the request's own hash list.
3. First-writer-wins is the honest attribution. A prefix that ten jobs share
   was paid for once; crediting the tenth reader would report reuse that
   never happened.

`unknown` is a real bucket and is reported rather than folded into `self`:
a hit on content this process never claimed (a connector store that outlived
the server, or a claim trimmed under `VLLM_KV_PROVENANCE_MAX_HASHES`) is a
hit whose source is genuinely not known here.

Off unless `VLLM_KV_PROVENANCE=1`. It costs a dict entry per distinct block
hash and a lookup per hit block on the scheduling path, which is small but
not free, and a baseline arm should be able to run without it.

Env:
    VLLM_KV_PROVENANCE            1 to enable (default 0)
    VLLM_KV_PROVENANCE_PATH       JSONL sink, one record per prefill.
                                  Unset = aggregates only, no file.
    VLLM_KV_PROVENANCE_MAX_HASHES claim-map cap, FIFO-trimmed (default 1e6)
    VLLM_KV_PROVENANCE_TOP        pairs on the `kv_reuse` line (default 8)
"""

import json
import os
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

UNTRACKED = "<untracked>"

# Mirrors `hbm_summary._JOB_ID_FIELD`: identity rides in
# `sampling_params.extra_args` and already crosses the front-end ->
# engine-core boundary, so nothing new is plumbed for it.
_JOB_ID_FIELD = "job_id"
_NODE_FIELD = "langgraph_node"
_AGENT_ID_FIELD = "agent_id"


def _extra_args(request) -> dict[str, Any] | None:
    sampling_params = getattr(request, "sampling_params", None)
    if sampling_params is None:
        return None
    extra_args = getattr(sampling_params, "extra_args", None)
    return extra_args or None


def job_id_for_request(request) -> str:
    """The LangGraph job that issued this request, or `<untracked>`.

    One job is one question: `tests/run_evaluate_prefix.py::target` mints a
    fresh `job_id` per graph invocation and forwards it in `extra_body`, so
    every LLM call the supervisor, researchers and compression nodes make
    while answering that question carries the same value.
    """
    extra_args = _extra_args(request)
    if not extra_args:
        return UNTRACKED
    job_id = extra_args.get(_JOB_ID_FIELD)
    if job_id is None:
        return UNTRACKED
    return str(job_id)


def _node_for_request(request) -> str | None:
    extra_args = _extra_args(request)
    if not extra_args:
        return None
    node = extra_args.get(_NODE_FIELD)
    return str(node) if node is not None else None


def _agent_for_request(request) -> str | None:
    extra_args = _extra_args(request)
    if not extra_args:
        return None
    agent_id = extra_args.get(_AGENT_ID_FIELD)
    return str(agent_id) if agent_id is not None else None


class KVReuseProvenance:
    """Block-hash -> first-writer job, and the reuse matrix that falls out."""

    def __init__(
        self,
        *,
        max_hashes: int,
        record_path: str | None,
        top_pairs: int,
    ) -> None:
        self.max_hashes = max_hashes
        self.record_path = record_path
        self.top_pairs = top_pairs

        # Set by `configure()` once the KV cache layout is known. Until then
        # nothing is recorded: without it there is no way to turn a hit token
        # count into a block index.
        self.hash_block_size: int | None = None

        # Interned job labels. The claim map holds one entry per distinct
        # block hash the server has ever cached, so the value has to be
        # small; an index into `_jobs` is 28 bytes where the string is 50+.
        self._jobs: list[str] = []
        self._job_index: dict[str, int] = {}

        # Insertion-ordered, so the FIFO trim below drops the oldest claims.
        # Oldest is the right thing to drop: a hash claimed a long time ago
        # and never hit since is the one least likely to be hit next.
        self._claims: dict[Any, int] = {}
        self.claims_trimmed = 0

        self._sink = None
        self._sink_failed = False

        self.epoch = 0
        self.epoch_label = "start"
        self._reset_totals()

    def _reset_totals(self) -> None:
        """Zero the aggregates. **Does not clear the claim map.**

        A measurement reset marks the end of warmup, not the end of the
        cache: the blocks the warmup left resident (and everything the
        connector holds) are exactly what the measured phase is supposed to
        hit. Dropping their claims would re-label every one of those hits
        `unknown` and erase the reuse the run exists to measure.
        """
        self.requests = 0
        self.hit_tokens = 0
        self.self_tokens = 0
        self.cross_tokens = 0
        self.unknown_tokens = 0
        self.local_hit_tokens = 0
        self.external_hit_tokens = 0
        # (source_job, consumer_job) -> tokens, cross-job only.
        self.pair_tokens: dict[tuple[str, str], int] = {}
        # source_job -> tokens it supplied to *other* jobs.
        self.source_tokens: dict[str, int] = {}
        # consumer_job -> tokens it took from *other* jobs.
        self.consumer_tokens: dict[str, int] = {}

    @classmethod
    def maybe_build(cls) -> "KVReuseProvenance | None":
        """Off unless `VLLM_KV_PROVENANCE=1`.

        Read straight from the environment rather than added to `envs.py`,
        matching `HBMSummaryLogger.maybe_build`: this is comparison
        instrumentation on a baseline clone, and keeping it out of the config
        surface keeps the diff against upstream small.
        """
        enabled = os.environ.get("VLLM_KV_PROVENANCE", "0").strip().lower()
        if enabled not in ("1", "true", "yes", "on"):
            return None
        try:
            max_hashes = int(os.environ.get("VLLM_KV_PROVENANCE_MAX_HASHES", "1000000"))
        except ValueError:
            max_hashes = 1_000_000
        try:
            top_pairs = int(os.environ.get("VLLM_KV_PROVENANCE_TOP", "8"))
        except ValueError:
            top_pairs = 8
        record_path = os.environ.get("VLLM_KV_PROVENANCE_PATH") or None
        logger.info(
            "kv_reuse: provenance tracking on (max_hashes=%d, path=%s)",
            max_hashes,
            record_path or "<none>",
        )
        return cls(
            max_hashes=max(max_hashes, 0),
            record_path=record_path,
            top_pairs=max(top_pairs, 0),
        )

    def configure(self, hash_block_size: int) -> None:
        self.hash_block_size = hash_block_size

    # -- claim side --------------------------------------------------------

    def claim(self, request, block_hashes, block_size: int) -> None:
        """First job to cache these hashes owns them.

        `block_size` is the granularity of `block_hashes` as `BlockPool`
        computed it. When KV cache groups disagree on block size it can be a
        multiple of `hash_block_size`, and those hashes live in a different
        key space than the ones the lookup side reads off
        `Request.block_hashes`. Claiming them would put entries in the map
        that can never be hit and, worse, could collide; the group that does
        hash at `hash_block_size` covers the same content anyway.
        """
        if self.hash_block_size is None or block_size != self.hash_block_size:
            return
        job = self._job_slot(job_id_for_request(request))
        for block_hash in block_hashes:
            if block_hash is None:
                continue
            # setdefault, not assignment: the *first* writer paid the
            # compute. Later jobs that reach this hash are hitting it, and
            # `cache_full_blocks` runs for them too when the block leaves
            # and re-enters the cache.
            if block_hash not in self._claims:
                self._claims[block_hash] = job
        self._trim()

    def _job_slot(self, label: str) -> int:
        slot = self._job_index.get(label)
        if slot is None:
            slot = len(self._jobs)
            self._jobs.append(label)
            self._job_index[label] = slot
        return slot

    def _trim(self) -> None:
        if self.max_hashes <= 0:
            return
        excess = len(self._claims) - self.max_hashes
        if excess <= 0:
            return
        iterator = iter(self._claims)
        for _ in range(excess):
            self._claims.pop(next(iterator), None)
            self.claims_trimmed += 1

    # -- lookup side -------------------------------------------------------

    def record_prefill(
        self,
        request,
        *,
        num_local_cached_tokens: int,
        num_external_cached_tokens: int,
        external_runs: list[tuple[int, int]] | None = None,
        phantom: bool = False,
    ) -> None:
        """One request's prefix-cache hit, resolved to the jobs that made it.

        Called once per request from `Scheduler.schedule`, where the local
        and external hit counts are both known and both final — the two
        tiers are looked up in different places (`KVCacheManager` and the
        connector) and neither one alone is the answer.

        `external_runs` are the connector's matched ranges in absolute prompt
        positions. When present they are used in preference to
        `num_external_cached_tokens`, which is only the *leading* run: with
        non-contiguous skip, later runs are served from the connector too,
        just on subsequent prefill steps. Both totals go into the record so
        the difference is visible rather than folded away.
        """
        if self.hash_block_size is None:
            return
        block_size = self.hash_block_size
        block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes:
            return

        consumer = job_id_for_request(request)

        # (start_block, num_blocks, tier)
        covered: list[tuple[int, int, str]] = []
        local_blocks = num_local_cached_tokens // block_size
        if local_blocks > 0:
            covered.append((0, local_blocks, "local"))

        external_from_runs = 0
        if external_runs:
            for start_token, length in external_runs:
                start_block = start_token // block_size
                num_blocks = length // block_size
                if num_blocks <= 0:
                    continue
                # A run can overlap the local hit when the connector reports
                # a prefix HBM already holds. Counting it in both tiers would
                # make the tokens sum past the prompt.
                if start_block < local_blocks:
                    shift = local_blocks - start_block
                    start_block, num_blocks = local_blocks, num_blocks - shift
                    if num_blocks <= 0:
                        continue
                covered.append((start_block, num_blocks, "external"))
                external_from_runs += num_blocks * block_size
        elif num_external_cached_tokens > 0:
            external_blocks = num_external_cached_tokens // block_size
            if external_blocks > 0:
                covered.append((local_blocks, external_blocks, "external"))

        # A full miss still gets a record. It is the denominator: "job 12
        # reused 40k tokens from job 7" means nothing without the prompts
        # that reused nothing, and those are exactly the requests with no
        # covered range.
        by_source: dict[str, int] = {}
        by_source_local: dict[str, int] = {}
        by_source_external: dict[str, int] = {}
        hit_tokens = 0
        local_tokens = 0
        external_tokens = 0
        unknown_tokens = 0

        for start_block, num_blocks, tier in covered:
            end_block = min(start_block + num_blocks, len(block_hashes))
            for index in range(start_block, end_block):
                slot = self._claims.get(block_hashes[index])
                source = UNTRACKED if slot is None else self._jobs[slot]
                hit_tokens += block_size
                if tier == "local":
                    local_tokens += block_size
                else:
                    external_tokens += block_size
                if slot is None:
                    unknown_tokens += block_size
                    continue
                by_source[source] = by_source.get(source, 0) + block_size
                bucket = by_source_local if tier == "local" else by_source_external
                bucket[source] = bucket.get(source, 0) + block_size

        self_tokens = by_source.get(consumer, 0)
        cross_tokens = hit_tokens - unknown_tokens - self_tokens

        if not phantom:
            self.requests += 1
            self.hit_tokens += hit_tokens
            self.local_hit_tokens += local_tokens
            self.external_hit_tokens += external_tokens
            self.self_tokens += self_tokens
            self.cross_tokens += cross_tokens
            self.unknown_tokens += unknown_tokens
            for source, tokens in by_source.items():
                if source == consumer:
                    continue
                key = (source, consumer)
                self.pair_tokens[key] = self.pair_tokens.get(key, 0) + tokens
                self.source_tokens[source] = (
                    self.source_tokens.get(source, 0) + tokens
                )
            if cross_tokens:
                self.consumer_tokens[consumer] = (
                    self.consumer_tokens.get(consumer, 0) + cross_tokens
                )

        self._write_record(
            {
                "ts": time.time(),
                "epoch": self.epoch,
                "epoch_label": self.epoch_label,
                "request_id": getattr(request, "request_id", None),
                "job_id": consumer,
                "node": _node_for_request(request),
                "agent_id": _agent_for_request(request),
                "phantom": phantom,
                "prompt_tokens": getattr(request, "num_tokens", 0),
                "block_size": block_size,
                "hit_tokens": hit_tokens,
                "hit_tokens_local": local_tokens,
                "hit_tokens_external": external_tokens,
                # Leading run only, as the scheduler counts it; differs from
                # `hit_tokens_external` exactly when the skip is
                # non-contiguous.
                "external_leading_tokens": num_external_cached_tokens,
                "external_run_tokens": external_from_runs,
                "self_tokens": self_tokens,
                "cross_tokens": cross_tokens,
                "unknown_tokens": unknown_tokens,
                "sources": by_source,
                "sources_local": by_source_local,
                "sources_external": by_source_external,
            }
        )

    # -- output ------------------------------------------------------------

    def _write_record(self, record: dict[str, Any]) -> None:
        if self.record_path is None or self._sink_failed:
            return
        if self._sink is None:
            try:
                directory = os.path.dirname(self.record_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                self._sink = open(self.record_path, "a", buffering=1)
            except OSError as exc:
                # A bad path is not worth failing a benchmark over, but it is
                # worth saying once: the run will otherwise finish with an
                # empty analysis and no explanation.
                logger.warning(
                    "kv_reuse: cannot open %s (%s); records disabled",
                    self.record_path,
                    exc,
                )
                self._sink_failed = True
                return
        try:
            self._sink.write(json.dumps(record) + "\n")
        except (OSError, TypeError) as exc:
            logger.warning("kv_reuse: record write failed (%s); disabled", exc)
            self._sink_failed = True

    def top_pair_list(self) -> list[tuple[str, str, int]]:
        pairs = sorted(self.pair_tokens.items(), key=lambda kv: -kv[1])
        if self.top_pairs:
            pairs = pairs[: self.top_pairs]
        return [(src, dst, tokens) for (src, dst), tokens in pairs]

    def summary(self) -> str:
        """One `kv_reuse` line, shaped like the `kv_hbm` line beside it."""
        hit = self.hit_tokens or 1
        top = ",".join(
            f"{src}->{dst}:{tokens}" for src, dst, tokens in self.top_pair_list()
        )
        return (
            f"kv_reuse epoch={self.epoch} label={self.epoch_label} "
            f"reqs={self.requests} "
            f"hit_tokens={self.hit_tokens} "
            f"local={self.local_hit_tokens} external={self.external_hit_tokens} "
            f"self={self.self_tokens} cross={self.cross_tokens} "
            f"unknown={self.unknown_tokens} "
            f"cross_frac={self.cross_tokens / hit:.3f} "
            f"src_jobs={len(self.source_tokens)} "
            f"claims={len(self._claims)} trimmed={self.claims_trimmed} "
            f"top={top or '-'}"
        )

    def stats(self) -> dict[str, Any]:
        hit = self.hit_tokens or 1
        return {
            "reuse_requests": self.requests,
            "reuse_hit_tokens": self.hit_tokens,
            "reuse_local_hit_tokens": self.local_hit_tokens,
            "reuse_external_hit_tokens": self.external_hit_tokens,
            "reuse_self_tokens": self.self_tokens,
            "reuse_cross_tokens": self.cross_tokens,
            "reuse_unknown_tokens": self.unknown_tokens,
            "reuse_cross_frac": self.cross_tokens / hit,
            "reuse_claims": len(self._claims),
            "reuse_claims_trimmed": self.claims_trimmed,
            "reuse_top_pairs": [
                {"source_job": src, "consumer_job": dst, "tokens": tokens}
                for src, dst, tokens in self.top_pair_list()
            ],
            "reuse_by_source_job": dict(
                sorted(self.source_tokens.items(), key=lambda kv: -kv[1])
            ),
            "reuse_by_consumer_job": dict(
                sorted(self.consumer_tokens.items(), key=lambda kv: -kv[1])
            ),
        }

    def reset(self, epoch: int, label: str) -> dict[str, Any]:
        before = self.stats()
        self.epoch = epoch
        self.epoch_label = label
        self._reset_totals()
        return before
