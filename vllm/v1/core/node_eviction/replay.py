# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline replay harness — plan/new-eviction/08-scoring.md §7.

**The formula cannot be validated by inspection, and it does not need a
server to validate.** This module replays a captured `FileStatLogger` JSONL
trace against a simulated block index and free queue — no model, no GPU, no
Redis — and reports what each policy would have done.

What it buys, none of which production logging can give you:

- **a counterfactual LRU arm on the identical trace**, which is the only way
  to answer "is this worse than doing nothing" (07 §8);
- **formula iteration in minutes**, so `tau`, the term set and
  threshold-vs-worst-K can be swept rather than argued about;
- **a falsification test per term** — drop `prob`, drop `decay`, drop
  `/ blocks`, and see which ablations actually move hit rate. It is entirely
  possible that one term carries everything and the rest are decoration.

Usage:

    python -m vllm.v1.core.node_eviction.replay \\
        --trace $VLLM_REQUEST_STATS_DIR/finished_requests_engine0_*.jsonl \\
        --num-blocks 4096 --block-size 16 --compare

The simulation is deliberately coarse: a trace records token counts, not
block hashes, so a key's prefix is modelled as a contiguous run of blocks
whose length is the request's prompt length. That is enough to rank policies
against each other, which is the question, and not enough to predict an
absolute hit rate, which is not.
"""

import argparse
import glob
import json
import math
from collections import deque
from dataclasses import dataclass, field

from vllm.v1.core.node_eviction.config import NodeEvictionConfig
from vllm.v1.core.node_eviction.scoring import score_key
from vllm.v1.core.node_eviction.types import ImportanceRow, NodeKey


@dataclass(slots=True)
class ReplayRequest:
    request_id: str
    key: NodeKey
    arrival_ts: float
    finish_ts: float
    num_prompt_tokens: int
    num_local_cached_tokens: int = 0
    num_external_cached_tokens: int = 0

    @property
    def observed_class(self) -> str:
        """Hit class as the engine actually recorded it (04 §5.4).

        Kept alongside the simulated outcome so a replay can be sanity
        checked against what really happened.
        """
        if self.num_local_cached_tokens > 0:
            return "gpu"
        if self.num_external_cached_tokens > 0:
            return "l1"
        return "cold"


def load_trace(paths: list[str], use_call_type: bool = True) -> list[ReplayRequest]:
    """Read one or more JSONL files written by `FileStatLogger`.

    Records for one job can be split across engine files under data
    parallelism, so everything is merged and sorted by `arrival_ts` (04 §4).
    """
    records: list[ReplayRequest] = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        # A trailing partial line is normal: the writer is
                        # line-buffered but not atomic.
                        continue
                    request = _row_to_request(row, use_call_type)
                    if request is not None:
                        records.append(request)
    records.sort(key=lambda r: r.arrival_ts)
    return records


def _row_to_request(row: dict, use_call_type: bool) -> ReplayRequest | None:
    job_id = row.get("job_id")
    node = row.get("langgraph_node")
    if not job_id or not node:
        # No identity: the policy would have been inert for this request, so
        # it contributes nothing to a policy comparison.
        return None
    call_type = row.get("call_type") if use_call_type else None
    arrival = row.get("arrival_ts")
    finish = row.get("finish_ts")
    if arrival is None:
        return None
    if finish is None:
        finish = arrival + float(row.get("e2e_latency") or 0.0)
    return ReplayRequest(
        request_id=str(row.get("request_id", "")),
        key=NodeKey(str(job_id), str(node), str(call_type or "")),
        arrival_ts=float(arrival),
        finish_ts=float(finish),
        num_prompt_tokens=int(row.get("num_prompt_tokens") or 0),
        num_local_cached_tokens=int(row.get("num_local_cached_tokens") or 0),
        num_external_cached_tokens=int(row.get("num_external_cached_tokens") or 0),
    )


@dataclass
class ReplayResult:
    policy: str
    blocks_needed: int = 0
    blocks_hit: int = 0
    requests: int = 0
    requests_full_hit: int = 0
    evictions: int = 0
    evictions_regretted: int = 0
    # Score/outcome correlation inputs (07 §5.1).
    needed_score_sum: float = 0.0
    needed_count: int = 0
    not_needed_score_sum: float = 0.0
    not_needed_count: int = 0
    per_key_hit: dict[NodeKey, tuple[int, int]] = field(default_factory=dict)

    @property
    def block_hit_rate(self) -> float:
        if self.blocks_needed == 0:
            return 0.0
        return self.blocks_hit / self.blocks_needed

    @property
    def full_hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.requests_full_hit / self.requests

    @property
    def regret_rate(self) -> float:
        if self.evictions == 0:
            return 0.0
        return self.evictions_regretted / self.evictions

    @property
    def score_outcome_correlation(self) -> float:
        if self.needed_count == 0 or self.not_needed_count == 0:
            return 0.0
        a = self.needed_score_sum / self.needed_count
        b = self.not_needed_score_sum / self.not_needed_count
        denom = abs(a) + abs(b)
        return (a - b) / denom if denom else 0.0

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "requests": self.requests,
            "block_hit_rate": round(self.block_hit_rate, 5),
            "full_hit_rate": round(self.full_hit_rate, 5),
            "blocks_hit": self.blocks_hit,
            "blocks_needed": self.blocks_needed,
            "evictions": self.evictions,
            "regret_rate": round(self.regret_rate, 5),
            "score_outcome_correlation": round(
                self.score_outcome_correlation, 5
            ),
        }


@dataclass(slots=True)
class _Ablation:
    """Which terms of the score are active (08 §7, falsification test)."""

    prob: bool = True
    decay: bool = True
    blocks: bool = True

    @classmethod
    def parse(cls, spec: str) -> "_Ablation":
        ablation = cls()
        for term in filter(None, (s.strip() for s in spec.split(","))):
            if term == "prob":
                ablation.prob = False
            elif term == "decay":
                ablation.decay = False
            elif term == "blocks":
                ablation.blocks = False
            elif term != "none":
                raise ValueError(f"Unknown ablation term: {term}")
        return ablation


class _Forecast:
    """Produces `PROB`-shaped rows for the simulated tick.

    Two modes, and the difference between them is the point:

    - `history` (default) — **causal**. `prob` and `time_to_next_call` come
      only from gaps already observed for that key. This is what the
      Prediction Engine can realistically know, so it measures the design.
    - `oracle` — reads the actual next arrival from the future of the trace.
      This measures the *mechanism* with a perfect forecast, i.e. the ceiling
      the design could ever reach. If the oracle arm does not beat LRU, the
      enforcement path is broken and no amount of prediction work will help.
    """

    def __init__(self, requests: list[ReplayRequest], mode: str) -> None:
        self.mode = mode
        self._next_arrival: dict[str, deque[float]] = {}
        self._gaps: dict[NodeKey, list[float]] = {}
        self._seen: dict[NodeKey, int] = {}
        self._last_finish: dict[NodeKey, float] = {}
        if mode == "oracle":
            for request in requests:
                self._next_arrival.setdefault(
                    _key_str(request.key), deque()
                ).append(request.arrival_ts)

    def observe(self, request: ReplayRequest) -> None:
        key = request.key
        self._seen[key] = self._seen.get(key, 0) + 1
        last_finish = self._last_finish.get(key)
        if last_finish is not None:
            gap_ms = max(0.0, (request.arrival_ts - last_finish) * 1000.0)
            self._gaps.setdefault(key, []).append(gap_ms)
        self._last_finish[key] = request.finish_ts
        if self.mode == "oracle":
            queue = self._next_arrival.get(_key_str(key))
            while queue and queue[0] <= request.arrival_ts:
                queue.popleft()

    def row_for(self, key: NodeKey, now_ts: float) -> ImportanceRow | None:
        if self.mode == "oracle":
            queue = self._next_arrival.get(_key_str(key))
            if not queue:
                # Never runs again. A floored probability rather than zero:
                # 0 makes a prefix the top victim, and the engine treats a
                # missing row as neutral anyway (05 §5.2.1 item 4).
                return ImportanceRow(
                    prob=0.01,
                    time_to_next_call_ms=3_600_000.0,
                    update_ts_ms=now_ts * 1000.0,
                )
            return ImportanceRow(
                prob=1.0,
                time_to_next_call_ms=max(0.0, (queue[0] - now_ts) * 1000.0),
                update_ts_ms=now_ts * 1000.0,
            )

        gaps = self._gaps.get(key)
        if not gaps:
            return None
        # Median gap: the distribution is bimodal by construction and one
        # long idle gap must not make a hot node look permanently unneeded
        # (04 §5.1, §5.3).
        ordered = sorted(gaps)
        median = ordered[len(ordered) // 2]
        last_finish = self._last_finish.get(key, now_ts)
        elapsed_ms = max(0.0, (now_ts - last_finish) * 1000.0)
        ttnc = max(0.0, median - elapsed_ms)
        # Seen n times, so it will very likely fire again; time-free by
        # construction, since the discount is applied to ttnc exactly once
        # (05 §5.2).
        n = self._seen.get(key, 1)
        prob = min(0.99, 1.0 - 1.0 / (n + 1))
        return ImportanceRow(
            prob=prob,
            time_to_next_call_ms=ttnc,
            update_ts_ms=now_ts * 1000.0,
        )


def _key_str(key: NodeKey) -> str:
    return "|".join(key)


class _SimPool:
    """A free queue and a per-key resident prefix, in plain Python.

    Blocks are integers. The free list is a deque used exactly as
    `FreeKVCacheBlockQueue` is: `popleft` allocates, freed blocks go to the
    tail in reverse prefix order (which is what
    `single_type_kv_cache_manager.free` already does), and the policy moves
    the worst-scoring blocks to the head.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.in_free: set[int] = set(range(num_blocks))
        # block -> (key, position); a block belongs to one key here because
        # the trace cannot express prefix sharing across nodes.
        self.owner: dict[int, tuple[NodeKey, int]] = {}
        # key -> position -> block
        self.resident: dict[NodeKey, dict[int, int]] = {}

    def contiguous_hit(self, key: NodeKey, need: int) -> int:
        """Longest run resident from position 0.

        A prefix is only worth anything as a contiguous run from position 0,
        so a hole in the middle truncates the match there.
        """
        positions = self.resident.get(key)
        if not positions:
            return 0
        hit = 0
        while hit < need and hit in positions:
            hit += 1
        return hit

    def _detach(self, block: int) -> tuple[NodeKey, int] | None:
        owned = self.owner.pop(block, None)
        if owned is None:
            return None
        key, position = owned
        positions = self.resident.get(key)
        if positions is not None and positions.get(position) == block:
            del positions[position]
            if not positions:
                del self.resident[key]
        return owned

    def allocate(self, count: int) -> tuple[list[int], list[tuple[int, NodeKey, int]]]:
        """Pop `count` blocks from the head. Returns the blocks and the
        cached ones that were destroyed to get them."""
        blocks: list[int] = []
        evicted: list[tuple[int, NodeKey, int]] = []
        for _ in range(count):
            if not self.free:
                break
            block = self.free.popleft()
            self.in_free.discard(block)
            owned = self._detach(block)
            if owned is not None:
                evicted.append((block, owned[0], owned[1]))
            blocks.append(block)
        return blocks, evicted

    def assign(self, key: NodeKey, blocks: list[int], start_position: int) -> None:
        positions = self.resident.setdefault(key, {})
        for i, block in enumerate(blocks):
            position = start_position + i
            positions[position] = block
            self.owner[block] = (key, position)

    def free_blocks(self, blocks: list[int]) -> None:
        """Return blocks to the tail. Callers pass them tail-of-prefix first,
        matching `free_blocks(reversed(req_blocks))`."""
        for block in blocks:
            if block in self.in_free:
                continue
            self.free.append(block)
            self.in_free.add(block)

    def splice_to_head(self, blocks: list[int]) -> None:
        """Move blocks to the head, preserving the given order.

        `blocks[0]` becomes the next block allocated, so callers pass their
        candidates in ascending score order — the same convention as
        `FreeKVCacheBlockQueue.appendleft_n`.
        """
        selected = [b for b in blocks if b in self.in_free]
        if not selected:
            return
        chosen = set(selected)
        remaining = [b for b in self.free if b not in chosen]
        self.free = deque(selected + remaining)

    @property
    def num_free_fresh(self) -> int:
        return sum(1 for block in self.free if block not in self.owner)


def replay(
    requests: list[ReplayRequest],
    policy: str = "scored",
    num_blocks: int = 4096,
    block_size: int = 16,
    config: NodeEvictionConfig | None = None,
    forecast_mode: str = "history",
    ablation: _Ablation | None = None,
    regret_horizon_s: float = 120.0,
) -> ReplayResult:
    """Run one policy arm over the trace.

    `policy` is `lru` (the counterfactual baseline) or `scored`.
    """
    config = config or NodeEvictionConfig()
    ablation = ablation or _Ablation()
    pool = _SimPool(num_blocks)
    forecast = _Forecast(requests, forecast_mode)
    result = ReplayResult(policy=policy)

    # request finish events, so blocks are held for the request's lifetime
    inflight: list[tuple[float, NodeKey, list[int]]] = []
    # evictions awaiting an outcome: key -> [(position, score, ts)]
    pending: dict[NodeKey, list[tuple[int, float, float]]] = {}

    for request in requests:
        now = request.arrival_ts

        # Release anything that finished before this arrival. Freed in
        # reverse prefix order so the tail of a prompt is evicted before its
        # head, which is the intra-request ordering vLLM already has.
        still_inflight = []
        for finish_ts, key, blocks in inflight:
            if finish_ts <= now:
                pool.free_blocks(list(reversed(blocks)))
            else:
                still_inflight.append((finish_ts, key, blocks))
        inflight = still_inflight

        need = max(1, math.ceil(request.num_prompt_tokens / block_size))
        key = request.key

        # Outcome side of the join: did we evict something this key now
        # wants?
        outstanding = pending.pop(key, [])
        survivors = []
        for position, score, ts in outstanding:
            if now - ts <= regret_horizon_s and position < need:
                result.evictions_regretted += 1
                result.needed_score_sum += score
                result.needed_count += 1
            elif now - ts > regret_horizon_s:
                result.not_needed_score_sum += score
                result.not_needed_count += 1
            else:
                survivors.append((position, score, ts))
        if survivors:
            pending[key] = survivors

        hit = pool.contiguous_hit(key, need)
        # A hit block is rescued from the free queue, exactly as
        # `BlockPool.touch` does.
        hit_blocks = [pool.resident[key][p] for p in range(hit)] if hit else []
        for block in hit_blocks:
            if block in pool.in_free:
                pool.in_free.discard(block)
                try:
                    pool.free.remove(block)
                except ValueError:
                    pass

        missing = need - hit
        new_blocks, evicted = pool.allocate(missing)
        for block, evicted_key, position in evicted:
            result.evictions += 1
            score = _score_of(
                evicted_key, pool, forecast, config, ablation, now
            )
            pending.setdefault(evicted_key, []).append((position, score, now))
        pool.assign(key, new_blocks, hit)

        result.requests += 1
        result.blocks_needed += need
        result.blocks_hit += hit
        if hit >= need:
            result.requests_full_hit += 1
        hits, needs = result.per_key_hit.get(key, (0, 0))
        result.per_key_hit[key] = (hits + hit, needs + need)

        inflight.append((request.finish_ts, key, hit_blocks + new_blocks))
        forecast.observe(request)

        if policy == "scored":
            _tick(pool, forecast, config, ablation, now)

    # Everything still pending was never needed again.
    for outstanding in pending.values():
        for _position, score, _ts in outstanding:
            result.not_needed_score_sum += score
            result.not_needed_count += 1

    return result


def _score_of(
    key: NodeKey,
    pool: _SimPool,
    forecast: _Forecast,
    config: NodeEvictionConfig,
    ablation: _Ablation,
    now: float,
) -> float:
    row = forecast.row_for(key, now)
    if row is None:
        return 0.0
    if not ablation.prob:
        row = ImportanceRow(
            prob=1.0,
            time_to_next_call_ms=row.time_to_next_call_ms,
            update_ts_ms=row.update_ts_ms,
            p_l1=row.p_l1,
            p_cold=row.p_cold,
        )
    if not ablation.decay:
        row = ImportanceRow(
            prob=row.prob,
            time_to_next_call_ms=0.0,
            update_ts_ms=row.update_ts_ms,
            p_l1=row.p_l1,
            p_cold=row.p_cold,
        )
    num_blocks = len(pool.resident.get(key, ()))
    if not ablation.blocks:
        num_blocks = 1
    return score_key(row, max(num_blocks, 1), config).score


def _tick(
    pool: _SimPool,
    forecast: _Forecast,
    config: NodeEvictionConfig,
    ablation: _Ablation,
    now: float,
) -> None:
    """One simulated tick: score, then splice the worst K to the head."""
    if pool.num_free_fresh > config.fresh_skip_threshold:
        # Rule 1: nothing to gain from ordering cached blocks that will not
        # be reached while unused blocks are still queued.
        return

    scores: dict[NodeKey, float] = {}
    for key in pool.resident:
        scores[key] = _score_of(key, pool, forecast, config, ablation, now)

    candidates: list[tuple[float, int, int]] = []
    for block in pool.free:
        owned = pool.owner.get(block)
        if owned is None:
            continue
        key, position = owned
        score = scores.get(key)
        if score is None:
            continue
        if config.score_threshold is not None and score >= config.score_threshold:
            continue
        # Ascending score, then tail-of-prefix first within a key.
        candidates.append((score, -position, block))

    if not candidates:
        return
    candidates.sort()
    pool.splice_to_head(
        [block for _score, _neg_pos, block in candidates[: config.splice_max_blocks]]
    )


def compare(
    requests: list[ReplayRequest],
    num_blocks: int,
    block_size: int,
    config: NodeEvictionConfig,
    forecast_mode: str,
    ablation: _Ablation,
) -> list[ReplayResult]:
    """Both arms on the identical trace. This is the whole point (07 §8)."""
    return [
        replay(
            requests,
            policy="lru",
            num_blocks=num_blocks,
            block_size=block_size,
            config=config,
            forecast_mode=forecast_mode,
            ablation=ablation,
        ),
        replay(
            requests,
            policy="scored",
            num_blocks=num_blocks,
            block_size=block_size,
            config=config,
            forecast_mode=forecast_mode,
            ablation=ablation,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a vLLM request trace against LRU and the node-aware "
            "eviction policy. No GPU, no model, no Redis."
        )
    )
    parser.add_argument(
        "--trace",
        nargs="+",
        required=True,
        help="FileStatLogger JSONL path(s); globs allowed",
    )
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--policy", choices=["lru", "scored"], default="scored"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run both arms and report the delta",
    )
    parser.add_argument(
        "--forecast",
        choices=["history", "oracle"],
        default="history",
        help=(
            "history: causal, what the Prediction Engine could know. "
            "oracle: perfect foresight, i.e. the ceiling of the design."
        ),
    )
    parser.add_argument(
        "--ablate",
        default="none",
        help="comma-separated terms to disable: prob, decay, blocks",
    )
    parser.add_argument("--tau-ms", type=float, default=None)
    parser.add_argument("--splice-k", type=int, default=None)
    parser.add_argument(
        "--config", default=None, help="JSON file of NodeEvictionConfig fields"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.config:
        from vllm.v1.core.node_eviction.config import load_config_from_file

        config = load_config_from_file(args.config)
    else:
        config = NodeEvictionConfig()
    if args.tau_ms is not None:
        config.tau_ms = args.tau_ms
    if args.splice_k is not None:
        config.splice_max_blocks = args.splice_k
    config.enabled = True

    ablation = _Ablation.parse(args.ablate)
    requests = load_trace(args.trace, use_call_type=config.use_call_type)
    if not requests:
        print("No usable records: does the trace carry job_id/langgraph_node?")
        return 1

    if args.compare:
        results = compare(
            requests,
            args.num_blocks,
            args.block_size,
            config,
            args.forecast,
            ablation,
        )
    else:
        results = [
            replay(
                requests,
                policy=args.policy,
                num_blocks=args.num_blocks,
                block_size=args.block_size,
                config=config,
                forecast_mode=args.forecast,
                ablation=ablation,
            )
        ]

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
        return 0

    print(f"trace: {len(requests)} requests, {args.num_blocks} blocks "
          f"x {args.block_size} tokens, forecast={args.forecast}")
    print()
    header = f"{'policy':>8}  {'block hit':>10}  {'full hit':>9}  " \
             f"{'evictions':>9}  {'regret':>7}  {'score/outcome':>13}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.policy:>8}  {result.block_hit_rate:>9.1%}  "
            f"{result.full_hit_rate:>8.1%}  {result.evictions:>9d}  "
            f"{result.regret_rate:>6.1%}  "
            f"{result.score_outcome_correlation:>13.3f}"
        )

    if len(results) == 2:
        lru, scored = results
        delta = scored.block_hit_rate - lru.block_hit_rate
        print()
        print(f"delta vs LRU: {delta:+.1%} block hit rate")
        if delta < 0:
            print(
                "  WORSE THAN LRU on this trace. Check the score/outcome "
                "correlation: if it is negative the policy is anti-correlated "
                "and the sign is inverted somewhere."
            )
        if scored.score_outcome_correlation < 0:
            print(
                "  NEGATIVE score/outcome correlation: blocks that were "
                "needed again scored lower than blocks that were not."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
