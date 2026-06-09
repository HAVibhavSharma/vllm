# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-robin demo: LRU vs agent-aware early eviction.

Scenario A from ``plan/lru_vs_agent_eviction.md``. Drives four agents
through a strict A -> B -> C -> D rotation against a deliberately
undersized prefix cache. In ``--mode baseline`` no probability hints are
sent (server falls back to LRU); in ``--mode treatment`` each request
forecasts the next agent in the rotation at probability 1.0 so the
agent-aware policy can keep its cache pinned.

The script prints per-call TTFT, a summary table, and a final snapshot of
``GET /v1/agents/eviction_stats`` so you can see the policy's view of the
world.

See the sibling ``README.md`` for the full setup (server flags, expected
results, knobs).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from prompts import build_rotation, build_system_prompt, pick_user_query


# ---------------------------------------------------------------------------
# Constants and small helpers.
# ---------------------------------------------------------------------------


DEFAULT_HIT_TTFT_THRESHOLD_MS = 200.0


def _post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_stream(url: str, payload: dict, timeout: float = 600.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Prefetch helper.
# ---------------------------------------------------------------------------


def prefetch_agent(
    base_url: str,
    agent_id: str,
    *,
    wait: bool = False,
    prefetch_top_k: int | None = None,
) -> dict | None:
    """POST /v1/agents/prefetch for ``agent_id``. Errors are swallowed.

    Returns the response body on success, ``None`` on failure. Prefetch
    is best-effort -- the benchmark must continue even if warming fails.
    """
    url = f"{base_url.rstrip('/')}/v1/agents/prefetch"
    payload: dict = {"agent_id": agent_id, "wait": wait}
    if prefetch_top_k is not None:
        payload["prefetch_top_k"] = prefetch_top_k
    try:
        return _post_json(url, payload, timeout=60.0)
    except urllib.error.HTTPError as e:
        print(
            f"  (prefetch agent={agent_id} failed: HTTP {e.code} {e.reason})",
            file=sys.stderr,
        )
    except urllib.error.URLError as e:
        print(
            f"  (prefetch agent={agent_id} failed: {e.reason})",
            file=sys.stderr,
        )
    return None


# ---------------------------------------------------------------------------
# Forecast construction.
# ---------------------------------------------------------------------------


def build_probabilities(
    current_agent: str,
    previous_agent: str | None,
    rotation: list[str],
) -> dict[str, float]:
    """Distance-aware forecast for round-robin firing order.

    For a strict round-robin of N agents, each agent's distance from the
    current one is well-defined: dist=1 is the next to fire, dist=N-1
    is the one that just fired. We give every agent a probability that
    linearly decreases with cycle distance:

        prob(dist) = (N - dist) / (N - 1)

    So:
        dist=1 (next)   -> 1.0   (must protect — fires imminently)
        dist=N-1 (prev) -> 0.0   (won't fire again until full cycle)
        dist in between -> linear ramp

    The naive "previous=0, next=1, rest=0.5" forecast we used before
    caused thrashing: when each agent fired it evicted the *just-fired*
    one — even with cache room to spare — and by the time the cycle
    came back around N-1 turns later, that agent's blocks were gone.
    The distance-aware ramp instead reflects the true cycle structure:
    only the agents *farthest* from re-firing become evictable, so the
    cache holds the upcoming ones.

    With the default threshold of 0.5, agents whose probability is
    strictly below 0.5 become evictable — that means the most-recently
    fired half of the cycle becomes the eviction pool.

    The current agent doesn't need to be in the dict — the policy
    implicitly self-votes 1.0 for its own ``agent_id``.
    """
    del previous_agent  # no longer needed; distance encodes it
    n = len(rotation)
    if n < 2:
        return {}
    idx = rotation.index(current_agent)
    probs: dict[str, float] = {}
    for i, agent in enumerate(rotation):
        if agent == current_agent:
            continue  # self, auto-protected by the policy
        # Cycle distance in firing order: dist=1 is next, dist=N-1 is
        # the one that just fired. Always in [1, N-1].
        dist = (i - idx) % n
        probs[agent] = (n - dist) / (n - 1)
    return probs


# ---------------------------------------------------------------------------
# One streaming call -> CallResult.
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    call_idx: int           # 1-based across the whole run
    round_idx: int          # 1-based round
    agent_id: str
    ttft_ms: float
    total_ms: float
    output_tokens: int
    preview: str
    error: str | None = None

    def is_hit(self, hit_ttft_ms: float) -> bool:
        if self.error is not None:
            return False
        return self.ttft_ms < hit_ttft_ms


def call_once(
    base_url: str,
    model: str,
    agent_id: str,
    system_prompt: str,
    user_query: str,
    max_tokens: int,
    *,
    probabilities: dict[str, float] | None,
    eviction_window: int | None,
    eviction_threshold: float | None,
) -> tuple[float, float, int, str, str | None]:
    """Issue one chat call and time TTFT + total wall time.

    Returns ``(ttft_ms, total_ms, output_tokens, preview, error)``.
    ``error`` is ``None`` on success.
    """
    # Use the agent-scoped chat endpoint so the server records this
    # prompt's chunk-aligned prefix in the per-agent registry. That
    # registry is what /v1/agents/prefetch reads from to warm APC for
    # the upcoming agent in the rotation.
    url = f"{base_url.rstrip('/')}/v1/agents/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "agent_id": agent_id,
    }
    if probabilities is not None:
        payload["agent_probabilities"] = probabilities
        if eviction_window is not None:
            payload["eviction_window"] = eviction_window
        if eviction_threshold is not None:
            payload["eviction_threshold"] = eviction_threshold

    start_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    pieces: list[str] = []
    output_tokens = 0
    try:
        for chunk in _post_stream(url, payload):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content is None:
                continue
            if first_token_ns is None:
                first_token_ns = time.perf_counter_ns()
            pieces.append(content)
            output_tokens += 1
    except urllib.error.HTTPError as e:
        msg = f"HTTPError {e.code}: {e.reason}"
        return float("nan"), float("nan"), 0, "", msg
    except urllib.error.URLError as e:
        return float("nan"), float("nan"), 0, "", f"URLError: {e.reason}"
    end_ns = time.perf_counter_ns()

    if first_token_ns is None:
        return float("nan"), (end_ns - start_ns) / 1e6, 0, "", "no tokens"

    ttft_ms = (first_token_ns - start_ns) / 1e6
    total_ms = (end_ns - start_ns) / 1e6
    preview = "".join(pieces).strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:77] + "..."
    return ttft_ms, total_ms, output_tokens, preview, None


# ---------------------------------------------------------------------------
# Run loop.
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    mode: str
    rounds: int
    warm_up_rounds: int
    hit_threshold_ms: float
    results: list[CallResult] = field(default_factory=list)

    def scored(self) -> list[CallResult]:
        """Results from after the warm-up rounds (used for stats)."""
        first_scored_round = self.warm_up_rounds + 1
        return [r for r in self.results if r.round_idx >= first_scored_round]


def run(
    base_url: str,
    model: str,
    rotation: list[str],
    rounds: int,
    *,
    mode: str,
    max_tokens: int,
    warm_up_rounds: int,
    hit_threshold_ms: float,
    eviction_window: int | None,
    eviction_threshold: float | None,
    filler_lines: int,
) -> RunSummary:
    summary = RunSummary(
        mode=mode,
        rounds=rounds,
        warm_up_rounds=warm_up_rounds,
        hit_threshold_ms=hit_threshold_ms,
    )
    system_prompts = {a: build_system_prompt(a, filler_lines) for a in rotation}
    previous_agent: str | None = None
    call_idx = 0

    for round_idx in range(1, rounds + 1):
        for slot_idx, agent_id in enumerate(rotation):
            call_idx += 1
            user_query = pick_user_query(agent_id, round_idx - 1)
            probs: dict[str, float] | None
            if mode == "treatment":
                probs = build_probabilities(agent_id, previous_agent, rotation)
            else:
                probs = None
            ttft_ms, total_ms, out_tokens, preview, err = call_once(
                base_url=base_url,
                model=model,
                agent_id=agent_id,
                system_prompt=system_prompts[agent_id],
                user_query=user_query,
                max_tokens=max_tokens,
                probabilities=probs,
                eviction_window=eviction_window,
                eviction_threshold=eviction_threshold,
            )
            result = CallResult(
                call_idx=call_idx,
                round_idx=round_idx,
                agent_id=agent_id,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                output_tokens=out_tokens,
                preview=preview,
                error=err,
            )
            summary.results.append(result)
            is_scored = round_idx > warm_up_rounds
            tag = "scored" if is_scored else "warmup"
            hit = "HIT" if result.is_hit(hit_threshold_ms) else "miss"
            if err is None:
                print(
                    f"[{tag:>6}] call={call_idx:03d} round={round_idx} "
                    f"slot={slot_idx} agent={agent_id} "
                    f"ttft={ttft_ms:7.1f}ms total={total_ms:7.1f}ms "
                    f"out={out_tokens:3d} {hit:4s} | {preview}"
                )
            else:
                print(
                    f"[{tag:>6}] call={call_idx:03d} round={round_idx} "
                    f"slot={slot_idx} agent={agent_id} ERROR: {err}",
                    file=sys.stderr,
                )
            # Fire-and-forget prefetch for the next agent in the
            # rotation, scored rounds only. wait=false lets the phantom
            # warm in the background while we set up the next HTTP call.
            if is_scored:
                next_agent = rotation[(slot_idx + 1) % len(rotation)]
                prefetch_agent(base_url, next_agent, wait=False)
            previous_agent = agent_id
    return summary


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def print_summary(summary: RunSummary) -> None:
    scored = [r for r in summary.scored() if r.error is None]
    print()
    print("=" * 76)
    print(f"Mode: {summary.mode}   rounds: {summary.rounds}   "
          f"warm-up: {summary.warm_up_rounds}   "
          f"hit threshold: {summary.hit_threshold_ms:.0f}ms")
    print("=" * 76)
    if not scored:
        print("(no scored results -- bump --rounds or check for errors)")
        return

    ttfts = [r.ttft_ms for r in scored]
    hits = [r for r in scored if r.is_hit(summary.hit_threshold_ms)]
    overall_rate = 100.0 * len(hits) / len(scored)
    print(
        f"Overall: calls={len(scored)} hit_rate={overall_rate:5.1f}% "
        f"mean_ttft={statistics.mean(ttfts):.1f}ms "
        f"median_ttft={statistics.median(ttfts):.1f}ms "
        f"p95_ttft={_percentile(ttfts, 0.95):.1f}ms"
    )
    print()
    print("Per-agent breakdown:")
    print(
        f"  {'agent':<10} {'n':>4} {'hit_rate':>9} "
        f"{'mean_ttft':>10} {'p95_ttft':>10}"
    )
    by_agent: dict[str, list[CallResult]] = {}
    for r in scored:
        by_agent.setdefault(r.agent_id, []).append(r)
    for agent_id in sorted(by_agent):
        rows = by_agent[agent_id]
        rate = 100.0 * sum(
            1 for r in rows if r.is_hit(summary.hit_threshold_ms)
        ) / len(rows)
        agent_ttfts = [r.ttft_ms for r in rows]
        print(
            f"  {agent_id:<10} {len(rows):>4d} {rate:>8.1f}% "
            f"{statistics.mean(agent_ttfts):>9.1f}ms "
            f"{_percentile(agent_ttfts, 0.95):>9.1f}ms"
        )


def print_eviction_stats(base_url: str) -> None:
    url = f"{base_url.rstrip('/')}/v1/agents/eviction_stats"
    try:
        body = _get_json(url)
    except urllib.error.HTTPError as e:
        print(f"\n(could not fetch eviction stats: HTTP {e.code} {e.reason})",
              file=sys.stderr)
        return
    except urllib.error.URLError as e:
        print(f"\n(could not fetch eviction stats: {e.reason})", file=sys.stderr)
        return
    print()
    print("Server-side policy snapshot (GET /v1/agents/eviction_stats):")
    print(json.dumps(body, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Round-robin demo for agent-aware early eviction.",
    )
    p.add_argument("--base-url", required=True,
                   help="vLLM server base URL, e.g. http://localhost:8000")
    p.add_argument("--model", required=True,
                   help="Model id served by vLLM, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", required=True, choices=["baseline", "treatment"],
                   help="baseline = no probabilities (pure LRU); "
                        "treatment = send agent_probabilities forecast")
    p.add_argument("--rounds", type=int, default=5,
                   help="Number of full rotations through every agent (default: 5)")
    p.add_argument("--num-agents", type=int, default=4,
                   help="Number of distinct agents in the rotation. Scale "
                        "this up on large GPUs (e.g. 16-32 on an H100 80GB) "
                        "so the prefix cache cannot hold every agent's "
                        "prompt at once -- otherwise there is no eviction "
                        "pressure and baseline and treatment tie. "
                        "Default: 4. Minimum: 2.")
    p.add_argument("--warm-up-rounds", type=int, default=1,
                   help="Initial rounds excluded from summary stats (default: 1)")
    p.add_argument("--hit-threshold-ms", type=float,
                   default=DEFAULT_HIT_TTFT_THRESHOLD_MS,
                   help=f"TTFT below this (ms) counts as a cache hit "
                        f"(default: {DEFAULT_HIT_TTFT_THRESHOLD_MS:.0f})")
    p.add_argument("--max-output-tokens", type=int, default=16,
                   help="max_tokens per call (default: 16)")
    p.add_argument("--filler-lines", type=int, default=220,
                   help="Lines of filler in each agent's system prompt; "
                        "controls prompt length (default: 220 ~ ~4K tokens)")
    p.add_argument("--eviction-window", type=int, default=None,
                   help="Optional eviction_window (treatment mode only)")
    p.add_argument("--eviction-threshold", type=float, default=None,
                   help="Optional eviction_threshold (treatment mode only)")
    p.add_argument("--skip-eviction-stats", action="store_true",
                   help="Don't fetch /v1/agents/eviction_stats at the end")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.warm_up_rounds >= args.rounds:
        print(
            f"error: --warm-up-rounds ({args.warm_up_rounds}) must be less "
            f"than --rounds ({args.rounds}); nothing would be scored.",
            file=sys.stderr,
        )
        return 2
    if args.num_agents < 2:
        print(
            f"error: --num-agents must be >= 2 (got {args.num_agents}); "
            "a rotation needs at least two distinct prompts.",
            file=sys.stderr,
        )
        return 2

    rotation = build_rotation(args.num_agents)
    print(
        f"rotation: {len(rotation)} agents "
        f"({rotation[0]} ... {rotation[-1]})"
    )

    summary = run(
        base_url=args.base_url,
        model=args.model,
        rotation=rotation,
        rounds=args.rounds,
        mode=args.mode,
        max_tokens=args.max_output_tokens,
        warm_up_rounds=args.warm_up_rounds,
        hit_threshold_ms=args.hit_threshold_ms,
        eviction_window=args.eviction_window,
        eviction_threshold=args.eviction_threshold,
        filler_lines=args.filler_lines,
    )
    print_summary(summary)
    if not args.skip_eviction_stats:
        print_eviction_stats(args.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
