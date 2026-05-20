# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demo workflow exercising the agent-scoped prefix prefetch endpoint.

Simulates a multi-agent pipeline that walks ``agent1 -> agent2 -> agent3``
in a fixed order. Each agent owns multiple long, stable preamble
variants and the workflow rotates through them per call. The point of
the demo is to expose the *cold-prefill cost* that the prefetch
endpoint is designed to eliminate -- and to make that cost large
enough to actually show up in TTFT.

Three modes:

  --mode baseline   POST to /v1/chat/completions only. The first time
                    each (agent, variant) is seen, you pay full cold
                    prefill. Subsequent visits hit vLLM APC.

  --mode warmup     Issue a dummy max_tokens=1 warmup before each call.
                    The "old way" -- pays one prefill per warmup.

  --mode prefetch   Two-step explicit warm + call:
                    1) POST /v1/agents/prefetch with
                       {agent_id, prefetch_top_k, wait: true} so the
                       server warms APC from LMCache CPU L1.
                    2) POST /v1/agents/chat/completions with
                       {agent_id, messages} so the chat handler runs
                       and the prefix is recorded for future warms.

To make the cost visible, dial up preamble length:

  python examples/online_serving/agent_prefetch_workflow.py \\
      --base-url http://<vllm-host>:8000 \\
      --model Qwen/Qwen3-8B \\
      --mode baseline \\
      --preamble-lines 1000 \\
      --variants-per-agent 4 \\
      --rounds 2

  # ~ 20K-token preambles, 4 variants/agent, 2 rounds.

The summary at the end splits TTFT into "first-visit" and "revisit" so
the cold tax is unmistakable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


# TTFT threshold (ms) under which a call is considered a "cache hit".
# The observed warm floor on Qwen3-8B is ~18-70 ms depending on prefix
# length; cold prefill of a 22K-token preamble is ~900 ms. 200 ms is
# comfortably between the two bands.
HIT_TTFT_THRESHOLD_MS = 200.0


# ---------------------------------------------------------------------------
# Preamble construction.
# ---------------------------------------------------------------------------


_LINE_TEMPLATE = (
    "(variant-{variant} line {i}: maintain prior conventions, "
    "observe the established style, avoid speculation, defer to the "
    "system designer's choices, and keep responses tightly scoped.)"
)


def build_preamble(role: str, variant: int, lines: int) -> str:
    """Stable per-(role, variant) preamble.

    Distinct ``(role, variant)`` pairs produce distinct prefixes (the
    leading sentinel differs), so APC/LMCache treat them as separate
    cache entries.
    """
    header = (
        f"You are {role} (persona variant {variant}), a careful and "
        "verbose assistant. Always think step by step before answering. "
        "Cite sources only when relevant. "
    )
    filler = " ".join(
        _LINE_TEMPLATE.format(variant=variant, i=i) for i in range(lines)
    )
    return header + filler


def estimate_tokens(text: str) -> int:
    """Cheap upper-bound estimate: ~4 chars/token for English."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Workflow definition. Queries are short and per-agent; the prefix being
# cached is the system preamble, which the workflow rotates among
# `variants_per_agent` distinct variants.
# ---------------------------------------------------------------------------


AGENT_QUERIES: dict[str, list[str]] = {
    "agent1": [
        "Summarize the goal of LMCache in one sentence.",
        "List two trade-offs of CPU-resident KV caches.",
        "Why is chunked prefill relevant here?",
    ],
    "agent2": [
        "Critique the previous summary for vagueness.",
        "Where would prefix mismatch most likely happen?",
    ],
    "agent3": [
        "Combine the prior outputs into a 3-bullet plan.",
        "Estimate the latency saving from APC hits.",
        "Propose one observability metric to track this.",
        "Suggest a stress test design.",
    ],
}

WORKFLOW_ORDER: list[str] = ["agent1", "agent2", "agent3"]


# ---------------------------------------------------------------------------
# Minimal HTTP client.
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    agent_id: str
    variant: int
    query_idx: int
    first_visit: bool
    ttft_ms: float
    total_ms: float
    apc_hit_tokens: int | None
    lmcache_extra_tokens: int | None
    response_preview: str
    round_idx: int = 0
    call_idx: int = 0  # 1-based index in the full predetermined schedule
    # In prefetch mode: ms spent in the explicit /v1/agents/prefetch
    # call that precedes the chat. NaN for other modes.
    prefetch_ms: float = float("nan")
    prefetch_submitted: int = 0
    prefetch_completed: int = 0
    available_prefixes: int = 0

    @property
    def inferred_hit(self) -> bool:
        """True if TTFT suggests this call hit the prefix cache.

        TTFT below ``HIT_TTFT_THRESHOLD_MS`` means the long preamble's
        prefill was skipped -- either APC was warm or the prefetch
        endpoint warmed it. If TTFT is NaN (HTTP error), reported as
        False.
        """
        return self.ttft_ms == self.ttft_ms and self.ttft_ms < HIT_TTFT_THRESHOLD_MS


def _post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_stream(url: str, payload: dict, timeout: float = 600.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
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
                yield (time.perf_counter_ns(), None)
                return
            try:
                yield (time.perf_counter_ns(), json.loads(data))
            except json.JSONDecodeError:
                continue


def _build_chat_payload(model: str, system: str, query: str,
                       max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }


# ---------------------------------------------------------------------------
# Per-mode call functions.
# ---------------------------------------------------------------------------


def call_baseline(base_url: str, model: str, system: str,
                  query: str, max_tokens: int) -> tuple[float, float, str,
                                                        dict | None]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = _build_chat_payload(model, system, query, max_tokens=max_tokens)
    return _run_streaming(url, payload)


def call_warmup_then_real(base_url: str, model: str, system: str,
                          query: str, max_tokens: int
                          ) -> tuple[float, float, str, dict | None]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    warmup_payload = _build_chat_payload(
        model, system, query="warmup", max_tokens=1)
    warmup_payload["stream"] = False
    try:
        _post_json(url, warmup_payload)
    except urllib.error.HTTPError as e:
        print(f"  [warn] warmup failed: {e}", file=sys.stderr)
    payload = _build_chat_payload(model, system, query, max_tokens=max_tokens)
    return _run_streaming(url, payload)


def call_prefetch_warm(base_url: str, agent_id: str,
                       prefetch_top_k: int | None,
                       wait: bool = True
                       ) -> tuple[float, dict | None]:
    """POST to /v1/agents/prefetch and time the round-trip.

    Returns ``(elapsed_ms, response_body | None)``. ``response_body``
    is the parsed JSON metadata the server returns (counts of
    submitted/completed phantoms). ``None`` on HTTP error.

    ``prefetch_top_k=None`` (or omitted) tells the server to warm every
    prefix the registry has stored for this agent.
    """
    url = f"{base_url.rstrip('/')}/v1/agents/prefetch"
    payload: dict = {"agent_id": agent_id, "wait": wait}
    if prefetch_top_k is not None:
        payload["prefetch_top_k"] = prefetch_top_k
    start_ns = time.perf_counter_ns()
    try:
        body = _post_json(url, payload)
    except urllib.error.HTTPError as e:
        print(f"  [warn] prefetch failed for {agent_id}: {e}",
              file=sys.stderr)
        return float("nan"), None
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
    return elapsed_ms, body


def call_prefetch(base_url: str, model: str, agent_id: str, system: str,
                  query: str, max_tokens: int,
                  prefetch_top_k: int | None
                  ) -> tuple[float, float, str, dict | None,
                             float, dict | None]:
    """Two-step prefetch + chat call.

    Returns ``(chat_ttft_ms, chat_total_ms, preview, chat_prefetch_meta,
    prefetch_ms, prefetch_response_body)`` so the caller can attribute
    timing to the warm vs. the chat phase.
    """
    prefetch_ms, prefetch_body = call_prefetch_warm(
        base_url, agent_id=agent_id, prefetch_top_k=prefetch_top_k,
        wait=True,
    )
    chat_url = f"{base_url.rstrip('/')}/v1/agents/chat/completions"
    payload = _build_chat_payload(model, system, query, max_tokens=max_tokens)
    payload["agent_id"] = agent_id
    ttft_ms, total_ms, preview, meta = _run_streaming(chat_url, payload)
    return ttft_ms, total_ms, preview, meta, prefetch_ms, prefetch_body


def _run_streaming(url: str, payload: dict
                   ) -> tuple[float, float, str, dict | None]:
    """Returns (ttft_ms, total_ms, preview_text, prefetch_meta)."""
    start_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    text_chunks: list[str] = []
    prefetch_meta: dict | None = None

    try:
        for evt_ns, chunk in _post_stream(url, payload):
            if chunk is None:
                break
            if isinstance(chunk, dict) and "prefetch_meta" in chunk:
                prefetch_meta = chunk["prefetch_meta"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_ns is None:
                        first_token_ns = evt_ns
                    text_chunks.append(content)
    except urllib.error.HTTPError as e:
        return float("nan"), float("nan"), f"<HTTP {e.code}: {e.reason}>", None

    end_ns = time.perf_counter_ns()
    ttft_ms = (
        (first_token_ns - start_ns) / 1e6
        if first_token_ns is not None else float("nan")
    )
    total_ms = (end_ns - start_ns) / 1e6
    preview = "".join(text_chunks)[:80].replace("\n", " ")
    return ttft_ms, total_ms, preview, prefetch_meta


# ---------------------------------------------------------------------------
# Workflow driver.
# ---------------------------------------------------------------------------


def _build_schedule(
    rounds: int, variants_per_agent: int
) -> list[tuple[int, int, str, int, int, str]]:
    """Return the full deterministic call schedule.

    Each entry is (call_idx, round_idx, agent_id, variant, query_idx,
    query_text) -- exactly the order in which run_workflow will issue
    calls. Useful for printing the plan up-front and for downstream
    plotting.
    """
    schedule: list[tuple[int, int, str, int, int, str]] = []
    call_idx = 0
    for round_idx in range(rounds):
        for agent_id in WORKFLOW_ORDER:
            for q_idx, query in enumerate(AGENT_QUERIES[agent_id]):
                variant = q_idx % variants_per_agent
                call_idx += 1
                schedule.append(
                    (call_idx, round_idx, agent_id, variant, q_idx, query)
                )
    return schedule


def _print_schedule(
    schedule: list[tuple[int, int, str, int, int, str]],
) -> None:
    """Pre-flight: announce the exact sequence the workflow will execute.

    This is the "predetermined path" -- everything below is fully
    decided before any HTTP call goes out.
    """
    print("=== Predetermined workflow path ===")
    current_round = -1
    for call_idx, round_idx, agent_id, variant, q_idx, query in schedule:
        if round_idx != current_round:
            current_round = round_idx
            print(f"  Round {round_idx + 1}:")
        # Truncate the query so the path stays scannable.
        short = query if len(query) <= 56 else query[:53] + "..."
        print(
            f"    [{call_idx:03d}] {agent_id} v{variant} q{q_idx}"
            f"  | {short}"
        )
    print(f"  Total: {len(schedule)} calls\n")


def run_workflow(args: argparse.Namespace) -> int:
    # Pre-build all preambles. Cost amortized; also useful for size print.
    preambles: dict[tuple[str, int], str] = {}
    for agent_id in AGENT_QUERIES:
        for variant in range(args.variants_per_agent):
            preambles[(agent_id, variant)] = build_preamble(
                agent_id, variant, args.preamble_lines)

    sample_text = next(iter(preambles.values()))
    est_tokens = estimate_tokens(sample_text)

    print(
        f"Mode: {args.mode} | rounds: {args.rounds} | "
        f"variants/agent: {args.variants_per_agent} | "
        f"preamble_lines: {args.preamble_lines} "
        f"(~{est_tokens} tokens per preamble) | "
        f"top_k: {args.prefetch_top_k if args.prefetch_top_k is not None else 'all'}"
    )
    print(f"Target: {args.base_url} | model: {args.model}")
    if args.sleep_between_rounds > 0:
        print(f"Sleep between rounds: {args.sleep_between_rounds}s")
    print()

    schedule = _build_schedule(args.rounds, args.variants_per_agent)
    _print_schedule(schedule)

    results: list[CallResult] = []
    seen_pairs: set[tuple[str, int]] = set()
    current_round = -1

    for call_idx, round_idx, agent_id, variant, q_idx, query in schedule:
        if round_idx != current_round:
            if current_round >= 0 and args.sleep_between_rounds > 0:
                print(f"  ...sleeping {args.sleep_between_rounds}s before "
                      f"round {round_idx + 1}...")
                time.sleep(args.sleep_between_rounds)
            print(f"=== Round {round_idx + 1} ===")
            current_round = round_idx

        pair = (agent_id, variant)
        first_visit = pair not in seen_pairs
        seen_pairs.add(pair)

        system = preambles[pair]

        prefetch_ms = float("nan")
        prefetch_body: dict | None = None

        if args.mode == "baseline":
            ttft, total, preview, meta = call_baseline(
                args.base_url, args.model, system, query,
                args.max_tokens)
        elif args.mode == "warmup":
            ttft, total, preview, meta = call_warmup_then_real(
                args.base_url, args.model, system, query,
                args.max_tokens)
        elif args.mode == "prefetch":
            (ttft, total, preview, meta,
             prefetch_ms, prefetch_body) = call_prefetch(
                args.base_url, args.model, agent_id, system, query,
                args.max_tokens, prefetch_top_k=args.prefetch_top_k)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        r = CallResult(
            agent_id=agent_id,
            variant=variant,
            query_idx=q_idx,
            first_visit=first_visit,
            ttft_ms=ttft,
            total_ms=total,
            apc_hit_tokens=(meta or {}).get("apc_hit_tokens"),
            lmcache_extra_tokens=(meta or {}).get("lmcache_extra_tokens"),
            response_preview=preview,
            round_idx=round_idx,
            call_idx=call_idx,
            prefetch_ms=prefetch_ms,
            prefetch_submitted=(prefetch_body or {}).get("submitted", 0),
            prefetch_completed=(prefetch_body or {}).get("completed", 0),
            available_prefixes=(prefetch_body or {}).get(
                "available_prefixes", 0),
        )
        results.append(r)

        first_visit_tag = "COLD" if first_visit else "warm"
        hit_tag = "HIT " if r.inferred_hit else "miss"
        extra = ""
        if r.apc_hit_tokens is not None:
            extra = (
                f" | apc={r.apc_hit_tokens} "
                f"lmc_extra={r.lmcache_extra_tokens}"
            )
        prefetch_tag = ""
        if args.mode == "prefetch":
            if prefetch_ms == prefetch_ms:  # not NaN
                prefetch_tag = (
                    f" prefetch={prefetch_ms:7.1f}ms"
                    f"(sub={r.prefetch_submitted}/"
                    f"done={r.prefetch_completed}/"
                    f"avail={r.available_prefixes})"
                )
            else:
                prefetch_tag = " prefetch=<err>"
        print(
            f"  [{call_idx:03d}] {agent_id} v{variant} q{q_idx} "
            f"[{first_visit_tag:4s}|{hit_tag}]:"
            f"{prefetch_tag} ttft={ttft:8.1f}ms total={total:8.1f}ms"
            f"{extra}  | {preview}"
        )
    print()

    _print_summary(results)

    if args.plot:
        try:
            _plot_results(results, args)
        except ImportError as e:
            print(
                f"\n[warn] --plot was requested but plotting deps are "
                f"missing: {e}\n"
                f"       Install with: pip install matplotlib",
                file=sys.stderr,
            )

    nan_count = sum(1 for r in results if r.ttft_ms != r.ttft_ms)
    return 1 if nan_count else 0


def _print_summary(results: list[CallResult]) -> None:
    cold_ttfts = [r.ttft_ms for r in results
                  if r.first_visit and r.ttft_ms == r.ttft_ms]
    warm_ttfts = [r.ttft_ms for r in results
                  if not r.first_visit and r.ttft_ms == r.ttft_ms]

    print("=== Summary ===")
    if cold_ttfts:
        print(
            f"  first-visit (cold): "
            f"n={len(cold_ttfts):3d}  "
            f"median={statistics.median(cold_ttfts):8.1f}ms  "
            f"min={min(cold_ttfts):8.1f}  max={max(cold_ttfts):8.1f}"
        )
    if warm_ttfts:
        print(
            f"  revisit     (warm): "
            f"n={len(warm_ttfts):3d}  "
            f"median={statistics.median(warm_ttfts):8.1f}ms  "
            f"min={min(warm_ttfts):8.1f}  max={max(warm_ttfts):8.1f}"
        )
    if cold_ttfts and warm_ttfts:
        delta = statistics.median(cold_ttfts) - statistics.median(warm_ttfts)
        print(
            f"  cold tax (median first-visit - median revisit): "
            f"{delta:+.1f}ms"
        )
        print(
            "  ^ this is the per-call cold-prefill cost that a working "
            "prefetch endpoint would eliminate."
        )

    # Inferred cache-hit rate based on TTFT vs HIT_TTFT_THRESHOLD_MS.
    valid = [r for r in results if r.ttft_ms == r.ttft_ms]
    if valid:
        hits = [r for r in valid if r.inferred_hit]
        misses = [r for r in valid if not r.inferred_hit]
        print(
            f"  inferred cache-hit rate "
            f"(TTFT < {HIT_TTFT_THRESHOLD_MS:.0f}ms): "
            f"{len(hits)}/{len(valid)} "
            f"({100.0 * len(hits) / len(valid):5.1f}%)"
        )

        # Break down hits / misses by first-visit so a working prefetch
        # endpoint shows as "first-visit calls that are still hits".
        first_visit_hits = sum(1 for r in hits if r.first_visit)
        revisit_misses = sum(1 for r in misses if not r.first_visit)
        if first_visit_hits:
            print(
                f"  first-visit HITS (prefetch worked): "
                f"{first_visit_hits}"
            )
        if revisit_misses:
            print(
                f"  revisit MISSES (cache evicted?): "
                f"{revisit_misses}"
            )

    # Prefetch round-trip stats (separate column -- only meaningful in
    # --mode prefetch). Lets the operator see how much of the wallclock
    # is being spent in the explicit warm call vs. the chat call.
    prefetch_ms_values = [r.prefetch_ms for r in results
                          if r.prefetch_ms == r.prefetch_ms]
    if prefetch_ms_values:
        no_op = sum(1 for r in results
                    if r.prefetch_ms == r.prefetch_ms
                    and r.prefetch_submitted == 0)
        print(
            f"  prefetch round-trip (n={len(prefetch_ms_values):3d}): "
            f"median={statistics.median(prefetch_ms_values):8.1f}ms  "
            f"min={min(prefetch_ms_values):8.1f}  "
            f"max={max(prefetch_ms_values):8.1f}  "
            f"no_op_calls={no_op}"
        )


def _plot_results(results: list[CallResult],
                  args: argparse.Namespace) -> None:
    """Render two PNGs:

    1. <plot>.timeline.png -- TTFT per call across the schedule, colored
       by inferred cache hit. First-visit calls get an outline so you
       can see "first-visit but already a hit" (= prefetch win).
    2. <plot>.bars.png     -- per-agent median TTFT, cold vs warm side
       by side.

    Matplotlib is imported lazily so the script still runs on hosts
    where it isn't installed.
    """
    import matplotlib  # noqa: F401  -- ImportError surfaces to caller

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = args.plot.rstrip(".png")
    timeline_path = f"{base}.timeline.png"
    bars_path = f"{base}.bars.png"

    # ---- timeline plot ----
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(results)), 5))
    xs = [r.call_idx for r in results]
    ys = [r.ttft_ms if r.ttft_ms == r.ttft_ms else 0.0 for r in results]
    colors = ["#2ca02c" if r.inferred_hit else "#d62728" for r in results]
    edges = ["black" if r.first_visit else "none" for r in results]
    ax.bar(xs, ys, color=colors, edgecolor=edges, linewidth=1.2)
    ax.axhline(
        HIT_TTFT_THRESHOLD_MS,
        color="gray", linestyle="--", linewidth=1,
        label=f"hit threshold ({HIT_TTFT_THRESHOLD_MS:.0f} ms)",
    )

    # Round boundaries.
    last_round = -1
    for r in results:
        if r.round_idx != last_round:
            ax.axvline(r.call_idx - 0.5, color="lightgray",
                       linestyle=":", linewidth=0.8)
            ax.text(
                r.call_idx, ax.get_ylim()[1] * 0.95 if ys else 1.0,
                f" R{r.round_idx + 1}", color="gray", fontsize=8,
            )
            last_round = r.round_idx

    ax.set_xlabel("Call index (predetermined order)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(
        f"Agent workflow TTFT per call -- mode={args.mode}, "
        f"variants/agent={args.variants_per_agent}, "
        f"~{estimate_tokens(build_preamble('x', 0, args.preamble_lines))} "
        f"tokens/preamble"
    )
    # Manual legend so first-visit outline + green/red both appear.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2ca02c", label="cache hit (inferred)"),
        Patch(facecolor="#d62728", label="cache miss (inferred)"),
        Patch(facecolor="white", edgecolor="black",
              label="first visit (black outline)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(timeline_path, dpi=120)
    plt.close(fig)
    print(f"\nwrote timeline plot: {timeline_path}")

    # ---- per-agent bars ----
    fig, ax = plt.subplots(figsize=(8, 5))
    agents = WORKFLOW_ORDER
    cold_med = []
    warm_med = []
    for agent_id in agents:
        cold = [r.ttft_ms for r in results
                if r.agent_id == agent_id and r.first_visit
                and r.ttft_ms == r.ttft_ms]
        warm = [r.ttft_ms for r in results
                if r.agent_id == agent_id and not r.first_visit
                and r.ttft_ms == r.ttft_ms]
        cold_med.append(statistics.median(cold) if cold else 0.0)
        warm_med.append(statistics.median(warm) if warm else 0.0)

    x = list(range(len(agents)))
    width = 0.4
    ax.bar([xi - width / 2 for xi in x], cold_med, width,
           color="#d62728", label="first-visit (cold) median")
    ax.bar([xi + width / 2 for xi in x], warm_med, width,
           color="#2ca02c", label="revisit (warm) median")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Median TTFT (ms)")
    ax.set_title(
        f"Per-agent TTFT: cold vs warm -- mode={args.mode}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(bars_path, dpi=120)
    plt.close(fig)
    print(f"wrote per-agent bar plot: {bars_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an agent1 -> agent2 -> agent3 workflow against "
        "vLLM, optionally exercising the agent-prefetch endpoint.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--mode", choices=["baseline", "warmup", "prefetch"],
                   default="baseline")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument(
        "--prefetch-top-k", type=int, default=None,
        help="Optional cap on how many of the agent's registered "
        "prefixes to warm per prefetch call. Default: omit -- the "
        "server warms every prefix it has stored for the agent.",
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--preamble-lines", type=int, default=80,
        help="Per-variant system-preamble length, in template lines "
        "(~20 tokens each). 80 is short; 1000+ is what you want to "
        "make cold-prefill cost obvious.",
    )
    p.add_argument(
        "--variants-per-agent", type=int, default=1,
        help="Distinct system-preamble variants per agent. The workflow "
        "cycles through variants based on query index, so each agent "
        "exercises this many distinct prefixes per round.",
    )
    p.add_argument(
        "--sleep-between-rounds", type=float, default=0.0,
        help="Seconds to sleep between rounds. Useful with concurrent "
        "load from other clients -- lets APC age out so revisits "
        "fall back to LMCache.",
    )
    p.add_argument(
        "--plot", default=None,
        help="If set, write timeline + per-agent PNGs using this as "
        "the filename base. Requires matplotlib. Example: --plot "
        "/tmp/agent_workflow",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_workflow(parse_args()))
