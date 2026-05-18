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

  --mode prefetch   POST to /v1/agents/chat/completions with
                    {agent_id, prefetch_top_k}. Requires the Commit-2/3
                    engine + endpoint changes.

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


def call_prefetch(base_url: str, model: str, agent_id: str, system: str,
                  query: str, max_tokens: int, prefetch_top_k: int
                  ) -> tuple[float, float, str, dict | None]:
    url = f"{base_url.rstrip('/')}/v1/agents/chat/completions"
    payload = _build_chat_payload(model, system, query, max_tokens=max_tokens)
    payload["agent_id"] = agent_id
    payload["prefetch_top_k"] = prefetch_top_k
    return _run_streaming(url, payload)


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
        f"top_k: {args.prefetch_top_k}"
    )
    print(f"Target: {args.base_url} | model: {args.model}")
    if args.sleep_between_rounds > 0:
        print(f"Sleep between rounds: {args.sleep_between_rounds}s")
    print()

    results: list[CallResult] = []
    seen_pairs: set[tuple[str, int]] = set()

    for round_idx in range(args.rounds):
        if round_idx > 0 and args.sleep_between_rounds > 0:
            print(f"  ...sleeping {args.sleep_between_rounds}s before round "
                  f"{round_idx + 1}...")
            time.sleep(args.sleep_between_rounds)

        print(f"=== Round {round_idx + 1} ===")
        for agent_id in WORKFLOW_ORDER:
            queries = AGENT_QUERIES[agent_id]
            for q_idx, query in enumerate(queries):
                variant = q_idx % args.variants_per_agent
                pair = (agent_id, variant)
                first_visit = pair not in seen_pairs
                seen_pairs.add(pair)

                system = preambles[pair]

                if args.mode == "baseline":
                    ttft, total, preview, meta = call_baseline(
                        args.base_url, args.model, system, query,
                        args.max_tokens)
                elif args.mode == "warmup":
                    ttft, total, preview, meta = call_warmup_then_real(
                        args.base_url, args.model, system, query,
                        args.max_tokens)
                elif args.mode == "prefetch":
                    ttft, total, preview, meta = call_prefetch(
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
                )
                results.append(r)

                tag = "COLD" if first_visit else "warm"
                extra = ""
                if r.apc_hit_tokens is not None:
                    extra = (
                        f" | apc={r.apc_hit_tokens} "
                        f"lmc_extra={r.lmcache_extra_tokens}"
                    )
                print(
                    f"  {agent_id} v{variant} q{q_idx} [{tag:4s}]: "
                    f"ttft={ttft:8.1f}ms total={total:8.1f}ms"
                    f"{extra}  | {preview}"
                )
        print()

    _print_summary(results)

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an agent1 -> agent2 -> agent3 workflow against "
        "vLLM, optionally exercising the agent-prefetch endpoint.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--mode", choices=["baseline", "warmup", "prefetch"],
                   default="baseline")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--prefetch-top-k", type=int, default=20)
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
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_workflow(parse_args()))
