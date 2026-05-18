# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demo workflow exercising the agent-scoped prefix prefetch endpoint.

The workflow simulates a multi-agent pipeline that runs in a fixed,
deterministic order:

    agent1 -> agent2 -> agent3 -> agent1 -> agent2 -> agent3 -> ...

Each agent has a long, stable "system" preamble (the prefix we want
cached) plus a short per-call query. On the second pass through the
agents the preambles should hit either:

  * vLLM APC (best -- if the prefetch landed in time), or
  * LMCache CPU L1 (good -- async CPU->GPU retrieve overlapped with
    model exec),

and you should observe a noticeable TTFT improvement compared to a
cold first pass.

The script supports three modes:

  --mode baseline   POST to /v1/chat/completions only. No prefetch.
                    Establishes the cost without LMCache-driven warmup.

  --mode warmup     Issue a dummy max_tokens=1 warmup before each call.
                    The "old way" -- effective but pays one full prefill
                    per warm-up.

  --mode prefetch   POST to /v1/agents/chat/completions with
                    {agent_id, prefetch_top_k}. Requires the
                    Commit-2/3 engine + endpoint changes to be live.

Run against the deployment described in the project notes:

  lmcache server --l1-size-gb 20 --eviction-policy LRU --chunk-size 16 \\
      --host 0.0.0.0 --port 10903

  vllm serve Qwen/Qwen3-8B --port 8000 --kv-transfer-config \\
      '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

Usage:

  python examples/online_serving/agent_prefetch_workflow.py \\
      --base-url http://<vllm-host>:8000 \\
      --model Qwen/Qwen3-8B \\
      --mode prefetch \\
      --rounds 2 \\
      --prefetch-top-k 20
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
# Workflow definition.
#
# Each agent has:
#   - A stable "system" preamble (long, cacheable as a prefix).
#   - A list of short user queries that vary per call.
#
# The preambles are deliberately verbose so that, at the default vLLM
# block_size=16 and LMCache chunk_size=16, each one spans many chunks
# and APC/LMCache hits are clearly visible in metrics.
# ---------------------------------------------------------------------------


def _long_preamble(role: str, body_lines: int = 80) -> str:
    """Build a stable, long system prompt. The text content does not
    matter for caching purposes -- only stability across calls does."""
    base = (
        f"You are {role}, a careful and verbose assistant. Always think "
        "step by step before answering. Cite sources only when relevant. "
    )
    filler = " ".join(
        f"({role}-context line {i}: maintain prior conventions, observe "
        "the established style, and avoid speculation.)"
        for i in range(body_lines)
    )
    return base + filler


AGENTS: dict[str, dict] = {
    "agent1": {
        "system": _long_preamble("Agent1-Research"),
        "queries": [
            "Summarize the goal of LMCache in one sentence.",
            "List two trade-offs of CPU-resident KV caches.",
            "Why is chunked prefill relevant here?",
        ],
    },
    "agent2": {
        "system": _long_preamble("Agent2-Critic"),
        "queries": [
            "Critique the previous summary for vagueness.",
            "Where would prefix mismatch most likely happen?",
        ],
    },
    "agent3": {
        "system": _long_preamble("Agent3-Synthesizer"),
        "queries": [
            "Combine the prior outputs into a 3-bullet plan.",
            "Estimate the latency saving from APC hits.",
            "Propose one observability metric to track this.",
            "Suggest a stress test design.",
        ],
    },
}

# Deterministic call order. The first pass is cold (populates LMCache
# during STORE); the second and later passes should benefit from the
# prefetch when --mode=prefetch.
WORKFLOW_ORDER: list[str] = ["agent1", "agent2", "agent3"]


# ---------------------------------------------------------------------------
# Minimal HTTP client (no openai / httpx dependency required).
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    agent_id: str
    query_idx: int
    ttft_ms: float
    total_ms: float
    apc_hit_tokens: int | None
    lmcache_extra_tokens: int | None
    response_preview: str


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_stream(url: str, payload: dict, timeout: float = 120.0):
    """Yield (event_time_ns, parsed_chunk_dict | None) tuples.

    Used for SSE streaming responses. ``None`` chunk_dict signals the
    terminal ``[DONE]`` message.
    """
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
                yield (time.perf_counter_ns(), None)
                return
            try:
                yield (time.perf_counter_ns(), json.loads(data))
            except json.JSONDecodeError:
                continue


def _build_chat_payload(model: str, system: str, query: str, *,
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
# Modes.
# ---------------------------------------------------------------------------


def call_baseline(base_url: str, model: str, agent_id: str,
                  system: str, query: str, max_tokens: int) -> CallResult:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = _build_chat_payload(model, system, query, max_tokens=max_tokens)
    return _run_streaming_call(url, payload, agent_id)


def call_warmup_then_real(base_url: str, model: str, agent_id: str,
                          system: str, query: str,
                          max_tokens: int) -> CallResult:
    """Old approach: send a dummy max_tokens=1 warmup before the real call.

    The warmup populates APC + LMCache, then the real call hits APC.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    warmup_payload = _build_chat_payload(
        model, system, query="warmup", max_tokens=1)
    warmup_payload["stream"] = False
    try:
        _post_json(url, warmup_payload)
    except urllib.error.HTTPError as e:
        print(f"  [warn] warmup failed for {agent_id}: {e}", file=sys.stderr)
    payload = _build_chat_payload(model, system, query, max_tokens=max_tokens)
    return _run_streaming_call(url, payload, agent_id)


def call_prefetch(base_url: str, model: str, agent_id: str,
                  system: str, query: str,
                  max_tokens: int, prefetch_top_k: int) -> CallResult:
    """New approach: hit /v1/agents/chat/completions with agent_id.

    Requires the Commit-2 + Commit-3 changes (engine plumbing + new
    HTTP route) to be deployed. Until then this will 404 and the script
    will report it.
    """
    url = f"{base_url.rstrip('/')}/v1/agents/chat/completions"
    payload = _build_chat_payload(model, system, query, max_tokens=max_tokens)
    payload["agent_id"] = agent_id
    payload["prefetch_top_k"] = prefetch_top_k
    return _run_streaming_call(url, payload, agent_id)


def _run_streaming_call(url: str, payload: dict,
                        agent_id: str) -> CallResult:
    start_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    text_chunks: list[str] = []
    prefetch_meta: dict | None = None

    try:
        for evt_ns, chunk in _post_stream(url, payload):
            if chunk is None:
                break
            # Extract prefetch_meta from the final chunk when present.
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
        return CallResult(
            agent_id=agent_id,
            query_idx=-1,
            ttft_ms=float("nan"),
            total_ms=float("nan"),
            apc_hit_tokens=None,
            lmcache_extra_tokens=None,
            response_preview=f"<HTTP {e.code}: {e.reason}>",
        )

    end_ns = time.perf_counter_ns()
    ttft_ms = (
        (first_token_ns - start_ns) / 1e6
        if first_token_ns is not None
        else float("nan")
    )
    total_ms = (end_ns - start_ns) / 1e6
    response_preview = "".join(text_chunks)[:80].replace("\n", " ")

    return CallResult(
        agent_id=agent_id,
        query_idx=-1,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        apc_hit_tokens=(prefetch_meta or {}).get("apc_hit_tokens"),
        lmcache_extra_tokens=(prefetch_meta or {}).get("lmcache_extra_tokens"),
        response_preview=response_preview,
    )


# ---------------------------------------------------------------------------
# Workflow driver.
# ---------------------------------------------------------------------------


def run_workflow(args: argparse.Namespace) -> int:
    print(
        f"Mode: {args.mode} | rounds: {args.rounds} | "
        f"top_k: {args.prefetch_top_k} | model: {args.model}"
    )
    print(f"Target: {args.base_url}")
    print(
        "Order: "
        + " -> ".join(WORKFLOW_ORDER * args.rounds)
        + "\n"
    )

    results: list[CallResult] = []
    per_agent_ttfts: dict[str, list[float]] = {a: [] for a in AGENTS}

    for round_idx in range(args.rounds):
        cold = round_idx == 0
        label = "COLD" if cold else f"WARM r{round_idx}"
        print(f"=== Round {round_idx + 1} ({label}) ===")
        for agent_id in WORKFLOW_ORDER:
            spec = AGENTS[agent_id]
            for q_idx, query in enumerate(spec["queries"]):
                if args.mode == "baseline":
                    r = call_baseline(
                        args.base_url, args.model, agent_id,
                        spec["system"], query, args.max_tokens)
                elif args.mode == "warmup":
                    r = call_warmup_then_real(
                        args.base_url, args.model, agent_id,
                        spec["system"], query, args.max_tokens)
                elif args.mode == "prefetch":
                    r = call_prefetch(
                        args.base_url, args.model, agent_id,
                        spec["system"], query, args.max_tokens,
                        prefetch_top_k=args.prefetch_top_k)
                else:
                    raise ValueError(f"Unknown mode: {args.mode}")

                r.query_idx = q_idx
                results.append(r)
                per_agent_ttfts[agent_id].append(r.ttft_ms)

                extra = ""
                if r.apc_hit_tokens is not None:
                    extra = (
                        f" | apc={r.apc_hit_tokens} "
                        f"lmc_extra={r.lmcache_extra_tokens}"
                    )
                print(
                    f"  {agent_id} q{q_idx}: "
                    f"ttft={r.ttft_ms:7.1f}ms total={r.total_ms:7.1f}ms"
                    f"{extra}  | {r.response_preview}"
                )
        print()

    # ----- summary -----
    print("=== Summary (median TTFT in ms) ===")
    for agent_id in WORKFLOW_ORDER:
        ttfts = [t for t in per_agent_ttfts[agent_id] if t == t]  # drop NaN
        if not ttfts:
            print(f"  {agent_id}: no successful calls")
            continue
        print(
            f"  {agent_id}: median={statistics.median(ttfts):7.1f}  "
            f"min={min(ttfts):7.1f}  max={max(ttfts):7.1f}  "
            f"n={len(ttfts)}"
        )

    nan_count = sum(1 for r in results if r.ttft_ms != r.ttft_ms)
    if nan_count:
        print(f"\n[warn] {nan_count} call(s) returned no first token "
              "(check server logs).")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an agent1 -> agent2 -> agent3 workflow against "
        "vLLM, optionally exercising the agent-prefetch endpoint.",
    )
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--mode", choices=["baseline", "warmup", "prefetch"],
                   default="baseline")
    p.add_argument("--rounds", type=int, default=2,
                   help="How many times to walk WORKFLOW_ORDER. "
                   "Round 1 is cold; round 2+ should benefit from caching.")
    p.add_argument("--prefetch-top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_workflow(parse_args()))
