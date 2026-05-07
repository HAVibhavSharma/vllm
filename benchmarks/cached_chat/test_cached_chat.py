# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for the manual /v1/cached_chat/completions endpoint.

Sends the same prompt N times and reports per-call latency. With the
manual KV pipeline wired up, call 1 pays full prefill (and triggers a
capture); calls 2..N skip prefill (the worker injects the saved KV) and
should be visibly faster.

Required server flags for a meaningful run:

    vllm serve <model> \\
        --no-enable-prefix-caching \\
        --worker-extension-cls \\
            vllm.entrypoints.openai.cached_chat.worker_ext.ManualKVWorkerExtension

Usage:

    python benchmarks/cached_chat/test_cached_chat.py \\
        --base-url http://localhost:8000 \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --repeats 3 --strict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PROMPT = (
    "Explain photosynthesis to a curious 12-year-old in roughly five "
    "sentences. Mention chlorophyll, sunlight, water, carbon dioxide, "
    "and the role of the chloroplast. Avoid analogies that involve cooking. "
    "After your explanation, list three follow-up questions a curious "
    "12-year-old might ask, and answer each one in a single sentence."
)


def post_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[int, dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{base_url.rstrip('/')}/v1/cached_chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            response_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status = e.code
        response_body = e.read().decode("utf-8", errors="replace")
    elapsed = time.perf_counter() - t0
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        parsed = {"_raw": response_body}
    return status, parsed, elapsed


def usage_summary(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send one discarded request first.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero unless call 2's latency is at least "
            "--strict-speedup x faster than call 1."
        ),
    )
    parser.add_argument("--strict-speedup", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=== cached_chat (manual KV) end-to-end test ===")
    print(f"  base url: {args.base_url}")
    print(f"  model   : {args.model}")
    print(f"  prompt  : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print()

    if args.warmup:
        print("[warmup] sending one discarded request")
        post_chat(
            args.base_url,
            args.model,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.timeout,
        )

    latencies: list[float] = []
    failures = 0
    for i in range(1, args.repeats + 1):
        status, body, elapsed = post_chat(
            args.base_url,
            args.model,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.timeout,
        )
        if status != 200:
            print(f"[call {i}] FAILED status={status} body={body}")
            failures += 1
            continue
        u = usage_summary(body)
        latencies.append(elapsed)
        expectation = "MISS (capture)" if i == 1 else f"HIT  (inject, #{i - 1})"
        print(
            f"[call {i}] expect={expectation:18s} latency={elapsed:.3f}s "
            f"prompt_tokens={u['prompt_tokens']} "
            f"completion_tokens={u['completion_tokens']}"
        )

    if failures:
        print(f"\n{failures} call(s) failed", file=sys.stderr)
        return 1

    if args.strict and len(latencies) >= 2:
        speedup = latencies[0] / max(latencies[1], 1e-9)
        if speedup < args.strict_speedup:
            print(
                f"\nSTRICT FAIL: call 1 / call 2 = {speedup:.2f}x, "
                f"need >= {args.strict_speedup:.2f}x. "
                "Is --no-enable-prefix-caching set, and is the worker "
                "extension installed?",
                file=sys.stderr,
            )
            return 2
        print(f"\nSTRICT OK: call 1 / call 2 = {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
