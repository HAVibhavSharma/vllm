# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-wise benchmark for the /v1/chunked_chat/completions endpoint.

Mimics an agentic loop where the prompt is a sandwich:

    [ static prefix (anchor) | dynamic body (grows per stage) | static suffix (anchor) ]

The static chunks stay byte-identical across stages so the server can treat
them as anchors; only the dynamic chunk grows. One HTTP request is sent per
stage and per-stage latency / token usage is reported.

Templates live in a folder described by `manifest.json`:

    templates/<topic>/
        manifest.json
        prefix.txt          # static anchor
        suffix.txt          # static anchor
        stage_1.txt ...     # dynamic, one per stage

Example:

    python benchmarks/chunked_chat/benchmark_chunked_chat.py \\
        --templates-dir benchmarks/chunked_chat/templates/photosynthesis \\
        --base-url http://localhost:8000 \\
        --model meta-llama/Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    stage: int
    latency_s: float
    ttft_s: float | None
    chunk_chars: list[int]
    prompt_tokens: int | None
    completion_tokens: int | None
    output: str
    status: int


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Template file missing: {path}")
    return path.read_text()


def load_manifest(templates_dir: Path) -> dict[str, Any]:
    """Load + validate the template manifest.

    Two manifest schemas are supported:

    1. Legacy "fixed-layout" mode (default):
           {
             "chunk_template":   ["prefix.txt", "stage_{n}.txt", "suffix.txt"],
             "anchor_indices":   [0, 2],
             "stages":           [1, 2, 3, 4]
           }
       Same set of files per stage; only `{n}` substitution differs.

    2. "growing_history" mode (mimics an agentic loop):
           {
             "mode":                   "growing_history",
             "static_prefix_chunks":   ["prefix.txt"],
             "history_turn_template":  "past_turn_{n}.txt",
             "current_turn_template":  "current_turn_{n}.txt",
             "static_suffix_chunks":   ["suffix.txt"],
             "stages":                 [1, 2, 3, 4]
           }
       Stage N composes the prompt as:
           [static_prefix_chunks]
           [past_turn_1, ..., past_turn_{N-1}]    ← grows by 1 per stage
           [current_turn_N]                       ← the only dynamic chunk
           [static_suffix_chunks]
       All chunks except `current_turn_N` are marked as anchors.
    """
    manifest_path = templates_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.json not found in {templates_dir}. Expected keys: "
            "chunk_template (list[str]), anchor_indices (list[int]), "
            "stages (list[int])."
        )
    manifest = json.loads(manifest_path.read_text())
    if "stages" not in manifest:
        raise ValueError("manifest.json missing required key: 'stages'")
    mode = manifest.get("mode", "fixed")
    if mode == "growing_history":
        required = (
            "static_prefix_chunks",
            "history_turn_template",
            "current_turn_template",
            "static_suffix_chunks",
        )
        for key in required:
            if key not in manifest:
                raise ValueError(
                    f"manifest.json (growing_history) missing required key: {key!r}"
                )
    else:
        for key in ("chunk_template", "anchor_indices"):
            if key not in manifest:
                raise ValueError(f"manifest.json missing required key: {key!r}")
    return manifest


def build_chunks(
    templates_dir: Path,
    chunk_template: list[str],
    stage: int,
) -> list[str]:
    """Resolve `chunk_template` filenames for a given stage.

    Filenames may contain the placeholder `{n}`, replaced with the stage
    number. Files without `{n}` are treated as static anchors (loaded once
    per stage but their bytes don't change).
    """
    return [_read_text(templates_dir / name.format(n=stage)) for name in chunk_template]


def build_growing_chunks(
    templates_dir: Path,
    manifest: dict[str, Any],
    stage: int,
) -> tuple[list[str], list[int]]:
    """Compose the prompt for `growing_history` mode at this stage.

    Returns `(chunks, anchor_indices)`.
    """
    prefix_files = list(manifest["static_prefix_chunks"])
    suffix_files = list(manifest["static_suffix_chunks"])
    history_template: str = manifest["history_turn_template"]
    current_template: str = manifest["current_turn_template"]

    prefix_chunks = [_read_text(templates_dir / f) for f in prefix_files]
    history_chunks = [
        _read_text(templates_dir / history_template.format(n=i))
        for i in range(1, stage)
    ]
    current_chunks = [_read_text(templates_dir / current_template.format(n=stage))]
    suffix_chunks = [_read_text(templates_dir / f) for f in suffix_files]

    chunks = prefix_chunks + history_chunks + current_chunks + suffix_chunks

    # Index layout:
    #   [0 .. P)               = static prefix chunks
    #   [P .. P+H)             = past history turns (static after stage 1)
    #   [P+H .. P+H+1)         = current dynamic turn  ← NOT an anchor
    #   [P+H+1 .. end)         = static suffix chunks
    P = len(prefix_chunks)
    H = len(history_chunks)
    current_idx = P + H
    suffix_start = current_idx + 1

    anchor_indices = (
        list(range(0, P))                   # prefixes
        + list(range(P, P + H))             # frozen-history turns
        + list(range(suffix_start, len(chunks)))  # suffixes
    )
    return chunks, anchor_indices


def post_chunked_chat_streaming(
    base_url: str,
    model: str,
    chunks: list[str],
    anchor_indices: list[int],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    agent_id: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Send a streaming chunked_chat request and measure TTFT.

    Returns (status, body) where body has the following extra keys:
        - ttft_s: time-to-first-content-token, in seconds
        - latency_s: total elapsed time from request start to stream end
        - output: assembled assistant text
        - prompt_tokens / completion_tokens: from the final usage frame
    """
    payload: dict[str, Any] = {
        "model": model,
        "chunks": chunks,
        "anchor_indices": anchor_indices,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Ask vLLM to include usage in the final stream frame; without
        # this many servers omit usage in streaming mode.
        "stream_options": {"include_usage": True},
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    body = json.dumps(payload).encode("utf-8")
    url = f"{base_url.rstrip('/')}/v1/chunked_chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    t0 = time.perf_counter()
    ttft: float | None = None
    output_parts: list[str] = []
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    status = 0
    error: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        output_parts.append(content)

                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            error = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            error = str(e)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - t0
    return status, {
        "ttft_s": ttft,
        "latency_s": elapsed,
        "output": "".join(output_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "_error": error,
    }


def run_stage(
    args: argparse.Namespace,
    templates_dir: Path,
    manifest: dict[str, Any],
    stage: int,
) -> StageResult:
    mode = manifest.get("mode", "fixed")
    if mode == "growing_history":
        chunks, anchor_indices = build_growing_chunks(
            templates_dir, manifest, stage
        )
    else:
        chunks = build_chunks(templates_dir, manifest["chunk_template"], stage)
        anchor_indices = list(manifest["anchor_indices"])
    chunk_chars = [len(c) for c in chunks]

    status, body = post_chunked_chat_streaming(
        base_url=args.base_url,
        model=args.model,
        chunks=chunks,
        anchor_indices=anchor_indices,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout,
        agent_id=args.agent_id,
    )

    output = body.get("output") or ""
    if status != 200 and not output:
        output = (body.get("_error") or json.dumps(body))[:500]

    return StageResult(
        stage=stage,
        latency_s=float(body.get("latency_s") or 0.0),
        ttft_s=body.get("ttft_s"),
        chunk_chars=chunk_chars,
        prompt_tokens=body.get("prompt_tokens"),
        completion_tokens=body.get("completion_tokens"),
        output=output,
        status=status,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--templates-dir",
        type=Path,
        required=True,
        help="Folder containing manifest.json and stage template files.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="vLLM server base URL.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (must match what the server is serving).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print the full assistant output for each stage.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send the first stage twice so caches are warm before timing.",
    )
    parser.add_argument(
        "--agent-id",
        default="benchmark-agent-1",
        help=(
            "Sent as the request's `agent_id` so the server namespaces "
            "anchor pools per agent. Defaults to a single agent for the "
            "whole run; pass an empty string to disable."
        ),
    )
    ns = parser.parse_args()
    if ns.agent_id == "":
        ns.agent_id = None
    return ns


def main() -> None:
    args = parse_args()
    templates_dir: Path = args.templates_dir
    manifest = load_manifest(templates_dir)
    stages: list[int] = manifest["stages"]
    topic = manifest.get("topic", templates_dir.name)
    mode = manifest.get("mode", "fixed")

    print(f"=== chunked_chat benchmark: {topic} ===")
    print(f"  templates dir : {templates_dir}")
    print(f"  base url      : {args.base_url}")
    print(f"  model         : {args.model}")
    print(f"  agent_id      : {args.agent_id if args.agent_id else '-'}")
    print(f"  mode          : {mode}")
    if mode == "growing_history":
        print(f"  prefix chunks : {manifest['static_prefix_chunks']}")
        print(f"  history tpl   : {manifest['history_turn_template']}")
        print(f"  current tpl   : {manifest['current_turn_template']}")
        print(f"  suffix chunks : {manifest['static_suffix_chunks']}")
    else:
        print(f"  chunk layout  : {manifest['chunk_template']}")
        print(f"  anchor indices: {manifest['anchor_indices']}")
    print(f"  stages        : {stages}")
    print()

    if args.warmup and stages:
        print("[warmup] running stage", stages[0], "once and discarding result")
        run_stage(args, templates_dir, manifest, stages[0])
        print()

    results: list[StageResult] = []
    for stage in stages:
        r = run_stage(args, templates_dir, manifest, stage)
        results.append(r)
        chars_str = "+".join(str(c) for c in r.chunk_chars)
        ttft_str = f"{r.ttft_s:.3f}s" if r.ttft_s is not None else "n/a"
        print(
            f"[stage {stage}] status={r.status} "
            f"ttft={ttft_str} "
            f"latency={r.latency_s:.3f}s "
            f"chunk_chars={chars_str} (total={sum(r.chunk_chars)}) "
            f"prompt_tokens={r.prompt_tokens} "
            f"completion_tokens={r.completion_tokens}"
        )
        if args.print_output:
            print(f"  output: {r.output!r}")

    print()
    print("=== summary ===")
    print(f"  {'stage':>5}  {'ttft':>8}  {'latency':>8}  prompt_tokens")
    baseline_ttft = results[0].ttft_s if results else None
    for r in results:
        ttft_str = f"{r.ttft_s:.3f}s" if r.ttft_s is not None else "n/a"
        delta = ""
        if baseline_ttft and r.ttft_s and r.stage != results[0].stage:
            speedup = baseline_ttft / r.ttft_s
            delta = f"  ({speedup:.2f}× vs stage {results[0].stage})"
        print(
            f"  {r.stage:>5}  {ttft_str:>8}  "
            f"{r.latency_s:7.3f}s  {r.prompt_tokens}{delta}"
        )


if __name__ == "__main__":
    main()
