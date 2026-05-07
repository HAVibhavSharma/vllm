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
    manifest_path = templates_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.json not found in {templates_dir}. Expected keys: "
            "chunk_template (list[str]), anchor_indices (list[int]), "
            "stages (list[int])."
        )
    manifest = json.loads(manifest_path.read_text())
    for key in ("chunk_template", "anchor_indices", "stages"):
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


def post_chunked_chat(
    base_url: str,
    model: str,
    chunks: list[str],
    anchor_indices: list[int],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    payload = {
        "model": model,
        "chunks": chunks,
        "anchor_indices": anchor_indices,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{base_url.rstrip('/')}/v1/chunked_chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            response_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status = e.code
        response_body = e.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        parsed = {"_raw": response_body}
    return status, parsed


def run_stage(
    args: argparse.Namespace,
    templates_dir: Path,
    chunk_template: list[str],
    anchor_indices: list[int],
    stage: int,
) -> StageResult:
    chunks = build_chunks(templates_dir, chunk_template, stage)
    chunk_chars = [len(c) for c in chunks]

    t0 = time.perf_counter()
    status, body = post_chunked_chat(
        base_url=args.base_url,
        model=args.model,
        chunks=chunks,
        anchor_indices=anchor_indices,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout,
    )
    elapsed = time.perf_counter() - t0

    output = ""
    prompt_tokens = None
    completion_tokens = None
    if status == 200:
        try:
            output = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            output = ""
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    else:
        output = json.dumps(body)[:500]

    return StageResult(
        stage=stage,
        latency_s=elapsed,
        chunk_chars=chunk_chars,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates_dir: Path = args.templates_dir
    manifest = load_manifest(templates_dir)
    chunk_template: list[str] = manifest["chunk_template"]
    anchor_indices: list[int] = manifest["anchor_indices"]
    stages: list[int] = manifest["stages"]
    topic = manifest.get("topic", templates_dir.name)

    print(f"=== chunked_chat benchmark: {topic} ===")
    print(f"  templates dir : {templates_dir}")
    print(f"  base url      : {args.base_url}")
    print(f"  model         : {args.model}")
    print(f"  chunk layout  : {chunk_template}")
    print(f"  anchor indices: {anchor_indices}")
    print(f"  stages        : {stages}")
    print()

    if args.warmup and stages:
        print("[warmup] running stage", stages[0], "once and discarding result")
        run_stage(args, templates_dir, chunk_template, anchor_indices, stages[0])
        print()

    results: list[StageResult] = []
    for stage in stages:
        r = run_stage(args, templates_dir, chunk_template, anchor_indices, stage)
        results.append(r)
        chars_str = "+".join(str(c) for c in r.chunk_chars)
        print(
            f"[stage {stage}] status={r.status} "
            f"latency={r.latency_s:.3f}s "
            f"chunk_chars={chars_str} (total={sum(r.chunk_chars)}) "
            f"prompt_tokens={r.prompt_tokens} "
            f"completion_tokens={r.completion_tokens}"
        )
        if args.print_output:
            print(f"  output: {r.output!r}")

    print()
    print("=== summary ===")
    for r in results:
        print(
            f"  stage {r.stage:>3}: {r.latency_s:7.3f}s "
            f"prompt_tokens={r.prompt_tokens}"
        )


if __name__ == "__main__":
    main()
