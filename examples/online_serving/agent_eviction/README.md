# Agent-Aware Eviction Demo — Scenario A (Round-Robin Under Pressure)

This demo shows when and why the **agent-aware early-eviction** policy
(`vllm/v1/agent_prefetch/eviction.py`) beats plain LRU on a prefix-cached
agent workload.

Plan doc: [`plan/lru_vs_agent_eviction.md`](../../../plan/lru_vs_agent_eviction.md)
Policy doc: [`plan/agent_eviction_policy.md`](../../../plan/agent_eviction_policy.md)

## The setup in one paragraph

A pool of `N` agents (configurable via `--num-agents`) each own a long,
stable system prompt. Requests are issued in strict round-robin order
(`agent_a → agent_b → ... → agent_N → agent_a → ...`) so the cache is
constantly being asked to hold something it just dropped. The GPU prefix
cache must be **sized so it cannot hold every agent's prompt at once**,
which forces an eviction on every cycle.

- **Baseline run:** requests carry only `agent_id` (no probabilities). The
  block pool falls back to plain LRU and evicts whichever block was used
  longest ago — which is always the agent about to fire next.
- **Treatment run:** each request also carries an `agent_probabilities`
  forecast: the next agent in the rotation at probability `1.0`, the one
  that just fired at probability `0.0`. The policy keeps the next agent's
  prefix pinned and evicts the genuinely-stale agent first.

The expected outcome is a large prefix-cache hit-rate gap, visible as a
TTFT gap on every request after the first full rotation.

## Files

- `run_round_robin.py` — the demo driver (HTTP client, two modes, metrics).
- `prompts.py` — agent prompt generator. The first four agents have
  hand-written personas; agents 5+ are synthesized from a role pool so
  the demo scales to arbitrary `N`.
- `README.md` — this file.

## Hardware target: H100 80 GB + Qwen2.5-72B-AWQ

This demo is sized for an **H100 (80 GB) running Qwen2.5-72B-Instruct
quantized to AWQ**. The 72B AWQ model is the sweet spot for this demo
because:

1. **Per-token KV cost is ~6× higher** than a 7B model (more layers,
   wider KV) so it takes far fewer agents and shorter prompts to fill
   the cache and force evictions.
2. **Cold prefill is ~8× slower**, so the TTFT gap between a hit and a
   miss is unmistakable — easy to see on a slide, easy to graph.
3. **AWQ weights fit comfortably** in ~38 GB, leaving ~38 GB for the KV
   pool. That's just enough room to hold ~12 of the 16 agent prompts at
   once — i.e. the cache must evict every cycle.

Rough KV math at the recommended settings:

| Item | Value |
| --- | --- |
| H100 VRAM | 80 GB |
| Qwen2.5-72B-AWQ weights | ~38 GB |
| KV pool at `--gpu-memory-utilization 0.95` | ~38 GB |
| KV bytes/token (80 layers, 8 KV heads, dim 128, fp16 KV) | ~327 KB |
| ⇒ cache capacity | ~115K tokens |
| Working set (16 agents × ~16K-token prompts at `--filler-lines 500`) | ~263K tokens |
| ⇒ cache holds ~7 of 16 prompts → 9 forced evictions per cycle | ✔ |

> AWQ quantizes weights only; the KV cache is still fp16, which is why
> per-token KV cost stays high.

## Prerequisites

Set up the env once per `AGENTS.md`:

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

## Start the server

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq_marlin \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768
```

First load takes 1–3 minutes — that's the model, not the demo.

> The 32K `--max-model-len` is intentional. Each filler line in
> `prompts.py` is ~33 tokens, so 500 lines lands at ~16.4K tokens — a
> 16K context window would reject the request. 32K gives headroom for
> the prompt + the 16-token output and lets you push `--filler-lines`
> higher if you want even more cache pressure.

## Run the demo

Run both modes back-to-back with **the same flags** so the only
difference is whether probabilities are sent:

```bash
# Baseline (pure LRU)
.venv/bin/python examples/online_serving/agent_eviction/run_round_robin.py \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --mode baseline \
    --num-agents 16 \
    --filler-lines 500 \
    --hit-threshold-ms 500 \
    --rounds 4

# Treatment (agent-aware) — same flags, just swap --mode
.venv/bin/python examples/online_serving/agent_eviction/run_round_robin.py \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --mode treatment \
    --num-agents 16 \
    --filler-lines 500 \
    --hit-threshold-ms 500 \
    --rounds 4
```

> Between runs, call
> `curl -X POST http://localhost:8000/v1/agents/reset_prefix_cache`
> (or restart the server) to clear cache state so baseline doesn't
> benefit from the treatment run's tags and vice versa.

The script prints per-call TTFT and a final summary table with:

- Cache hit rate (TTFT below `--hit-threshold-ms` ⇒ counted as a hit)
- Mean / p95 TTFT
- Per-agent breakdown
- A final pull from `GET /v1/agents/eviction_stats` so you can see the
  policy's view of the world (tagged blocks, low-probability agents,
  effective threshold).

## Expected outcome

| Run       | Hit rate | Mean TTFT | What's happening |
| --------- | -------- | --------- | ---------------- |
| baseline  | near 0%  | ~2000 ms (cold prefill every call) | LRU evicts the next agent every cycle |
| treatment | `~K/N` (≈44% if cache holds 7 of 16) | mixed: hits ~200–300 ms, misses ~2000 ms | Policy protects the predicted-next agent and evicts the just-fired one |

`K` is how many full prompts the prefix cache can hold at once — set by
the KV-pool size and the per-prompt token count. The treatment ceiling
is around `K/N` because the current forecast in `run_round_robin.py`
only marks the *previous* agent as low-probability (probability 0.0);
every other agent stays at the neutral 0.5. A smarter forecast that
ramps probability with cycle distance would push the ceiling higher
— that's a follow-up worth doing if you want a louder demo number.

What's load-bearing for the presentation is the **gap**: baseline ~0%
vs treatment ~40%+, with the TTFT delta dominated by cold prefills on
baseline.

If baseline still shows hits, the cache isn't pressured enough — bump
`--num-agents` (16 → 20 → 24) or `--filler-lines` (500 → 700 → 1000) and
re-run. The goal is for baseline to land near 0% scored-round hit rate
before measuring the treatment win.

## Caveats

- **Each call takes longer.** Total wall time per call is ~2–4 s on a 72B
  vs ~0.3 s on a 7B. With `--num-agents 16 --rounds 4` you're looking
  at ~3–5 minutes per run, not 30 seconds.
- **`--hit-threshold-ms 500`** is calibrated to 72B-AWQ on an H100 — a
  warm hit floor is ~150–300 ms because even a no-prefill call still has
  to push the first token through 80 layers. On smaller models you'd use
  a lower threshold; on bigger ones, higher.

## Knobs

| Flag | What it does | Default |
| --- | --- | --- |
| `--mode {baseline,treatment}` | LRU vs agent-aware | required |
| `--num-agents` | Distinct agents in the rotation; main pressure knob | 4 |
| `--rounds` | Full rotations to run | 5 |
| `--filler-lines` | Lines of filler in each agent prompt (~33 tok/line) | 220 |
| `--hit-threshold-ms` | TTFT below this counts as a cache hit | 200 |
| `--warm-up-rounds` | Initial rounds excluded from summary stats | 1 |
| `--max-output-tokens` | Response length per call | 16 |
| `--eviction-window` | The N in "next N turns" (treatment mode) | server default (3) |
| `--eviction-threshold` | Below this probability ⇒ evict (treatment mode) | server default (0.5) |
| `--skip-eviction-stats` | Don't fetch `/v1/agents/eviction_stats` at the end | off |
