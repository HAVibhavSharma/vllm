# Execution Commands

End-to-end runbook for testing the agent-scoped prefix prefetch
implementation. Run sections **in order** unless noted otherwise.

Assumed paths:

- vLLM repo: `~/Build/KVCOMM-VLLM/vllm`
- Python venv: `~/Build/KVCOMM-VLLM/vllm/.venv`
- LMCache server (separate process) on port `10903`
- vLLM serve on port `8000`
- Model: `Qwen/Qwen3-8B`

---

## 0. Prerequisites (one-time)

```bash
cd ~/Build/KVCOMM-VLLM/vllm
source .venv/bin/activate

# Plotting deps (optional, only for --plot)
.venv/bin/python -m pip install matplotlib
```

---

## 1. Reinstall the Python changes

vLLM is installed editable. All our changes are Python-only, no rebuild
required:

```bash
cd ~/Build/KVCOMM-VLLM/vllm
.venv/bin/python -m pip install -e . --no-deps
```

---

## 2. Start the services

### 2a. LMCache server (terminal A)

```bash
lmcache server \
    --l1-size-gb 20 \
    --eviction-policy LRU \
    --chunk-size 16 \
    --host 0.0.0.0 \
    --port 10903
```

### 2b. vLLM server (terminal B)

```bash
cd ~/Build/KVCOMM-VLLM/vllm

vllm serve Qwen/Qwen3-8B \
    --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

For more diagnostic output on the new prefetch path, prefix with
`VLLM_LOGGING_LEVEL=DEBUG` or pass `--log-level debug`:

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3-8B \
    --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

Log lines to watch for:

```
Finalized prefetch_only request prefetch::agent1::... (N computed tokens, registered in APC)
agent_prefetch: submitted N phantom prefetches for agent agent1
agent_prefetch: recorded prefix for agent agent1 (N tokens after chunk alignment)
agent_prefetch: initialized registry (default_top_k=20, ...)
agent_prefetch: initialized phantom submitter
```

---

## 3. Verify the new route mounted

```bash
curl -s http://localhost:8000/openapi.json \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for path in sorted(data.get('paths', {})):
    if 'agent' in path or 'chat' in path:
        print(path)
"
```

Expected: includes `/v1/agents/chat/completions`.

---

## 4. Unit tests (registry + hashing)

```bash
cd ~/Build/KVCOMM-VLLM/vllm
.venv/bin/python -m pytest tests/v1/agent_prefetch/ -v
```

Expected: all green.

---

## 5. Baseline measurement (no prefetch)

Establishes the cold-tax that the prefetch endpoint is supposed to
eliminate. Two rounds: round 1 is cold, round 2 should hit APC.

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
    --mode baseline \
    --preamble-lines 500 --variants-per-agent 4 \
    --rounds 2 \
    --plot /tmp/agent_baseline
```

Expected:

- Round 1: 9 calls, ~900 ms TTFT each (first visits, cold)
- Round 2: 9 calls, ~60 ms TTFT each (revisits, APC warm)
- Summary: `cold tax ... ≈ +860 ms`
- Summary: `inferred cache-hit rate ... ≈ 50.0%` (9 hits / 18)
- Plots: `/tmp/agent_baseline.timeline.png` + `/tmp/agent_baseline.bars.png`

---

## 6. Smoke test the phantom-prefetch endpoint

Two-pass run. Pass 1 populates the registry (cold). Pass 2 should hit
the prefetch endpoint with a warm registry -- if Commit 2+3 work, the
cold tax collapses.

### Pass 1 -- prime the registry

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
    --mode prefetch \
    --preamble-lines 500 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_pass1
```

Expected: 9 cold calls. `first-visit HITS (prefetch worked): 0` (no
history to prefetch from).

### Pass 2 -- registry is now populated

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
    --mode prefetch \
    --preamble-lines 500 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_pass2
```

Expected: `first-visit HITS (prefetch worked): N` with **N > 0**.
Timeline plot shows short green bars *with black outlines* (first
visits that were already warm thanks to the prefetch).

---

## 7. Comparison runs

### Old warmup-request approach

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
    --mode warmup \
    --preamble-lines 500 --variants-per-agent 4 \
    --rounds 2 \
    --plot /tmp/agent_warmup
```

Expected: every TTFT is warm (~60 ms) because each call's preamble is
warmed by a dummy max_tokens=1 request just before it. Total wall
clock is higher than prefetch mode because of the wasted warmups.

---

## 8. Direct curl smoke (phantom path, no agent endpoint)

If the agent endpoint fails for any reason, this isolates whether the
underlying engine-level phantom-prefetch hook works:

```bash
# First, do a normal chat call so LMCache + APC see the prefix.
curl http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "Qwen/Qwen3-8B",
        "prompt": "The quick brown fox jumps over the lazy dog. ",
        "max_tokens": 8,
        "temperature": 0
    }'

# Then fire a phantom with kv_transfer_params.prefetch_only=true:
curl -w '\n--- %{time_total}s\n' http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "Qwen/Qwen3-8B",
        "prompt": "The quick brown fox jumps over the lazy dog. ",
        "max_tokens": 1,
        "temperature": 0,
        "extra_body": {"kv_transfer_params": {"prefetch_only": true}}
    }'
```

Expected: the phantom returns fast (well under cold-prefill cost), with
`finish_reason="stop"` and essentially empty `text`. The vLLM log
should show a `Finalized prefetch_only request ...` line.

---

## 9. Copy plots off the remote box

```bash
# From your local machine:
scp chisel-8:/tmp/agent_baseline.timeline.png    ./
scp chisel-8:/tmp/agent_baseline.bars.png        ./
scp chisel-8:/tmp/agent_prefetch_pass1.timeline.png ./
scp chisel-8:/tmp/agent_prefetch_pass1.bars.png  ./
scp chisel-8:/tmp/agent_prefetch_pass2.timeline.png ./
scp chisel-8:/tmp/agent_prefetch_pass2.bars.png  ./
```

What the plots should look like:

| Plot | Round 1 bars | Round 2 / pass 2 bars |
|---|---|---|
| baseline.timeline | tall red, outlined | short green, no outline |
| prefetch_pass1.timeline | tall red, outlined | n/a (1 round) |
| prefetch_pass2.timeline | short green, *outlined* | n/a (1 round) -- this outline-on-green pattern is the win |

---

## 10. Server log triage (if something looks wrong)

In the vLLM serve terminal (or via `journalctl` if running as a
service), grep for the agent-prefetch tags:

```bash
# Live tail:
grep -E "agent_prefetch|Finalized prefetch_only|prefetch::" vllm.log

# Most recent 200 lines containing those tags:
grep -E "agent_prefetch|Finalized prefetch_only|prefetch::" vllm.log | tail -200
```

Common failure modes:

| Symptom | Likely cause |
|---|---|
| `404` from `/v1/agents/chat/completions` | router not registered. Confirm `register_agent_chat_api_router(app)` is in `vllm/entrypoints/openai/generate/api_router.py`. |
| `500` with `KeyError: 'engine_client'` | submitter built before chat handler. Check `_get_or_init_state` in `agent_chat/api_router.py`. |
| Pass 2 still shows tall red bars | phantoms aren't firing OR aren't hitting LMCache. Tail the log for `agent_prefetch: submitted N`. If N=0, the registry didn't record from pass 1 -- check for `agent_prefetch: recorded prefix` lines. |
| `assert num_external_computed_tokens > 0` | phantom landed but LMCache had no hits. The submitter is targeting a prefix LMCache evicted. Acceptable in heavy churn; should be rare in this demo. |

---

## 11. Resetting state between runs

To force everything cold (clean APC, fresh registry, fresh LMCache):

```bash
# Clear LMCache.
curl -X POST http://localhost:10903/clear-cache

# Restart vLLM to reset APC + the in-process registry.
# (Stop the vllm serve process in terminal B and start it again from §2b.)
```
