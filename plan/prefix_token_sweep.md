# Small prefix-size sweep (~500–1000 prefix tokens)

Sweep `--preamble-lines` to produce prefixes targeting ~500, 600, 700,
800, 900, and 1000 tokens (using the script's `chars // 4` estimator
in `examples/online_serving/agent_prefetch_workflow.py:91`).

| Target tokens | `--preamble-lines` | Est. tokens (chars/4) | Preamble chars |
|---:|---:|---:|---:|
|  500 | 11 |  512 | 2,049 |
|  600 | 13 |  599 | 2,397 |
|  700 | 15 |  686 | 2,745 |
|  800 | 18 |  816 | 3,267 |
|  900 | 20 |  903 | 3,615 |
| 1000 | 22 |  990 | 3,963 |

Note: these prefixes are very small for prefix-cache testing — the
caching wins normally show up at thousands of tokens. Use this sweep
to characterise behaviour at the small end of the curve.

## Commands

Assumes the server is already running on `http://localhost:8000`
serving the script's default model
(`Qwen/Qwen2.5-72B-Instruct-AWQ`). Other workflow flags match the
example you ran earlier.

### ~500 tokens

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 \
    --mode prefetch --preamble-lines 11 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_500tok
```

### ~600 tokens

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 \
    --mode prefetch --preamble-lines 13 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_600tok
```

### ~700 tokens

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 \
    --mode prefetch --preamble-lines 15 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_700tok
```

### ~800 tokens

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 \
    --mode prefetch --preamble-lines 18 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_800tok
```

### ~900 tokens

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 \
    --mode prefetch --preamble-lines 20 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_900tok
```

### ~1000 tokens

```bash
python examples/online_serving/agent_prefetch_workflow.py \
    --base-url http://localhost:8000 \
    --mode prefetch --preamble-lines 22 --variants-per-agent 4 \
    --rounds 1 --prefetch-top-k 20 \
    --plot /tmp/agent_prefetch_1000tok
```

## Run the whole sweep in one shot

```bash
for lines_tok in "11:500" "13:600" "15:700" "18:800" "20:900" "22:1000"; do
    lines="${lines_tok%%:*}"
    tok="${lines_tok##*:}"
    echo "=== ~${tok} tokens (--preamble-lines ${lines}) ==="
    python examples/online_serving/agent_prefetch_workflow.py \
        --base-url http://localhost:8000 \
        --mode prefetch --preamble-lines "${lines}" --variants-per-agent 4 \
        --rounds 1 --prefetch-top-k 20 \
        --plot "/tmp/agent_prefetch_${tok}tok"
done
```

## Tip: reset APC between runs

If you want each run to start from a cold prefix cache, hit the reset
endpoint between iterations (added in this branch):

```bash
curl -X POST http://localhost:8000/v1/agents/reset_prefix_cache
```
