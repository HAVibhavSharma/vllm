# chunked_chat benchmark

Stage-wise driver for the `/v1/chunked_chat/completions` endpoint. Each stage
sends one HTTP request whose prompt is a sandwich:

```
[ static prefix (anchor) | dynamic body (grows per stage) | static suffix (anchor) ]
```

The static chunks stay byte-identical across stages so the server can treat
them as anchors; only the dynamic chunk grows. This mimics an agentic loop
where the system prompt and trailing format spec are fixed while the
conversation history accumulates.

## Layout

```
benchmarks/chunked_chat/
    benchmark_chunked_chat.py
    templates/
        photosynthesis/
            manifest.json
            prefix.txt          # static anchor (chunk index 0)
            suffix.txt          # static anchor (chunk index 2)
            stage_1.txt         # dynamic body, stage 1
            stage_2.txt         # dynamic body, stage 2
            ...
```

`manifest.json` declares the chunk layout, which positions are anchors, and
which stages to run:

```json
{
  "topic": "photosynthesis",
  "chunk_template": ["prefix.txt", "stage_{n}.txt", "suffix.txt"],
  "anchor_indices": [0, 2],
  "stages": [1, 2, 3, 4]
}
```

`{n}` in a filename is replaced with the current stage number. Files without
`{n}` are loaded once per stage but their bytes don't change, so they are the
natural choice for anchors.

## Running

Start a vLLM server that serves a generate-capable model, then:

```bash
python benchmarks/chunked_chat/benchmark_chunked_chat.py \
    --templates-dir benchmarks/chunked_chat/templates/photosynthesis \
    --base-url http://localhost:8000 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-tokens 128 \
    --warmup
```

Output is one line per stage with latency and token usage, followed by a
summary table. Use `--print-output` to dump the assistant's reply for each
stage.

## Adding a new topic

1. Create `templates/<topic>/`.
2. Drop `manifest.json` plus the referenced `.txt` files into it.
3. Pass `--templates-dir benchmarks/chunked_chat/templates/<topic>` on the CLI.

No code changes needed.
